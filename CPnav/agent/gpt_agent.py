import os
import io
import re
import ast
import json
import base64
import logging
import hashlib
import asyncio
from typing import Any, Dict, Optional, Tuple, Literal

import numpy as np
from PIL import Image
from openai import AsyncOpenAI


# Types for readability (optional)
StopResult = Tuple[str, Dict[str, Any], Dict[str, Any], str]    # (text, data, payload_log, prompt_text)
ActionResult = Tuple[str, Dict[str, Any], Dict[str, Any], str]


class GPTAgent:
    """
    Uses the OpenAI Responses API with input_image (ASYNC).
    """

    def __init__(
        self,
        model: str = "gpt-5-nano",
        api_key: str = None,
        store: bool = False,
        mode: Literal["cp_nav", "simple_set", "prompt_set", "ensemble_set"]="cp_nav"
    ):
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.store = store
        self.mode = mode

    @staticmethod
    def _encode_rgb_np_to_base64_png(rgb: np.ndarray) -> str:
        pil = Image.fromarray(rgb)
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    @staticmethod
    def _sha256_str(s: str) -> str:
        return hashlib.sha256(s.encode("utf-8")).hexdigest()

    def _build_request(
        self,
        rgb: np.ndarray,
        goal: str,
        prompt_type: str,
        num_actions: int,
    ) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
        """
        Returns:
        prompt_text: the exact user input_text sent (string)
        payload: full Responses API request (includes base64 image_url)
        payload_log: sanitized payload for logging (no base64)
        """
        # _construct_prompt now returns (system_prompt_or_None, user_prompt)
        system_prompt, user_prompt = self._construct_prompt(goal, prompt_type, num_actions)

        base64_png = self._encode_rgb_np_to_base64_png(rgb)
        image_url = f"data:image/png;base64,{base64_png}"

        # Build the list of messages to send
        input_messages = []

        if system_prompt is not None:
            system_msg = {
                "role": "system",
                "content": [
                    {"type": "input_text", "text": system_prompt}
                ],
            }
            input_messages.append(system_msg)

        # user message always present
        prompt_msg = {
            "role": "user",
            "content": [
                {"type": "input_text", "text": user_prompt},
                {"type": "input_image", "image_url": image_url},
            ],
        }
        input_messages.append(prompt_msg)

        payload = {
            "model": self.model,
            "input": input_messages,
            "store": self.store,
        }

        # Sanitized log (don’t store base64)
        payload_log_input = []

        if system_prompt is not None:
            payload_log_input.append({
                "role": "system",
                "content": [
                    {"type": "input_text", "text": system_prompt}
                ]
            })

        payload_log_input.append({
            "role": "user",
            "content": [
                {"type": "input_text", "text": user_prompt},
                {
                    "type": "input_image",
                    "image_meta": {
                        "shape": list(rgb.shape),
                        "dtype": str(rgb.dtype),
                        "base64_sha256": self._sha256_str(base64_png),
                    },
                },
            ],
        })

        payload_log = {
            "model": self.model,
            "store": self.store,
            "input": payload_log_input,
        }

        # Return the user prompt text (so existing code that expects prompt_text keeps working)
        return user_prompt, payload, payload_log

    async def ask_gpt_async(
        self,
        rgb: np.ndarray,
        goal: str,
        prompt_type: str,
        num_actions: int,
        timeout_s: Optional[float] = None,
    ) -> Tuple[str, Dict[str, Any], Dict[str, Any], str]:
        
        prompt_text, payload, payload_log = self._build_request(
            rgb=rgb, goal=goal, prompt_type=prompt_type, num_actions=num_actions
        )

        # Handle Ensemble Logic for Actions
        if self.mode == "ensemble_set" and prompt_type == "action":
            return await self._get_ensemble_scores(payload, payload_log, prompt_text, num_actions)

        # Standard Single Query Logic
        resp = await self.client.responses.create(**payload)
        text = (resp.output_text or "").strip()

        if prompt_type == "stopping":
            data = eval_vlm_response_stopping(text)
            return text, data, payload_log, prompt_text

        # Standard action parsing
        data = self._parse_action_data(text)
        return text, data, payload_log, prompt_text

    async def _get_ensemble_scores(
        self, 
        payload: Dict[str, Any], 
        payload_log: Dict[str, Any], 
        prompt_text: str, 
        num_actions: int, 
        n_trials: int = 20
    ) -> Tuple[str, Dict[str, Any], Dict[str, Any], str]:
        """
        Queries the VLM 20 times concurrently and computes action frequency.
        """
        # Run 20 requests in parallel for efficiency
        tasks = [self.client.responses.create(**payload) for _ in range(n_trials)]
        responses = await asyncio.gather(*tasks)
        
        action_counts = [0] * (num_actions + 1)
        valid_responses = 0
        last_text = ""

        for resp in responses:
            text = (resp.output_text or "").strip()
            last_text = text # Keep the last one for the 'text' return value
            data = eval_vlm_response(text)
            
            # Extract action index
            action_idx = data.get("action")
            if action_idx is None:
                action_idx = data.get("action_index")
            
            try:
                idx = int(action_idx)
                if 0 <= idx <= num_actions:
                    action_counts[idx] += 1
                    valid_responses += 1
            except (ValueError, TypeError):
                continue

        # Calculate frequencies (Confidence Scores)
        # If no valid responses, default to 0.0
        conf_scores = [
            count / valid_responses if valid_responses > 0 else 0.0 
            for count in action_counts
        ]
        
        # The 'best' action is the one with the highest frequency
        best_action = int(np.argmax(conf_scores))
        
        ensemble_data = {
            "action": best_action,
            "score": conf_scores[best_action],
            "confidence_scores": conf_scores,
            "trials": n_trials,
            "valid_samples": valid_responses
        }

        return last_text, ensemble_data, payload_log, prompt_text

    def _parse_action_data(self, text: str) -> Dict[str, Any]:
        """Helper to centralize the parsing/casting logic for actions."""
        data = eval_vlm_response(text)
        if "action" in data:
            try:
                data["action"] = int(data["action"])
            except Exception:
                pass

        if "action" not in data and "action_index" in data:
            try:
                data["action"] = int(data["action_index"])
            except Exception:
                data["action"] = data["action_index"]

        if "confidence_scores" in data:
            try:
                data["confidence_scores"] = [float(score) for score in data["confidence_scores"]]
            except Exception:
                pass
        if "action_set" in data:
            try:
                data["action_set"] = [int(action_id) for action_id in data["action_set"]]
            except Exception:
                pass
        if "action" not in data and "action_index" in data:
            data["action"] = data["action_index"]
        try:
            if "action" in data: data["action"] = int(data["action"])
        except: pass
        return data

    def _construct_prompt(
            self, 
            goal: str, 
            prompt_type: Literal["stopping", "action"], 
            num_actions: int=0):

        system_prompt = (
            "You are an advanced Embodied Visual Navigation Agent. "
            "Your sole purpose is to navigate a physical environment to locate a specified target object using RGB camera input.\n\n"
            
            "### CORE CAPABILITIES\n"
            "1. **Visual Recognition:** You can identify household objects and distinguish between similar items (e.g., differentiating a sofa from a bed).\n"
            "2. **Spatial Reasoning:** You use common sense about home layouts to guide your search (e.g., searching for a bed implies looking for a bedroom, not a kitchen).\n"
            "3. **Structured Output:** You must strictly follow the requested JSON schemas for every response. Do not output conversational filler.\n\n"
            
            "### OPERATIONAL MODES\n"
            "You will be prompted in one of two states:\n"
            "- **State A (Analysis):** You will assess if the goal is found and if the area is navigable.\n"
            "- **State B (Action):** You will see an image with navigational overlays (labeled arrows) and must output a move decision.\n\n"
            
            "Maintain a logical, objective, and probabilistic mindset. Prioritize exploration of new areas over backtracking unless the path is blocked."
        )

        if prompt_type == 'stopping':
            stopping_prompt = (
                f"### CONTEXT\n"
                f"You are a navigation assistant for an agent searching for a {goal.upper()}. "
                f"The input is a camera view from the agent's current position.\n\n"

                f"### STEP 1: VISUAL ANALYSIS\n"
                f"Describe the visible environment. State explicitly if a {goal.upper()} is present. "
                f"**Strict Identification Rules:**\n"
                f"- A chair is NOT a sofa; a sofa is NOT a bed.\n"
                f"- Do NOT infer the {goal.upper()} exists based on room type (e.g., don't assume a bed is there just because it's a bedroom).\n"
                f"- You must see the object with high confidence to confirm its presence.\n\n"

                f"### STEP 2: GOAL VERIFICATION (JSON)\n"
                f"Determine if the agent has found {goal.upper()}. Output the following JSON:\n"
                f"{{\n"
                f"  \"done\": <1 or 0>\n"
                f"}}\n"
                f"* Set to **1** ONLY if the {goal.upper()} is clearly visible and within 2 meters reach.\n"
                f"* Set to **0** if the {goal.upper()} is not visible, or you are uncertain.\n\n"

                f"### STEP 3: EXPLORATION POTENTIAL (JSON)\n"
                f"Independently rate the environment's 'Global Semantic Score' (0.0 to 1.0). "
                f"**This score must be based ONLY on navigability, NOT on the presence of the {goal.upper()}.**\n\n"

                f"**Scoring Rubric:**\n"
                f"* **0.0 - 0.2:** View is blocked (e.g., facing a wall), cluttered, or has no navigable paths.\n"
                f"* **0.3 - 0.6:** A clear outlet exists or there is a modest navigable space.\n"
                f"* **0.7 - 1.0:** High potential; multiple corridors, open doorways, or large unobstructed spaces.\n\n"

                f"Output the score in this format:\n"
                f"{{\n"
                f"  \"global_semantic_score\": <float>\n"
                f"}}"
            )
            return system_prompt, stopping_prompt
        if prompt_type == 'action' and (self.mode in ["cp_nav", "simple_set"]):
            action_prompt = (
                f"### TASK\n"
                f"Choose the path to move close to the nearest {goal.upper()} "
                f"Use common sense regarding typical home layouts to guide your search.\n\n"

                f"### VISUAL INPUT\n"
                f"The provided image is a views from your current position. "
                f"Available paths are indicated by red arrows with white numeric labels (0 to {num_actions}).\n\n"

                f"### DECISION RULES\n"
                f"* **Action 0 (Turn Around):** Select this if you need to backtrack or if no visible path leads toward the {goal.upper()}.\n"
                f"* **Exploration:** Prioritize directions that likely lead to rooms where a {goal.upper()} is typically found.\n\n"

                f"### RESPONSE FORMAT\n"
                f"Provide your response in three parts:\n"
                f"1. **Observation:** Describe the current scene and identify any visual leads for the {goal.upper()}.\n"
                f"2. **Strategy:** State which general direction is most promising and why.\n"
                f"3. **Action (JSON):** Output a JSON object exactly as follows:\n"
                f"{{\n"
                f"  \"action\": <int: best_action_index>,\n"
                f"  \"score\": <float_highest_score>,\n"
                f"  \"confidence_scores\": [<score_0>, ..., <score_{num_actions}>]\n"
                f"}}\n\n"

                f"### CONSTRAINTS\n"
                f"* Provide exactly {num_actions + 1} scores in the 'confidence_scores' list, ordered sequentially from Action 0 to Action {num_actions}.\n"
                f"* Each score must be between 0 and 1. Do NOT normalize (they do not need to sum to 1).\n"
                f"* The 'score' must represent the estimated probability (0.0 to 1.0) that the selected action will successfully lead toward the {goal.upper()}."
            )
        elif prompt_type == 'action' and (self.mode in ["prompt_set"]):
            action_prompt = (
                f"### TASK\n"
                f"Choose the path to move close to the nearest {goal.upper()} "
                f"Use common sense regarding typical home layouts to guide your search.\n\n"

                f"### VISUAL INPUT\n"
                f"The provided image is a views from your current position. "
                f"Available paths are indicated by red arrows with white numeric labels (0 to {num_actions}).\n\n"

                f"### DECISION RULES\n"
                f"* **Action 0 (Turn Around):** Select this if you need to backtrack or if no visible path leads toward the {goal.upper()}.\n"
                f"* **Exploration:** Prioritize directions that likely lead to rooms where a {goal.upper()} is typically found.\n\n"

                f"### RESPONSE FORMAT\n"
                f"Provide your response in three parts:\n"
                f"1. **Observation:** Describe the current scene and identify any visual leads for the {goal.upper()}.\n"
                f"2. **Strategy:** State which general direction is most promising and why.\n"
                f"3. **Action (JSON):** Output a JSON object exactly as follows:\n"
                f"{{\n"
                f"  \"action\": <int: best_action_index>,\n"
                f"  \"action_set\": [<int: highest_priority>, <int: second_priority>, ...]\n"
                f"}}\n\n"

                f"### CONSTRAINTS\n"
                f"- 'action_set' must include the indices of all paths (you can also include backtrack) that may move you closer to the goal, sorted from most to least promising.\n"
                f"- Do not include any text after the JSON block."
            )
        elif prompt_type == 'action' and (self.mode in ["ensemble_set"]):
            action_prompt = (
                f"### TASK\n"
                f"Choose the path to move close to the nearest {goal.upper()} "
                f"Use common sense regarding typical home layouts to guide your search.\n\n"

                f"### VISUAL INPUT\n"
                f"The provided image is a views from your current position. "
                f"Available paths are indicated by red arrows with white numeric labels (0 to {num_actions}).\n\n"

                f"### DECISION RULES\n"
                f"* **Action 0 (Turn Around):** Select this if you need to backtrack or if no visible path leads toward the {goal.upper()}.\n"
                f"* **Exploration:** Prioritize directions that likely lead to rooms where a {goal.upper()} is typically found.\n\n"

                f"### RESPONSE FORMAT\n"
                f"1. **Observation:** Describe the current scene and identify any visual leads for the {goal.upper()}.\n"
                f"2. **Strategy:** State which general direction is most promising and why.\n"
                f"3. **Action (JSON):** Output a JSON object exactly as follows:\n"
                f"{{\n"
                f"  \"action\": <int: best_action_index>,\n"
                f"}}\n\n"

                f"### CONSTRAINTS\n"
                f"- Do not include any text after the JSON block."
            )
        
        return system_prompt, action_prompt 
    
    async def query_stop_and_action(
        self,
        obs_rgb,
        overlay_rgb,
        goal: str,
        num_actions: int,
        timeout_s: Optional[float] = None,
    ) -> Tuple[StopResult, ActionResult]:
        stop_coroutine = self.ask_gpt_async(
            rgb=obs_rgb,
            goal=goal,
            prompt_type="stopping",
            num_actions=0,               # ignored for stopping in your code
            timeout_s=timeout_s,
        )

        action_coroutine = self.ask_gpt_async(
            rgb=overlay_rgb,
            goal=goal,
            prompt_type="action",
            num_actions=num_actions,
            timeout_s=timeout_s,
        )

        # Run both concurrently
        stop_res, action_res = await asyncio.gather(stop_coroutine, action_coroutine)
        return stop_res, action_res


def clean_json_string(json_str: str) -> str:
    """
    Removes C-style comments (// and /* */) from a JSON-like string.
    """
    # Remove single line comments // ...
    json_str = re.sub(r"//.*", "", json_str)
    # Remove multi-line comments /* ... */
    json_str = re.sub(r"/\*[\s\S]*?\*/", "", json_str)
    return json_str.strip()

def eval_vlm_response(response: str) -> dict:
    try:
        # 1. Extract the JSON block
        start = response.rindex("{")
        end = response.rindex("}") + 1
        json_block = response[start:end]
        
        # 2. Clean comments
        clean_block = clean_json_string(json_block)
        
        # 3. Use json.loads (standard for JSON)
        try:
            return json.loads(clean_block)  # try proper JSON first
        except json.JSONDecodeError:
            return ast.literal_eval(clean_block)  # fallback for Python dict
    except (ValueError, IndexError) as e:
        logging.error(f"Error parsing response: {e}")
        print()
        print(response)
        print()
        return {}

def eval_vlm_response_stopping(response: str) -> dict:
    """
    Support two JSON snippets in one message:
      Step 2: {"done": 0/1}                 → take the FIRST done
      Step 3: {"global_semantic_score": x}  → take the LAST score

    (This matches Agent._eval_response_stopping in agent-2.py)
    """
    merged = {}
    gsv_last = None
    done_first = None
    done_conflict = False

    for block in re.findall(r"\{[\s\S]*?\}", response):
        try:
            obj = json.loads(block)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue

        # lock first 'done'
        if "done" in obj:
            try:
                d = int(obj["done"])
                if done_first is None:
                    done_first = d
                elif done_first != d:
                    done_conflict = True
            except Exception:
                pass

        # keep last gsv
        if "global_semantic_score" in obj:
            try:
                gsv_last = float(obj["global_semantic_score"])
            except Exception:
                pass

        merged.update(obj)

    if done_first is not None:
        merged["done"] = done_first
    if gsv_last is not None:
        merged["global_semantic_score"] = gsv_last

    # coerce types
    if "done" in merged:
        try:
            merged["done"] = int(merged["done"])
        except Exception:
            pass
    if "global_semantic_score" in merged:
        try:
            merged["global_semantic_score"] = float(merged["global_semantic_score"])
        except Exception:
            pass

    if done_conflict:
        merged["_warning"] = "conflicting_done_values_detected"

    return merged
