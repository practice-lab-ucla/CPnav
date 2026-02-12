import os
import io
import re
import ast
import json
import base64
import logging
import hashlib
import asyncio
from typing import Any, Dict, Optional, Tuple

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
    ):
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.store = store

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
          prompt_text: the exact input_text sent
          payload: full Responses API request (includes base64 image_url)
          payload_log: sanitized payload for logging (no base64)
        """
        prompt_text = self._construct_prompt(goal, prompt_type, num_actions)

        base64_png = self._encode_rgb_np_to_base64_png(rgb)
        image_url = f"data:image/png;base64,{base64_png}"

        prompt_msg = {
            "role": "user",
            "content": [
                {"type": "input_text", "text": prompt_text},
                {"type": "input_image", "image_url": image_url},
            ],
        }

        payload = {
            "model": self.model,
            "input": [prompt_msg],
            "store": self.store,
        }

        # Sanitized log (don’t store base64)
        payload_log = {
            "model": self.model,
            "store": self.store,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt_text},
                        {
                            "type": "input_image",
                            "image_meta": {
                                "shape": list(rgb.shape),
                                "dtype": str(rgb.dtype),
                                "base64_sha256": self._sha256_str(base64_png),
                            },
                        },
                    ],
                }
            ],
        }

        return prompt_text, payload, payload_log

    async def ask_gpt_async(
        self,
        rgb: np.ndarray,
        goal: str,
        prompt_type: str,
        num_actions: int,
        timeout_s: Optional[float] = None,  # left unused for parity
    ) -> Tuple[str, Dict[str, Any], Dict[str, Any], str]:
        """
        Returns:
          text: raw model output text
          data: parsed dict (same logic as before)
          payload_log: sanitized request payload for logging
          prompt_text: the exact prompt text
        """
        prompt_text, payload, payload_log = self._build_request(
            rgb=rgb,
            goal=goal,
            prompt_type=prompt_type,
            num_actions=num_actions,
        )

        resp = await self.client.responses.create(**payload)
        text = (resp.output_text or "").strip()

        if prompt_type == "stopping":
            data = eval_vlm_response_stopping(text)
            if "done" in data:
                try:
                    data["done"] = int(data["done"])
                except Exception:
                    pass
            if "global_semantic_score" in data:
                try:
                    data["global_semantic_score"] = float(data["global_semantic_score"])
                except Exception:
                    pass
            return text, data, payload_log, prompt_text

        # action
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

        if "confidence_score" in data:
            try:
                data["confidence_score"] = [float(score) for score in data["confidence_score"]]
            except Exception:
                pass

        return text, data, payload_log, prompt_text

    @staticmethod
    def _construct_prompt(goal: str, prompt_type: str, num_actions: int=0):
        if prompt_type == 'stopping':
            stopping_prompt = (
                f"The agent has been tasked with navigating to a {goal.upper()}. The agent has sent you an image from its current location."
                f"The image is a fusion of three views from one position at different angles."
                f"Your job is to decide if the agent is VERY CLOSE from a {goal}, and you have to CLEARLY see the goal with very high confidence, based ONLY on what is VISIBLE in the image."
                f"Important: a chair is NOT a sofa, a sofa is NOT a bed, a plant MUST be inside the room. Do NOT infer the {goal} from the room type or context.\n"

                f"Step 1: Describe what is visible in the image and state explicitly whether a {goal} is present.\n"

                f"Step 2: Choose an action and output it in the format {{\"done\": <1 or 0>}}."
                f"- Return 1 ONLY if the {goal} is clearly visible and not far from it."
                f"- Return 0 if the {goal} is not visible or you are uncertain.\n"

                f"Step 3: Independently, rate the SCENE'S EXPLORATION POTENTIAL as a float in [0.0, 1.0], named global_semantic_score."
                f"This score MUST depend only on the current environment, NOT on whether the goal is present or visible."
                f"High scores mean the scene has open, traversable, informative paths."
                f"Low scores mean likely dead-ends, cluttered/tight spaces, blocked passages, or no promising directions.\n"

                f"Important rules for global_semantic_score:"
                f"- Do NOT increase the score just because the {goal} is visible."
                f"- Base it on openness, navigability cues, line of sight, and apparent paths.\n"

                f"Examples:"
                f"0.0 to 0.1 → the view is completely blocked, directly facing a wall, with CLEARLY NO navigable path\n"
                f"0.1 to 0.3 → the view has no clear outlet, close to a wall, or almost blocked\n"
                f"0.3 to 0.7 → the view has a clear outlet or large navigable space (the higher the score, the clearer and more navigable it looks)\n"
                f"0.7 to 1.0 → the view has multiple outlets, corridors, or very large navigable space to navigate\n"
                f"After Step 3, immediately output this JSON line:"
                f"{{\"global_semantic_score\": <float 0.0 to 1.0>}}"
            )
            return stopping_prompt
        if prompt_type == 'action':
            action_prompt = (
                f"TASK: NAVIGATE TO THE NEAREST {goal.upper()}, and get as close to it as possible. "
                f"Use your prior knowledge about where items are typically located within a home. "
                f"There are {num_actions+1} actions that you can choose from (including action 0). "
                f"Actions are shown with red arrows superimposed onto your observation, labeled with numbers in white circles. "
                f"The image is a fusion of three views from one position at different angles."
                f"{'NOTE: Choose action 0 (rewind) if you want to rewind to a previous step or do not see anything worth exploring '}"
                f"First, tell me what you see in your sensor observation, and if you have any leads on finding the {goal.upper()}. "
                f"Second, tell me which general direction you should go in. "
                f"Lastly, explain which action achieves that best and return it as JSON in the format: "
                f"{{'action': <action_key>, 'score': <confidence_score>, 'confidence_score': [<score_0>, <score_1>, ..., <score_n>]}}. "
                f"The 'confidence_score' list represents probabilities for each action "
                f"'action' must be an integer not a string and an independent confidence value in [0, 1]  "
                f"Do NOT normalize or force the scores to sum to 1. "
                f"You must generate exactly {num_actions+1} confidence scores, one for each action shown. "
                f"{'If Action 0 (rewind) is available, its confidence score must appear first in the list, followed by Action 1, Action 2, etc.' if True else 'The scores should be listed in order: Action 1, Action 2, Action 3, and so on.'}"

            )
            return action_prompt

    @staticmethod
    def normalize_scores(confidence_scores):
        """
        Normalizes a list of confidence scores to ensure they sum exactly to 1.
        """
        total = sum(confidence_scores)
        
        if total == 0:
            return [1.0 / len(confidence_scores)] * len(confidence_scores)
        
        # Normalize scores
        normalized_scores = [s / total for s in confidence_scores]
        normalized_scores = [round(n, 3) for n in normalized_scores]  # round to # ofdecimal places to avoid long numbers

        return normalized_scores
    
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
