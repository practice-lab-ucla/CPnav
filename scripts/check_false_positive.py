import argparse
import asyncio
import base64
import json
import os
import random
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List, Dict
from tqdm import tqdm

from openai import AsyncOpenAI

from dotenv import load_dotenv
load_dotenv(dotenv_path=".env")


# ----------------------------
# Helpers
# ----------------------------

FILENAME_RE = re.compile(r"^(?P<episode>\d+)_(?P<goal>.+)\.png$", re.IGNORECASE)

def copy_false_positive_images(folder: Path, bad_episode_ids: List[int], fp_dir: Path) -> int:
    """
    Copy images whose episode_id is in bad_episode_ids into fp_dir, preserving filename.
    Returns number of copied images.
    """
    fp_dir.mkdir(parents=True, exist_ok=True)
    bad_set = set(bad_episode_ids)

    copied = 0
    for p in folder.glob("*.png"):
        parsed = parse_episode_goal(p)
        if not parsed:
            continue
        episode_id, _goal = parsed
        if episode_id in bad_set:
            dst = fp_dir / p.name
            shutil.copy2(p, dst)
            copied += 1

    return copied


def parse_episode_goal(p: Path) -> Optional[Tuple[int, str]]:
    """
    Expect: {episode_id}_{goal}.png
    Returns (episode_id, goal) or None if filename doesn't match.
    """
    m = FILENAME_RE.match(p.name)
    if not m:
        return None
    episode_id = int(m.group("episode"))
    goal = m.group("goal").strip()
    if not goal:
        return None
    return episode_id, goal

def resolve_output_json_path(folder: Path, save_json_arg: str) -> Path:
    """
    - If save_json_arg empty: {folder}/fp_detection/results.json
    - If save_json_arg is a directory: {save_json_arg}/results.json
    - If save_json_arg is a file path: use it
    """
    if not save_json_arg:
        out_dir = folder / "fp"
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir / "results.json"

    p = Path(save_json_arg)

    # If user passed a directory (existing dir OR endswith /)
    if p.exists() and p.is_dir():
        p.mkdir(parents=True, exist_ok=True)
        return p / "results.json"

    # Heuristic: if it has a suffix, treat as file, else treat as dir
    if p.suffix.lower() == ".json":
        p.parent.mkdir(parents=True, exist_ok=True)
        return p
    else:
        p.mkdir(parents=True, exist_ok=True)
        return p / "results.json"


def encode_image_data_url(image_path: Path) -> str:
    """
    Encode image bytes into a data URL for Responses API input_image.
    PNG expected by your naming convention.
    """
    b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return f"data:image/png;base64,{b64}"

def extract_json_object(text: str) -> dict:
    """
    Best-effort JSON extraction when the model returns extra text.
    We ask for JSON-only, but this makes the script robust.
    """
    text = text.strip()
    # If already valid JSON
    try:
        return json.loads(text)
    except Exception:
        pass

    # Try to find the first {...} block
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        return json.loads(candidate)

    raise ValueError(f"Could not parse JSON from model output: {text[:200]}...")

async def with_backoff(coro_fn, *, max_retries: int = 5, base_delay: float = 0.8):
    """
    Retry wrapper for transient errors / rate limits.
    """
    for attempt in range(max_retries + 1):
        try:
            return await coro_fn()
        except Exception as e:
            if attempt == max_retries:
                raise
            # jittered exponential backoff
            delay = base_delay * (2 ** attempt) + random.uniform(0, 0.25)
            await asyncio.sleep(delay)


# ----------------------------
# Core verification call
# ----------------------------

@dataclass
class CheckResult:
    episode_id: int
    goal: str
    found: bool
    confidence: float
    raw: Dict

async def check_one_image(
    client: AsyncOpenAI,
    image_path: Path,
    episode_id: int,
    goal: str,
    model: str,
) -> CheckResult:
    """
    Ask the model if the image contains the goal object.
    Returns a structured result.
    """
    data_url = encode_image_data_url(image_path)

    instructions = (
        "You are verifying dataset labels.\n"
        "Decide whether the image contains the object described by `goal`.\n"
        "Be conservative: if the object is not clearly present, answer found=false.\n"
        "Return ONLY valid JSON with keys: found (bool), confidence (0..1), reason (string).\n"
        "No extra text."
    )

    user_text = (
        f"goal: {goal}\n"
        "Question: Does this image contain the goal object?"
    )

    async def _call():
        # Responses API with image input blocks. :contentReference[oaicite:2]{index=2}
        resp = await client.responses.create(
            model=model,
            instructions=instructions,
            input=[{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": user_text},
                    {"type": "input_image", "image_url": data_url},
                ],
            }],
        )
        return resp

    resp = await with_backoff(_call)

    # SDK convenience field aggregates output text. :contentReference[oaicite:3]{index=3}
    out_text = (resp.output_text or "").strip()
    obj = extract_json_object(out_text)

    found = bool(obj.get("found", False))
    conf = obj.get("confidence", 0.0)
    try:
        conf = float(conf)
    except Exception:
        conf = 0.0

    return CheckResult(
        episode_id=episode_id,
        goal=goal,
        found=found,
        confidence=max(0.0, min(1.0, conf)),
        raw=obj,
    )


# ----------------------------
# Runner
# ----------------------------

async def run(folder: Path, model: str, concurrency: int) -> List[CheckResult]:
    client = AsyncOpenAI()
    sem = asyncio.Semaphore(concurrency)

    pngs = sorted(folder.glob("*.png"))
    tasks = []

    for p in pngs:
        parsed = parse_episode_goal(p)
        if not parsed:
            continue
        episode_id, goal = parsed

        async def _guarded(path=p, ep=episode_id, g=goal):
            async with sem:
                return await check_one_image(client, path, ep, g, model)

        tasks.append(asyncio.create_task(_guarded()))

    results: List[CheckResult] = []

    pbar = tqdm(total=len(tasks), desc="Checking images", unit="img")
    for t in asyncio.as_completed(tasks):
        try:
            results.append(await t)
        except Exception as e:
            results.append(CheckResult(
                episode_id=-1, goal="__ERROR__", found=False, confidence=0.0,
                raw={"error": str(e)}
            ))
        finally:
            pbar.update(1)
    pbar.close()

    return results



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", type=str, required=True, help="Folder containing {episode_id}_{goal}.png")
    ap.add_argument("--model", type=str, default="gpt-5-mini",
                    help="Vision-capable model for checking (default: gpt-5-mini)")
    ap.add_argument("--concurrency", type=int, default=20, help="Max concurrent requests")
    ap.add_argument("--min_confidence", type=float, default=0.0,
                    help="Optional: flag as not-found if confidence < this threshold (even if found=true)")
    args = ap.parse_args()

    folder = Path(args.folder)
    if not folder.exists():
        raise SystemExit(f"Folder does not exist: {folder}")

    # 1. Run the async check
    results = asyncio.run(run(folder, args.model, args.concurrency))

    # 2. Filter: Identify False Positives (Goal not found or low confidence)
    bad_episode_ids = []
    for r in results:
        if r.episode_id < 0:
            continue
        if (not r.found) or (r.confidence < args.min_confidence):
            bad_episode_ids.append(r.episode_id)

    # Sort and remove duplicates
    bad_episode_ids = sorted(list(set(bad_episode_ids)))

    # 3. Handle output directories
    fp_dir = folder / "fp"
    fp_dir.mkdir(parents=True, exist_ok=True)

    # 4. Write the False Positive Index List to the experiment folder
    fp_index_path = folder / "false_positive.txt"
    fp_index_path.write_text(json.dumps(bad_episode_ids, indent=2))
    
    # 5. Copy images and save full results
    n_copied = copy_false_positive_images(folder, bad_episode_ids, fp_dir)
    
    print(f"\n--- Summary ---")
    print(f"False Positive IDs: {bad_episode_ids}")
    print(f"Total FP found: {len(bad_episode_ids)}")
    print(f"Images copied to: {fp_dir}")
    print(f"Index list saved to: {fp_index_path}")

    # Save the full detailed results log
    out_path = resolve_output_json_path(folder, "")
    detailed_out = [
        {
            "episode_id": r.episode_id,
            "goal": r.goal,
            "found": r.found,
            "confidence": r.confidence,
            "raw": r.raw,
        }
        for r in results
    ]
    out_path.write_text(json.dumps(detailed_out, indent=2))
    print(f"Detailed log saved to: {out_path}")



if __name__ == "__main__":
    main()
