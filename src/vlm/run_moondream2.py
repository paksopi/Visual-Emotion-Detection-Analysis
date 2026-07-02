"""Track B runner: Moondream2 (vikhyat/moondream), fp16 on GPU.

Fits comfortably in the 6GB budget at fp16 (1.9B params, no quantization
needed here per plan §3's quantization note, which is aimed at the larger
candidates). Uses the model's `.query()` VQA interface with a fixed prompt
asking for emotion + scene-grounded reasoning, matching the rubric dimensions
in reports/evaluation_plan.md §3 (emotion correctness, contextual grounding,
hallucination, usefulness are scored separately in a later rubric pass —
this script only captures latency/VRAM + raw model output per image).
"""
import csv
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForCausalLM

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from eval import LatencyTimer, VRAMTracker, RunLogger  # noqa: E402

IMAGES_DIR = REPO_ROOT / "data" / "track_b" / "images"
GROUND_TRUTH = REPO_ROOT / "data" / "track_b" / "ground_truth.csv"

PROMPT = (
    "Describe the emotional state of the main person in this image and "
    "explain your reasoning using visible details from the whole scene "
    "(posture, environment, objects, activity) - not just their face."
)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        "vikhyatk/moondream2",
        revision="2025-01-09",
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map={"": str(device)},
    )

    rows = list(csv.DictReader(open(GROUND_TRUTH, encoding="utf-8")))

    logger = RunLogger(model_name="moondream2", track="B")
    timer = LatencyTimer()

    with VRAMTracker(use_torch=True) as vram:
        for row in rows:
            image_id = row["image_id"]
            img_path = IMAGES_DIR / f"{image_id}.jpg"
            img = Image.open(img_path).convert("RGB")
            with timer.measure():
                result = model.query(img, PROMPT)
            answer = result["answer"].strip()
            logger.record(
                image_id=image_id,
                ground_truth=row["ground_truth"],
                raw_response=answer,
            )
            print(f"[{image_id}] {answer[:120]}...")

    summary = {
        "latency": timer.summary(),
        "peak_vram_mb": vram.peak_mb(),
        "n_images": len(rows),
        "prompt": PROMPT,
    }
    out = logger.write_summary(summary)
    logger.close()
    print("latency:", timer.summary())
    print("peak VRAM MB:", vram.peak_mb())
    print("summary written to", out)
    print("raw responses logged to", logger.log_path)


if __name__ == "__main__":
    main()
