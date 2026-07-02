"""Track B "native capability" runner: Florence-2-base.

Florence-2 is confirmed "wrong checkpoint" in the survey doc (README's
Emotion-capability verification table) -- but unlike PaliGemma, there's no
"use the -mix checkpoint instead" fix available: Florence-2-base is a
task-token model (<CAPTION>, <OD>, <DENSE_REGION_CAPTION>, ...), not an
instruction/chat model, and there's no larger/instruction-tuned Florence-2
checkpoint to swap in. Forcing an open-ended emotion prompt through it
degenerates into garbage (confirmed: '<VQA>What emotion is the person
feeling?' decodes to 'QA>Emotion', not a real answer -- its VQA task token
only supports fixed closed-set questions, not free text).

So this runs it on its actual native task instead -- dense scene captioning
-- against the same Track B image set the other VLMs use, to see what it's
actually proficient at (231M params, well within the 6GB budget; it's a
checkpoint-capability mismatch, not a VRAM one).
"""
import csv
import sys
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from eval import LatencyTimer, VRAMTracker, RunLogger  # noqa: E402

IMAGES_DIR = REPO_ROOT / "data" / "track_b" / "images"
GROUND_TRUTH = REPO_ROOT / "data" / "track_b" / "ground_truth.csv"

NATIVE_TASK = "<DETAILED_CAPTION>"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Florence-2-base", trust_remote_code=True, torch_dtype=torch.float16
    ).to(device)
    processor = AutoProcessor.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True)
    n_params = sum(p.numel() for p in model.parameters())

    rows = list(csv.DictReader(open(GROUND_TRUTH, encoding="utf-8")))

    logger = RunLogger(model_name="florence2", track="B-native")
    timer = LatencyTimer()

    with VRAMTracker(use_torch=True) as vram:
        for row in rows:
            image_id = row["image_id"]
            img = Image.open(IMAGES_DIR / f"{image_id}.jpg").convert("RGB")
            inputs = processor(text=NATIVE_TASK, images=img, return_tensors="pt").to(device, torch.float16)
            with timer.measure():
                gen_ids = model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=256,
                    num_beams=3,
                )
            raw_text = processor.batch_decode(gen_ids, skip_special_tokens=False)[0]
            parsed = processor.post_process_generation(raw_text, task=NATIVE_TASK, image_size=(img.width, img.height))
            caption = parsed[NATIVE_TASK]
            logger.record(image_id=image_id, ground_truth=row["ground_truth"], caption=caption)
            print(f"[{image_id}] {caption[:120]}...")

    summary = {
        "capability": "native task-token dense captioning (no open-ended prompting -- see docstring)",
        "task_token_used": NATIVE_TASK,
        "n_params_m": round(n_params / 1e6, 1),
        "latency": timer.summary(),
        "peak_vram_mb": vram.peak_mb(),
        "n_images": len(rows),
        "note": (
            "Not scored against the Track B emotion rubric -- Florence-2-base cannot take an "
            "open-ended emotion prompt at all (confirmed: '<VQA>What emotion...' degenerates to "
            "garbled output, not a real answer). This captures what it's actually good at: dense, "
            "detailed scene captioning, which a downstream system could still use as a cheap "
            "(231M param) scene-description signal even without direct emotion reasoning."
        ),
    }
    out = logger.write_summary(summary)
    logger.close()
    print("latency:", timer.summary())
    print("peak VRAM MB:", vram.peak_mb())
    print("summary written to", out)


if __name__ == "__main__":
    main()
