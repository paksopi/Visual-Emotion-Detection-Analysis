"""Track B runner: Qwen2.5-VL-3B-Instruct, 4-bit (nf4) quantized per plan §3's
quantization note (this candidate only fits the 6GB budget quantized).
"""
import csv
import sys
from pathlib import Path

import torch
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from eval import LatencyTimer, VRAMTracker, RunLogger  # noqa: E402

IMAGES_DIR = REPO_ROOT / "data" / "track_b" / "images"
GROUND_TRUTH = REPO_ROOT / "data" / "track_b" / "ground_truth.csv"
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

PROMPT = (
    "Describe the emotional state of the main person in this image and "
    "explain your reasoning using visible details from the whole scene "
    "(posture, environment, objects, activity) - not just their face."
)


def main():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4"
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, quantization_config=bnb_config, device_map="cuda"
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    rows = list(csv.DictReader(open(GROUND_TRUTH, encoding="utf-8")))

    logger = RunLogger(model_name="qwen25vl3b_4bit", track="B")
    timer = LatencyTimer()

    with VRAMTracker(use_torch=True) as vram:
        for row in rows:
            image_id = row["image_id"]
            img_path = IMAGES_DIR / f"{image_id}.jpg"
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": str(img_path)},
                        {"type": "text", "text": PROMPT},
                    ],
                }
            ]
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to("cuda")

            with timer.measure():
                with torch.no_grad():
                    out = model.generate(**inputs, max_new_tokens=200)
            trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out)]
            answer = processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()

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
        "quantization": "4-bit nf4",
    }
    out = logger.write_summary(summary)
    logger.close()
    print("latency:", timer.summary())
    print("peak VRAM MB:", vram.peak_mb())
    print("summary written to", out)
    print("raw responses logged to", logger.log_path)


if __name__ == "__main__":
    main()
