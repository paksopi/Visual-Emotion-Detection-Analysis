"""Live webcam demo: runs each Track B VLM in turn against your own webcam
feed, for SECONDS_PER_MODEL seconds per model.

Unlike the Track A CV models, these take the whole frame (no face crop -
that's the point of Track B's scene-context reasoning) and take seconds per
call, so this doesn't run per-frame: it captures one live frame, blocks on
inference, shows the frame + printed answer, then repeats until the time
slice ends. Only one model is loaded onto the GPU at a time (each is
freed before the next loads) to stay inside the 6GB budget.

Florence-2 has no open-ended emotion prompt (see run_florence2.py) so it
runs its native <DETAILED_CAPTION> task instead of the emotion PROMPT.

Run: .venv/Scripts/python src/vlm/live_webcam_vlm_demo.py [seconds_per_model]
Controls: 'q' quits the whole demo early, any other key captures the next frame early.
"""
import gc
import os
import sys
import time
from pathlib import Path

import cv2
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

# Moondream2's remote code hard-imports pyvips, which needs the native
# libvips DLL (not installable via pip on Windows) - point it at the
# prebuilt binary fetched into data/vips/.
_VIPS_BIN = REPO_ROOT / "data" / "vips" / "extracted" / "vips-dev-8.18" / "bin"
if _VIPS_BIN.is_dir():
    os.add_dll_directory(str(_VIPS_BIN))
    os.environ["PATH"] = str(_VIPS_BIN) + os.pathsep + os.environ.get("PATH", "")

from eval import RunLogger  # noqa: E402

SECONDS_PER_MODEL = 30
EMOTION_PROMPT = (
    "Describe the emotional state of the main person in this image and "
    "explain your reasoning using visible details from the whole scene "
    "(posture, environment, objects, activity) - not just their face."
)
# Latency here is decode-bound: cost scales with OUTPUT tokens, not input
# image size (see reports/model_comparison_results.md discussion). Forcing a
# one-word answer instead of a reasoned explanation cuts Qwen2.5-VL from
# ~10-17s to ~1.2s and Moondream2 from ~5s to ~0.9s (measured) - close to
# Florence-2's ~0.6-0.9s native-captioning speed, but with a real emotion
# label Florence-2 cannot produce at all. Trade-off: no scene-grounded
# reasoning/explanation, just the label.
SHORT_EMOTION_PROMPT = (
    "In ONE word, what is the main person's dominant emotion "
    "(angry/happy/sad/neutral/surprised/fearful/disgusted)? Answer with just the word."
)


def make_moondream2_predictor():
    from transformers import AutoModelForCausalLM

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        "vikhyatk/moondream2", revision="2025-01-09", trust_remote_code=True,
        torch_dtype=torch.float16, device_map={"": device},
    )

    def predict(frame_rgb):
        img = Image.fromarray(frame_rgb)
        return model.query(img, EMOTION_PROMPT)["answer"].strip()

    return predict, model


def make_moondream2_fast_predictor():
    """Same model as make_moondream2_predictor, short-answer prompt instead
    of the reasoning prompt - see SHORT_EMOTION_PROMPT note above.
    """
    from transformers import AutoModelForCausalLM

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        "vikhyatk/moondream2", revision="2025-01-09", trust_remote_code=True,
        torch_dtype=torch.float16, device_map={"": device},
    )

    def predict(frame_rgb):
        img = Image.fromarray(frame_rgb)
        return model.query(img, SHORT_EMOTION_PROMPT)["answer"].strip()

    return predict, model


def make_qwen25vl_predictor():
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
    from qwen_vl_utils import process_vision_info

    model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4"
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, quantization_config=bnb_config, device_map="cuda", attn_implementation="sdpa"
    )
    processor = AutoProcessor.from_pretrained(model_id)

    def predict(frame_rgb):
        img = Image.fromarray(frame_rgb)
        messages = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": EMOTION_PROMPT}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=200)
        trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out)]
        return processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()

    return predict, model


def make_qwen25vl_fast_predictor():
    """Same model as make_qwen25vl_predictor, short-answer prompt + a low
    max_new_tokens cap instead of the reasoning prompt - see
    SHORT_EMOTION_PROMPT note above.
    """
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
    from qwen_vl_utils import process_vision_info

    model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4"
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, quantization_config=bnb_config, device_map="cuda", attn_implementation="sdpa"
    )
    processor = AutoProcessor.from_pretrained(model_id)

    def predict(frame_rgb):
        img = Image.fromarray(frame_rgb)
        messages = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": SHORT_EMOTION_PROMPT}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=10)
        trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out)]
        return processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()

    return predict, model


def make_florence2_predictor():
    from transformers import AutoModelForCausalLM, AutoProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Florence-2-base", trust_remote_code=True, torch_dtype=torch.float16
    ).to(device)
    processor = AutoProcessor.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True)
    task = "<DETAILED_CAPTION>"

    def predict(frame_rgb):
        img = Image.fromarray(frame_rgb)
        inputs = processor(text=task, images=img, return_tensors="pt").to(device, torch.float16)
        gen_ids = model.generate(
            input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"],
            max_new_tokens=256, num_beams=3,
        )
        raw_text = processor.batch_decode(gen_ids, skip_special_tokens=False)[0]
        parsed = processor.post_process_generation(raw_text, task=task, image_size=(img.width, img.height))
        return parsed[task]

    return predict, model


MODELS = [
    ("Moondream2", make_moondream2_predictor),
    ("Qwen2.5-VL-3B (4-bit)", make_qwen25vl_predictor),
    ("Florence-2 (native captioning, no emotion prompt)", make_florence2_predictor),
    ("Moondream2 (fast, one-word emotion)", make_moondream2_fast_predictor),
    ("Qwen2.5-VL-3B (fast, one-word emotion)", make_qwen25vl_fast_predictor),
]


def wrap_text(text, width=60):
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return lines


def run_model_live(cap, model_name, predict_fn, seconds, logger):
    start = time.time()
    n_calls = 0
    last_lines = ["..."]
    while time.time() - start < seconds:
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Show the captured frame + a "thinking" indicator before blocking on
        # inference (a call can take 8-18s) so the window doesn't look frozen.
        thinking_frame = frame.copy()
        for i, line in enumerate(last_lines[:6]):
            cv2.putText(thinking_frame, line, (20, 30 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 0), 2)
        cv2.putText(thinking_frame, f"{model_name}  |  thinking...",
                    (20, thinking_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        cv2.imshow("Live VLM Demo", thinking_frame)
        cv2.waitKey(1)

        t0 = time.time()
        try:
            answer = predict_fn(frame_rgb)
        except Exception as e:
            answer = f"error: {e}"
        latency_s = time.time() - t0
        n_calls += 1
        last_lines = wrap_text(answer)
        print(f"[{model_name} #{n_calls}, {latency_s:.1f}s] {answer[:160]}")
        logger.record(elapsed_s=round(time.time() - start, 3), latency_s=round(latency_s, 3), answer=answer)

        for i, line in enumerate(last_lines[:6]):
            cv2.putText(frame, line, (20, 30 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 0), 2)
        remaining = seconds - (time.time() - start)
        cv2.putText(frame, f"{model_name}  |  {remaining:0.1f}s left  |  q = quit",
                    (20, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow("Live VLM Demo", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            return False, n_calls
    return True, n_calls


def main():
    seconds = float(sys.argv[1]) if len(sys.argv) > 1 else SECONDS_PER_MODEL
    # Optional 2nd arg: comma-separated case-insensitive substrings to filter
    # MODELS by name, e.g. `python live_webcam_vlm_demo.py 30 moondream,qwen`
    name_filter = sys.argv[2].lower().split(",") if len(sys.argv) > 2 else None
    models = MODELS if not name_filter else [
        (n, f) for n, f in MODELS if any(sub in n.lower() for sub in name_filter)
    ]

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Could not open webcam (index 0). Check camera permissions/connection.")
        sys.exit(1)

    print(f"Running {len(models)} VLMs, {seconds:.0f}s each. Each call blocks for several seconds. Press 'q' to quit early.")
    try:
        for model_name, make_predictor in models:
            print(f"\n=== {model_name} === loading...")
            try:
                predict_fn, model = make_predictor()
            except Exception as e:
                print(f"Skipping {model_name}: failed to load ({e})")
                continue
            print(f"=== {model_name} === running for {seconds:.0f}s")
            model_key = model_name.split(" ")[0].lower().replace("(", "").replace(")", "")
            logger = RunLogger(model_name=f"{model_key}_live", track="live-webcam")
            try:
                keep_going, n_calls = run_model_live(cap, model_name, predict_fn, seconds, logger)
                out = logger.write_summary({"seconds": seconds, "n_calls": n_calls})
                print(f"summary written to {out}")
            finally:
                logger.close()
            del predict_fn, model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if not keep_going:
                print("Quit requested.")
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
