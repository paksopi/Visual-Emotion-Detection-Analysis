# Visual Emotion Detection Models for Perception Layer

## Executive Summary
This document outlines 14 open-source models suitable for integrating visual emotion detection into local perception layers. The models are evaluated against a strict 6GB VRAM hardware constraint and are divided into two categories: ultra-lightweight Computer Vision (CV) models and deeper Vision-Language Models (VLMs).

## 1. Traditional ML / Computer Vision Models (Lightweight)
These models are highly optimized for fast facial expression recognition. They operate in milliseconds and require cropping the face from the webcam frame prior to inference. They output categorical emotion labels (e.g., Happy, Sad, Neutral) rather than contextual reasoning.

| Model Name | Type | Estimated VRAM Needed | Official Repository / Link |
| :--- | :--- | :--- | :--- |
| **Mini-Xception (FER2013)** | CV / CNN | **< 100 MB** (Can run entirely on CPU) | [oarriaga/face_classification](https://github.com/oarriaga/face_classification) |
| **MediaPipe** | CV / Tracking | **< 100 MB** (CPU/Mobile optimized) | [google-ai-edge/mediapipe](https://github.com/google-ai-edge/mediapipe) |
| **fer** | CV / Python Lib | **< 1 GB** | [justinshenk/fer](https://github.com/justinshenk/fer) |
| **DeepFace** | CV / Python Lib | **~1 GB** (Depends on backend e.g., VGG-Face) | [serengil/deepface](https://github.com/serengil/deepface) |
| **EmotiEffLib (HSEmotion)**| CV / EfficientNet| **~1 GB** (PyTorch / ONNX) | [HSE-asavchenko/face-emotion-recognition](https://github.com/HSE-asavchenko/face-emotion-recognition) |
| **EfficientFace** | CV / ResNet | **~1 GB** | [zengqunzhao/EfficientFace](https://github.com/zengqunzhao/EfficientFace) |
| **Py-Feat** | CV / Toolkit | **~1 GB - 1.5 GB** | [cosanlab/py-feat](https://github.com/cosanlab/py-feat) |
| **OpenFace** | CV / Toolkit | **~1 GB - 2 GB** (Tracks facial muscle Action Units) | [TadasBaltrusaitis/OpenFace](https://github.com/TadasBaltrusaitis/OpenFace) |

## 2. Vision-Language Models (VLMs)
These models analyze the entire scene, providing reasoning and context behind the user's emotional state (e.g., body language, environment). They are significantly heavier and will consume a large portion of a 6GB VRAM budget, reducing concurrency headroom for other models.

| Model Name | Type | Estimated VRAM Needed | Official Repository / Link |
| :--- | :--- | :--- | :--- |
| **Florence-2** | VLM (0.2B - 0.7B) | **0.9 GB - 1.5 GB** | [microsoft/Florence-2-base](https://huggingface.co/microsoft/Florence-2-base) |
| **Qwen2.5-VL-3B** | VLM (3B) | **2.0 GB - 3.0 GB** (Using 4-bit Quantization) | [Qwen/Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) |
| **Moondream2** | VLM (1.8B) | **3.0 GB - 4.0 GB** | [vikhyat/moondream](https://github.com/vikhyat/moondream) |
| **Gemma Multimodal (PaliGemma)**| VLM (3B) | **3.0 GB - 4.0 GB** (Using 4-bit Quantization) | [google/paligemma-3b-pt-224](https://huggingface.co/google/paligemma-3b-pt-224) |
| **MiniCPM-V 2.6** | VLM (8B) | **~5.5 GB - 6.0 GB** (Using 4-bit Quantization) | [OpenBMB/MiniCPM-V](https://github.com/OpenBMB/MiniCPM-V) |
| **LLaVA-1.5-7B** | VLM (7B) | **> 6.0 GB** (Too heavy for a 6GB VRAM limit) | [haotian-liu/LLaVA](https://github.com/haotian-liu/LLaVA) |

## 3. Core Trade-off Summary
* **Go with CV Models if:** The system requires continuous, real-time background monitoring (high frame rate) and needs to leave the GPU entirely free for other primary processes.
* **Go with VLMs if:** The system triggers visual perception selectively (e.g., only when the user speaks) and requires deep, human-like understanding of *why* the user feels a certain way based on their environment.
