# Track A — self-collected occlusion/lighting stress test

Robustness check, not an accuracy benchmark: real webcam frames under conditions FER2013 doesn't cover (dim lighting, partial occlusion, off-angle), with no emotion ground truth. Reports face-detection success and predicted label per model per condition.

Frames per condition: 3. Conditions: baseline, dim_light, occluded, off_angle.

## Face-detection rate by condition

| Condition | Frames | Face detected |
|---|---|---|
| baseline | 3 | 2/3 |
| dim_light | 3 | 3/3 |
| occluded | 3 | 0/3 |
| off_angle | 3 | 1/3 |

## Predicted label by model per condition (face-detected frames only)

| Condition | fer | deepface | hsemotion | efficientface |
|---|---|---|---|---|
| baseline | sad, neutral | sad, angry | neutral, neutral | neutral, neutral |
| dim_light | neutral, neutral, happy | angry, angry, neutral | neutral, neutral, neutral | neutral, neutral, neutral |
| occluded | no face detected | no face detected | no face detected | no face detected |
| off_angle | sad | sad | neutral | surprise |

## Raw per-frame results

```json
[
  {
    "condition": "baseline",
    "path": "C:\\Users\\faids\\Documents\\PersonalGitHub\\Visual Emotion Detection\\data\\track_a_stress\\baseline\\00.jpg",
    "face_detected": true,
    "predictions": {
      "fer": "sad",
      "deepface": "sad",
      "hsemotion": "neutral",
      "efficientface": "neutral"
    }
  },
  {
    "condition": "baseline",
    "path": "C:\\Users\\faids\\Documents\\PersonalGitHub\\Visual Emotion Detection\\data\\track_a_stress\\baseline\\01.jpg",
    "face_detected": false,
    "predictions": {}
  },
  {
    "condition": "baseline",
    "path": "C:\\Users\\faids\\Documents\\PersonalGitHub\\Visual Emotion Detection\\data\\track_a_stress\\baseline\\02.jpg",
    "face_detected": true,
    "predictions": {
      "fer": "neutral",
      "deepface": "angry",
      "hsemotion": "neutral",
      "efficientface": "neutral"
    }
  },
  {
    "condition": "dim_light",
    "path": "C:\\Users\\faids\\Documents\\PersonalGitHub\\Visual Emotion Detection\\data\\track_a_stress\\dim_light\\00.jpg",
    "face_detected": true,
    "predictions": {
      "fer": "neutral",
      "deepface": "angry",
      "hsemotion": "neutral",
      "efficientface": "neutral"
    }
  },
  {
    "condition": "dim_light",
    "path": "C:\\Users\\faids\\Documents\\PersonalGitHub\\Visual Emotion Detection\\data\\track_a_stress\\dim_light\\01.jpg",
    "face_detected": true,
    "predictions": {
      "fer": "neutral",
      "deepface": "angry",
      "hsemotion": "neutral",
      "efficientface": "neutral"
    }
  },
  {
    "condition": "dim_light",
    "path": "C:\\Users\\faids\\Documents\\PersonalGitHub\\Visual Emotion Detection\\data\\track_a_stress\\dim_light\\02.jpg",
    "face_detected": true,
    "predictions": {
      "fer": "happy",
      "deepface": "neutral",
      "hsemotion": "neutral",
      "efficientface": "neutral"
    }
  },
  {
    "condition": "occluded",
    "path": "C:\\Users\\faids\\Documents\\PersonalGitHub\\Visual Emotion Detection\\data\\track_a_stress\\occluded\\00.jpg",
    "face_detected": false,
    "predictions": {}
  },
  {
    "condition": "occluded",
    "path": "C:\\Users\\faids\\Documents\\PersonalGitHub\\Visual Emotion Detection\\data\\track_a_stress\\occluded\\01.jpg",
    "face_detected": false,
    "predictions": {}
  },
  {
    "condition": "occluded",
    "path": "C:\\Users\\faids\\Documents\\PersonalGitHub\\Visual Emotion Detection\\data\\track_a_stress\\occluded\\02.jpg",
    "face_detected": false,
    "predictions": {}
  },
  {
    "condition": "off_angle",
    "path": "C:\\Users\\faids\\Documents\\PersonalGitHub\\Visual Emotion Detection\\data\\track_a_stress\\off_angle\\00.jpg",
    "face_detected": false,
    "predictions": {}
  },
  {
    "condition": "off_angle",
    "path": "C:\\Users\\faids\\Documents\\PersonalGitHub\\Visual Emotion Detection\\data\\track_a_stress\\off_angle\\01.jpg",
    "face_detected": true,
    "predictions": {
      "fer": "sad",
      "deepface": "sad",
      "hsemotion": "neutral",
      "efficientface": "surprise"
    }
  },
  {
    "condition": "off_angle",
    "path": "C:\\Users\\faids\\Documents\\PersonalGitHub\\Visual Emotion Detection\\data\\track_a_stress\\off_angle\\02.jpg",
    "face_detected": false,
    "predictions": {}
  }
]
```
