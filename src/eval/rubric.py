"""Track B qualitative rubric, per reports/evaluation_plan.md section 3."""
from dataclasses import dataclass, field

RUBRIC_DIMENSIONS = [
    "emotion_correctness",
    "contextual_grounding",
    "hallucination_count",
    "response_usefulness",
]


@dataclass
class RubricScore:
    image_id: str
    model: str
    emotion_correctness: int  # 1-5
    contextual_grounding: int  # 1-5
    hallucination_count: int  # raw count, not 1-5
    response_usefulness: int  # 1-5
    raw_response: str = ""
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "image_id": self.image_id,
            "model": self.model,
            "emotion_correctness": self.emotion_correctness,
            "contextual_grounding": self.contextual_grounding,
            "hallucination_count": self.hallucination_count,
            "response_usefulness": self.response_usefulness,
            "raw_response": self.raw_response,
            "notes": self.notes,
        }
