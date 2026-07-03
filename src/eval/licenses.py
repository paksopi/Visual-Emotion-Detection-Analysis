"""License registry - encodes reports/license_comparison.md as importable
data so code (the Track C aggregator, model_registry.py adapters) can look
up commercial-use eligibility by license_id rather than drifting from the
prose doc. license_comparison.md stays the authoritative human-readable
source; this file is what code reads. A license_id used on a ModelAdapter
without a matching row here is a loud KeyError, not silent drift.
"""

LICENSE_REGISTRY: dict[str, dict] = {
    "mit": {"name": "MIT", "commercial_ok": True},
    "apache-2.0": {"name": "Apache-2.0", "commercial_ok": True},
    "qwen-research": {"name": "Qwen RESEARCH LICENSE", "commercial_ok": False},
    "openface-academic": {
        "name": "OpenFace Custom Academic/Non-Commercial License (CMU)",
        "commercial_ok": False,
    },
    "openface3-academic": {
        "name": "OpenFace 3.0 Software License Agreement - Academic/Non-Profit Noncommercial Research Use Only (CMU)",
        "commercial_ok": False,
    },
    "gemma": {"name": "Gemma Terms of Use (Google)", "commercial_ok": True},
    "unknown": {"name": "Unknown / unverified", "commercial_ok": False},
}


def lookup(license_id: str) -> dict:
    if license_id not in LICENSE_REGISTRY:
        raise KeyError(
            f"license_id '{license_id}' not in LICENSE_REGISTRY - add it (and update "
            f"reports/license_comparison.md) before registering a model with this id."
        )
    return LICENSE_REGISTRY[license_id]
