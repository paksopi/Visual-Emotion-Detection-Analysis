"""Track C harness CLI driver - single entry point replacing N x M bespoke
scripts: run any registered model through any registered method.

Usage:
    python src/eval/run_method.py --model <registry_key> --method <method_name> [--limit N]

List available keys:
    python src/eval/run_method.py --list
"""
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from eval.methods import METHOD_REGISTRY  # noqa: E402
from eval.model_registry import MODEL_REGISTRY  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="registry key, see --list")
    parser.add_argument("--method", help="method name, see --list")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--list", action="store_true", help="list registered models/methods and exit")
    args = parser.parse_args()

    if args.list or not args.model or not args.method:
        print("Registered models:", ", ".join(sorted(MODEL_REGISTRY)))
        print("Registered methods:", ", ".join(sorted(METHOD_REGISTRY)))
        if not args.list:
            parser.error("--model and --method are required (or pass --list)")
        return

    if args.model not in MODEL_REGISTRY:
        parser.error(f"unknown model '{args.model}'. Choices: {sorted(MODEL_REGISTRY)}")
    if args.method not in METHOD_REGISTRY:
        parser.error(f"unknown method '{args.method}'. Choices: {sorted(METHOD_REGISTRY)}")

    adapter = MODEL_REGISTRY[args.model]
    method = METHOD_REGISTRY[args.method]
    print(f"Running model='{adapter.key}' through method='{method.name}' (limit={args.limit})...")
    result = method.run(adapter, limit=args.limit)
    for k, v in result.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
