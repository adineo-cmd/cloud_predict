# evaluate.py
# DEPRECATED standalone evaluation script — the canonical pipeline lives in the
# CLI: `python -m cli evaluate` (held-out test split, ensemble, full metrics).
# This shim forwards to it so old habits still work.
import sys

if __name__ == "__main__":
    print("⚠ evaluate.py is deprecated — forwarding to the canonical CLI (python -m cli evaluate)...")
    from cli.evaluate_command import run_evaluate

    split = "validation" if "--validation" in sys.argv else "test"
    run_evaluate(plot_roc="--no-roc" not in sys.argv, split=split)
