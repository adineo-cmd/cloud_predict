# cloud.py
# DEPRECATED: the original standalone "all bugs fixed" training + prediction script.
# The canonical pipeline now lives in the CLI — it adds seeded stratified splits,
# a configurable backbone, a real seed ensemble, a held-out test split and
# validation-only calibration, which this file never had:
#
#   python -m cli train | predict | evaluate | calibrate
#   python run.py          (one-command pipeline)
#
# Kept as a shim so old invocations keep working; previous divergent behavior
# (positional 20% validation split, single MobileNetV2, no test set) is retired.
import sys

if __name__ == "__main__":
    print("⚠ cloud.py is deprecated — forwarding to the canonical CLI.")
    from cli.main import main

    args = sys.argv[1:]
    if not args or args[0] in ("train", "--train"):
        sys.argv = [sys.argv[0]] + ["train"]
    elif args and args[0] == "predict":
        sys.argv = [sys.argv[0]] + ["predict"] + args[1:]
    else:
        sys.argv = [sys.argv[0]] + args
    main()
