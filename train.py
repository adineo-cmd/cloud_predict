# train.py
# DEPRECATED standalone training script — the canonical, seeded pipeline lives in
# the CLI: `python -m cli train` (or `python run.py train`).
# This shim forwards to it so old habits still work, with the same ensemble,
# split and calibration behavior as everything else.
import sys

if __name__ == "__main__":
    print("⚠ train.py is deprecated — forwarding to the canonical CLI (python -m cli train)...")
    from cli.train_command import run_train

    # Minimal flag passthrough: --epochs N, --batch-size N, --lr F, --no-finetune, --seed N
    epochs = batch_size = lr = None
    finetune, seed = None, 42
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--epochs" and i + 1 < len(args):
            epochs = int(args[i + 1]); i += 2
        elif args[i] == "--batch-size" and i + 1 < len(args):
            batch_size = int(args[i + 1]); i += 2
        elif args[i] == "--lr" and i + 1 < len(args):
            lr = float(args[i + 1]); i += 2
        elif args[i] == "--seed" and i + 1 < len(args):
            seed = int(args[i + 1]); i += 2
        elif args[i] == "--no-finetune":
            finetune = False; i += 1
        else:
            i += 1
    run_train(epochs=epochs, batch_size=batch_size, learning_rate=lr,
              finetune=finetune, seed=seed)
