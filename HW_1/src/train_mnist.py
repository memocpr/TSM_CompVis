# This script was removed per request (TensorFlow not installed). Use src/model.py utilities when TF is available.

# Minimal CLI wrapper to load or train an MNIST model using src.model utilities.

from pathlib import Path
import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="Minimal trainer/loader for MNIST model.")
    parser.add_argument("--model-path", "-m", default=str(Path(__file__).parent / "mnist_model"),
                        help="Path to load/save the model (directory or file).")
    args = parser.parse_args()

    try:
        # Prefer package-style import (works when running as module or package is on PYTHONPATH)
        from src.model import load_or_train_default  # type: ignore
    except Exception:
        # Fallback: if the script is executed directly (python train_mnist.py), try loading model.py from the same folder
        try:
            import importlib.util
            model_path = Path(__file__).parent / "model.py"
            spec = importlib.util.spec_from_file_location("hw1_src_model", str(model_path))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)  # type: ignore
            load_or_train_default = module.load_or_train_default
        except Exception as e:
            print("Failed to import load_or_train_default from src.model or local model.py:", e, file=sys.stderr)
            print("Install TensorFlow and ensure package import path is set, or run as a module (python -m HW_1.src.train_mnist).", file=sys.stderr)
            sys.exit(1)

    model_path = Path(args.model_path)
    # Ensure parent directory exists for file-style paths
    if not model_path.parent.exists():
        model_path.parent.mkdir(parents=True, exist_ok=True)

    # This function will load the model if present, otherwise train a small model and save it.
    model = load_or_train_default(str(model_path))
    print(f"Model ready at: {model_path}")


if __name__ == "__main__":
    main()
