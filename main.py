import argparse
import os
import sys

import config.settings as settings
from model.generation import generate_text
from model.model_utils import load_model, load_tokenizer
from training.trainer import prepare_dataset, train_model
from utils.device import get_device, show_device_info
from utils.file import clean_temp_dir


def _apply_device_flags(args: argparse.Namespace) -> None:
    """Apply global device-related flags to settings and environment."""
    settings.USE_CPU = getattr(args, "use_cpu", False)
    settings.OPTIMIZE_MEMORY = getattr(args, "optimize_memory", False)

    if settings.USE_CPU:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
        print("⚠️ CPU mode enabled. All GPUs (CUDA and MPS) will be disabled.")


def _train_command(args: argparse.Namespace) -> None:
    """Handle `python main.py train`."""
    _apply_device_flags(args)
    device = get_device()

    tokenizer = load_tokenizer()
    model = load_model(force_train=True, use_original=args.use_original)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    if args.clean:
        clean_temp_dir()

    print("🔁 Training model...")
    tokenized_dataset, data_collator = prepare_dataset(tokenizer)
    train_model(model, tokenized_dataset, data_collator, tokenizer)
    print("✅ Training completed!")


def _generate_command(args: argparse.Namespace) -> None:
    """Handle `python main.py generate`."""
    if not args.prompt:
        print("Error: Prompt is required for generate command")
        sys.exit(1)

    _apply_device_flags(args)
    device = get_device()

    tokenizer = load_tokenizer()
    model = load_model(force_train=False, use_original=args.use_original)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    model_type = "original" if args.use_original else "fine-tuned"
    print(f"\n📜 Prompt: {args.prompt}")
    print(f"\n📘 Generated Text ({model_type} model):\n")
    print(generate_text(model, tokenizer, args.prompt, device))


def _serve_command(args: argparse.Namespace) -> None:
    """Handle `python main.py serve`."""
    try:
        from inference.server import run_server
    except ImportError as exc:
        print(f"Error importing inference server: {exc}")
        sys.exit(1)

    _apply_device_flags(args)
    run_server(host=args.host, port=args.port)


def _dataset_command(args: argparse.Namespace) -> None:
    """Handle `python main.py dataset`."""
    try:
        from datasets.loader import list_datasets, preview_dataset
    except ImportError as exc:
        print(f"Error importing datasets module: {exc}")
        sys.exit(1)

    if args.action == "list":
        list_datasets()
    elif args.action == "preview":
        if not args.name:
            print("Error: --name is required for dataset preview")
            sys.exit(1)
        preview_dataset(args.name, limit=args.limit)


def parse_args() -> argparse.Namespace:
    """Define top-level CLI and subcommands."""
    parser = argparse.ArgumentParser(
        description="EduSLM - Educational Small Language Model Training Framework"
    )
    parser.add_argument(
        "--show-device-info",
        action="store_true",
        help="Show detailed device information and exit",
    )

    subparsers = parser.add_subparsers(dest="command", required=False)

    # Train command
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean temporary directory before training",
    )
    train_parser.add_argument(
        "--use-original",
        action="store_true",
        help="Use the original pre-trained model without fine-tuning",
    )
    train_parser.add_argument(
        "--optimize-memory",
        action="store_true",
        help="Apply device-specific memory optimizations",
    )
    train_parser.add_argument(
        "--use-cpu",
        action="store_true",
        help="Force CPU usage",
    )

    # Generate command
    gen_parser = subparsers.add_parser("generate", help="Generate text from a prompt")
    gen_parser.add_argument("prompt", nargs="?", help="Prompt text for generation")
    gen_parser.add_argument(
        "--use-original",
        action="store_true",
        help="Use the original pre-trained model without fine-tuning",
    )
    gen_parser.add_argument(
        "--optimize-memory",
        action="store_true",
        help="Apply device-specific memory optimizations",
    )
    gen_parser.add_argument(
        "--use-cpu",
        action="store_true",
        help="Force CPU usage",
    )

    # Serve command
    serve_parser = subparsers.add_parser(
        "serve", help="Start the FastAPI inference server"
    )
    serve_parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host interface to bind the server (default: 0.0.0.0)",
    )
    serve_parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind the server (default: 8000)",
    )
    serve_parser.add_argument(
        "--optimize-memory",
        action="store_true",
        help="Apply device-specific memory optimizations",
    )
    serve_parser.add_argument(
        "--use-cpu",
        action="store_true",
        help="Force CPU usage",
    )

    # Dataset command
    dataset_parser = subparsers.add_parser(
        "dataset", help="Inspect and manage datasets"
    )
    dataset_parser.add_argument(
        "action",
        choices=["list", "preview"],
        help="Dataset action to perform",
    )
    dataset_parser.add_argument(
        "--name",
        help="Dataset name or path (for preview)",
    )
    dataset_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of samples to show when previewing",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.show_device_info and not args.command:
        # Show device info and exit early
        _apply_device_flags(args)
        show_device_info()
        return

    if not args.command:
        print(
            "Error: a command is required. "
            "Use one of: train, generate, serve, dataset or --help for usage."
        )
        sys.exit(1)

    if args.command == "train":
        _train_command(args)
    elif args.command == "generate":
        _generate_command(args)
    elif args.command == "serve":
        _serve_command(args)
    elif args.command == "dataset":
        _dataset_command(args)
    else:
        print(f"Unknown command: {args.command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
