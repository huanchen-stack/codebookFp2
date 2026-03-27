"""Download nvidia/Qwen3-30B-A3B-NVFP4 from HuggingFace."""

import argparse
from huggingface_hub import snapshot_download


def main():
    parser = argparse.ArgumentParser(description="Download NVFP4 model from HuggingFace")
    parser.add_argument(
        "--model-id",
        type=str,
        default="nvidia/Qwen3-30B-A3B-NVFP4",
        help="HuggingFace model ID",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./models/Qwen3-30B-A3B-NVFP4",
        help="Local directory to save the model",
    )
    args = parser.parse_args()

    print(f"Downloading {args.model_id} to {args.output_dir}...")
    snapshot_download(repo_id=args.model_id, local_dir=args.output_dir)
    print(f"Done. Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
