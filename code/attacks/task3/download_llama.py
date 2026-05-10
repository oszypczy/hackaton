#!/usr/bin/env python3
"""Download Llama-2-7B + Llama-3-8B (NousResearch mirror, ungated).

Llama is the canonical Kirchenbauer-watermark generator → its PPL on
watermarked text should differ characteristically from clean.
"""
import argparse
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--model", default="NousResearch/Llama-2-7b-hf",
                    help="HF model id (use NousResearch mirror to bypass auth)")
    ap.add_argument("--also-3", action="store_true", help="also download Llama-3-8B")
    args = ap.parse_args()

    os.environ["HF_HOME"] = args.cache_dir
    os.environ["TRANSFORMERS_OFFLINE"] = "0"
    os.environ["HF_DATASETS_OFFLINE"] = "0"

    from huggingface_hub import snapshot_download

    models = [args.model]
    if args.also_3:
        models.append("NousResearch/Meta-Llama-3-8B")

    for m in models:
        print(f"\nDownloading {m} -> {args.cache_dir}/hub")
        try:
            path = snapshot_download(
                repo_id=m,
                cache_dir=os.path.join(args.cache_dir, "hub"),
                allow_patterns=["*.json", "*.safetensors", "tokenizer.model", "*.txt"],
                max_workers=4,
            )
            print(f"  OK: {path}")
        except Exception as e:
            print(f"  FAIL: {e}")
            sys.exit(1)
    print("\nAll downloads complete.")


if __name__ == "__main__":
    main()
