from __future__ import annotations

import argparse

import torch  # pyright: ignore[reportMissingImports]
from transformers import AutoModelForCausalLM, AutoTokenizer  # pyright: ignore[reportMissingImports]

assert torch.cuda.is_available(), "CUDA is required"


def load_model(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="auto",
    )
    model.eval()
    return model, tokenizer


@torch.inference_mode()
def generate(model, tokenizer, prompt: str, max_new_tokens: int = 100) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(output_ids[0, inputs["input_ids"].shape[-1]:], skip_special_tokens=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    args = parser.parse_args()

    model, tokenizer = load_model(args.model_path)
    print(generate(model, tokenizer, args.prompt, max_new_tokens=args.max_new_tokens))


if __name__ == "__main__":
    main()
