from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch  # pyright: ignore[reportMissingImports]
import lm_eval  # pyright: ignore[reportMissingImports]
from lm_eval.models.huggingface import HFLM  # pyright: ignore[reportMissingImports]
from transformers import AutoModelForCausalLM, AutoTokenizer  # pyright: ignore[reportMissingImports]

assert torch.cuda.is_available(), "CUDA is required"

TASK_REGISTRY = {
    "gsm8k": {
        "lm_eval_task": "gsm8k_cot",
        "primary_metric": "exact_match,strict-match",
        "display_name": "GSM8k-CoT (8-shot)",
    },
    "hellaswag": {
        "lm_eval_task": "hellaswag",
        "primary_metric": "acc_norm,none",
        "display_name": "HellaSwag (0-shot)",
    },
    "winogrande": {
        "lm_eval_task": "winogrande",
        "primary_metric": "acc,none",
        "display_name": "WinoGrande (0-shot)",
    },
}

DEFAULT_TASKS = ["gsm8k", "hellaswag", "winogrande"]


def extract_metrics(eval_results: dict, task_key: str) -> dict:
    task_info = TASK_REGISTRY[task_key]
    lm_task = task_info["lm_eval_task"]
    task_results = eval_results["results"].get(lm_task, {})
    return {k: v for k, v in task_results.items() if isinstance(v, (int, float)) and k != "alias"}


def run_eval(
    model_path: str,
    tasks: list[str],
    batch_size: int | str = "auto",
    limit: int | None = None,
    num_fewshot: int | None = None,
) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="auto",
    )
    model.eval()

    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size)

    lm_eval_tasks = [TASK_REGISTRY[t]["lm_eval_task"] for t in tasks]
    eval_results = lm_eval.simple_evaluate(
        model=lm,
        tasks=lm_eval_tasks,
        num_fewshot=num_fewshot,
        batch_size=batch_size,
        limit=limit,
        log_samples=False,
    )

    results = {}
    for task_key in tasks:
        task_info = TASK_REGISTRY[task_key]
        all_metrics = extract_metrics(eval_results, task_key)
        results[task_key] = {
            "primary_metric": all_metrics.get(task_info["primary_metric"]),
            "primary_metric_name": task_info["primary_metric"],
            "display_name": task_info["display_name"],
            "all_metrics": all_metrics,
        }

    return results


def print_results(results: dict, tasks: list[str]) -> None:
    header = f"{'Task':<25} {'Metric':<30} {'Score':>10}"
    print(header)
    print("-" * len(header))
    for task_key in tasks:
        r = results.get(task_key, {})
        display = r.get("display_name", task_key)
        metric_name = r.get("primary_metric_name", "n/a")
        val = r.get("primary_metric")
        val_str = f"{val:.4f}" if val is not None else "N/A"
        print(f"{display:<25} {metric_name:<30} {val_str:>10}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--tasks", type=str, nargs="+", default=None, choices=list(TASK_REGISTRY.keys()))
    parser.add_argument("--batch-size", type=str, default="auto")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-fewshot", type=int, default=None)
    parser.add_argument("--output-file", type=str, default=None)
    args = parser.parse_args()

    tasks = args.tasks or DEFAULT_TASKS

    results = run_eval(
        model_path=args.model_path,
        tasks=tasks,
        batch_size=args.batch_size,
        limit=args.limit,
        num_fewshot=args.num_fewshot,
    )

    print_results(results, tasks)

    if args.output_file is not None:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
