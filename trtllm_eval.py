from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tqdm import tqdm  # pyright: ignore[reportMissingImports]

from tensorrt_llm import LLM, SamplingParams  # pyright: ignore[reportMissingImports]
from tensorrt_llm.evaluate import (  # pyright: ignore[reportMissingImports]
    GSM8K,
    GPQADiamond,
    GPQAExtended,
    GPQAMain,
    # LongBenchV1,
    # LongBenchV2,
    MMLU,
    MMMU,
    CnnDailymail,
    JsonModeEval,
)

# ---------------------------------------------------------------------------
# Task registry – every evaluator available in tensorrt_llm.evaluate
# ---------------------------------------------------------------------------

TASK_REGISTRY: dict[str, dict[str, Any]] = {
    # ---- math / reasoning (fast) ------------------------------------------
    "gsm8k": {
        "evaluator_cls": GSM8K,
        "display_name": "GSM8K",
        "default_max_tokens": 256,
    },
    # ---- GPQA variants (fast-medium) --------------------------------------
    "gpqa_diamond": {
        "evaluator_cls": GPQADiamond,
        "display_name": "GPQA Diamond",
        "default_max_tokens": 32768,
    },
    "gpqa_main": {
        "evaluator_cls": GPQAMain,
        "display_name": "GPQA Main",
        "default_max_tokens": 32768,
    },
    "gpqa_extended": {
        "evaluator_cls": GPQAExtended,
        "display_name": "GPQA Extended",
        "default_max_tokens": 32768,
    },
    # ---- knowledge (medium) -----------------------------------------------
    "mmlu": {
        "evaluator_cls": MMLU,
        "display_name": "MMLU",
        "default_max_tokens": 512,
    },
    # ---- multimodal (medium) ----------------------------------------------
    "mmmu": {
        "evaluator_cls": MMMU,
        "display_name": "MMMU",
        "default_max_tokens": 512,
        "evaluator_kwargs": {"is_multimodal": True, "apply_chat_template": True},
    },
    # # ---- long context (slow) ----------------------------------------------
    # "longbench_v1": {
    #     "evaluator_cls": LongBenchV1,
    #     "display_name": "LongBench V1",
    #     "default_max_tokens": 512,
    # },
    # "longbench_v2": {
    #     "evaluator_cls": LongBenchV2,
    #     "display_name": "LongBench V2",
    #     "default_max_tokens": 512,
    # },
    # ---- generation quality -----------------------------------------------
    "cnn_dailymail": {
        "evaluator_cls": CnnDailymail,
        "display_name": "CNN/DailyMail",
        "default_max_tokens": 512,
    },
    "json_mode_eval": {
        "evaluator_cls": JsonModeEval,
        "display_name": "JSON Mode Eval",
        "default_max_tokens": 512,
    },
}

DEFAULT_TASKS = ["mmlu"]

DEFAULT_MODELS = [
    # "/data/models/Qwen3-30B-A3B-NVFP4",
    # "/data/models/Qwen3-30B-A3B-NVFP4-CBINT2",
    "/data/models/Qwen3-30B-A3B-CBINT2",
]

# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


def _evaluate_tasks(
    llm: Any,
    tasks: list[str],
    *,
    apply_chat_template: bool = False,
    max_tokens: int | None = None,
    limit: int | None = None,
    model_label: str = "",
) -> dict[str, dict[str, Any]]:
    """Run *tasks* against an already-loaded ``LLM`` and return per-task results."""

    results: dict[str, dict[str, Any]] = {}

    task_bar = tqdm(
        tasks,
        desc=f"  Tasks ({model_label})" if model_label else "  Tasks",
        unit="task",
        leave=True,
    )

    for task_key in task_bar:
        task_info = TASK_REGISTRY[task_key]
        task_bar.set_postfix_str(task_info["display_name"])

        evaluator_kwargs: dict[str, Any] = {
            "num_samples": limit,
            "apply_chat_template": apply_chat_template,
        }
        evaluator_kwargs.update(task_info.get("evaluator_kwargs", {}))

        evaluator = task_info["evaluator_cls"](**evaluator_kwargs)

        tokens = max_tokens or task_info["default_max_tokens"]
        sampling_params = SamplingParams(max_tokens=tokens, temperature=0)

        score = evaluator.evaluate(llm, sampling_params)
        results[task_key] = {
            "score": score,
            "display_name": task_info["display_name"],
        }

    return results


def run_multi_model_eval(
    model_paths: list[str],
    tasks: list[str],
    *,
    tokenizer: str | None = None,
    tp_size: int = 1,
    pp_size: int = 1,
    limit: int | None = None,
    apply_chat_template: bool = False,
    max_tokens: int | None = None,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Evaluate one or more models on the same task set.

    Returns ``{model_path: {task_key: {"score": float, "display_name": str}}}``.
    """

    all_results: dict[str, dict[str, dict[str, Any]]] = {}

    model_bar = tqdm(model_paths, desc="Models", unit="model")

    for model_path in model_bar:
        model_label = Path(model_path).name
        model_bar.set_postfix_str(model_label)

        tqdm.write(f"\n{'=' * 60}")
        tqdm.write(f"  Loading model: {model_label}")
        tqdm.write(f"{'=' * 60}")

        llm = LLM(
            model=model_path,
            tokenizer=tokenizer,
            tensor_parallel_size=tp_size,
            pipeline_parallel_size=pp_size,
        )

        all_results[model_path] = _evaluate_tasks(
            llm,
            tasks,
            apply_chat_template=apply_chat_template,
            max_tokens=max_tokens,
            limit=limit,
            model_label=model_label,
        )

        llm.shutdown()

    return all_results


# ---------------------------------------------------------------------------
# Table printing
# ---------------------------------------------------------------------------


def print_results_table(
    all_results: dict[str, dict[str, dict[str, Any]]],
    tasks: list[str],
) -> None:
    """Print a bordered comparison table (rows = tasks, columns = models)."""

    model_paths = list(all_results.keys())
    model_labels = [Path(p).name for p in model_paths]

    task_w = max(20, *(len(TASK_REGISTRY[t]["display_name"]) for t in tasks)) + 2
    score_w = max(12, *(len(lbl) for lbl in model_labels)) + 2

    def _sep() -> str:
        parts = ["+" + "-" * (task_w + 2)]
        for _ in model_labels:
            parts.append("+" + "-" * (score_w + 2))
        parts.append("+")
        return "".join(parts)

    def _row(task_cell: str, score_cells: list[str]) -> str:
        parts = [f"| {task_cell:<{task_w}} "]
        for cell in score_cells:
            parts.append(f"| {cell:^{score_w}} ")
        parts.append("|")
        return "".join(parts)

    sep = _sep()

    print(f"\n{sep}")
    print(_row("Task", model_labels))
    print(sep)

    for task_key in tasks:
        display = TASK_REGISTRY[task_key]["display_name"]
        scores: list[str] = []
        for mp in model_paths:
            r = all_results.get(mp, {}).get(task_key, {})
            s = r.get("score")
            scores.append(f"{s:.2f}" if s is not None else "N/A")
        print(_row(display, scores))

    print(sep)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate TensorRT-LLM models on standard benchmarks",
    )

    parser.add_argument(
        "--model-paths",
        type=str,
        nargs="+",
        default=None,
        help=(
            "One or more model paths to evaluate "
            "(default: both Qwen3-30B-A3B variants)"
        ),
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Single model path (kept for backward compatibility)",
    )

    parser.add_argument("--tokenizer", type=str, default=None,
                        help="Tokenizer path (needed when model-path is an engine dir)")
    parser.add_argument("--tasks", type=str, nargs="+", default=None,
                        choices=list(TASK_REGISTRY.keys()))
    parser.add_argument("--tp-size", type=int, default=1,
                        help="Tensor parallelism size")
    parser.add_argument("--pp-size", type=int, default=1,
                        help="Pipeline parallelism size")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max samples per task (default: full dataset)")
    parser.add_argument("--apply-chat-template", action="store_true",
                        help="Apply chat template to prompts")
    parser.add_argument("--max-tokens", type=int, default=None,
                        help="Override default max output tokens per task")
    parser.add_argument("--output-file", type=str, default=None,
                        help="Save JSON results to this path")

    args = parser.parse_args()

    if args.model_paths:
        model_paths = args.model_paths
    elif args.model_path:
        model_paths = [args.model_path]
    else:
        model_paths = DEFAULT_MODELS

    tasks = args.tasks or DEFAULT_TASKS

    all_results = run_multi_model_eval(
        model_paths=model_paths,
        tasks=tasks,
        tokenizer=args.tokenizer,
        tp_size=args.tp_size,
        pp_size=args.pp_size,
        limit=args.limit,
        apply_chat_template=args.apply_chat_template,
        max_tokens=args.max_tokens,
    )

    print_results_table(all_results, tasks)

    if args.output_file is not None:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nSaved to: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
