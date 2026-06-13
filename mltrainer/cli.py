from __future__ import annotations

import argparse

from mltrainer.core.downloader import download_all_datasets, download_dataset, ensure_dataset_available
from mltrainer.core.model_registry import DATASET_SPECS
from mltrainer.core.optimizer import optimize_dataset
from mltrainer.core.storage import overwrite_best_model, read_metrics, update_metrics, write_training_outputs
from mltrainer.core.trainer import train_dataset
from mltrainer.ui.console import (
    print_error,
    print_info,
    print_list_models,
    print_optimization_summary,
    print_results_table,
    print_training_summary,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mltrainer", description="ML Trainer CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list-models", help="List supported models")

    dataset_choices = sorted(DATASET_SPECS)

    download_parser = subparsers.add_parser("download", help="Download dataset files")
    download_group = download_parser.add_mutually_exclusive_group(required=True)
    download_group.add_argument("--dataset", choices=dataset_choices)
    download_group.add_argument("--all", action="store_true")

    train_parser = subparsers.add_parser("train", help="Train all models for a dataset")
    train_parser.add_argument("--dataset", required=True, choices=dataset_choices)

    compare_parser = subparsers.add_parser("compare", help="Compare saved model results")
    compare_parser.add_argument("--dataset", required=True, choices=dataset_choices)

    optimize_parser = subparsers.add_parser("optimize", help="Optimize the recommended model")
    optimize_parser.add_argument("--dataset", required=True, choices=dataset_choices)
    optimize_parser.add_argument("--method", choices=["grid", "random"], default="random")

    return parser


def run_list_models() -> int:
    print_list_models()
    return 0


def run_train(dataset: str) -> int:
    ensure_dataset_available(dataset, progress_callback=print_info)
    metrics, model = train_dataset(dataset)
    metrics_file, model_file = write_training_outputs(dataset, metrics, model)
    print_training_summary(metrics, str(metrics_file), str(model_file))
    return 0


def run_compare(dataset: str) -> int:
    metrics = read_metrics(dataset)
    print_results_table(metrics, title="Saved Training Results")
    return 0


def run_optimize(dataset: str, method: str) -> int:
    metrics = read_metrics(dataset)
    ensure_dataset_available(dataset, progress_callback=print_info)
    optimization, best_model = optimize_dataset(dataset, method)
    metrics["optimization"] = optimization
    update_metrics(dataset, metrics)
    model_file = overwrite_best_model(dataset, best_model)
    print_optimization_summary(dataset, optimization, str(model_file))
    return 0


def run_download(dataset: str | None, download_all: bool) -> int:
    if download_all:
        downloaded = download_all_datasets()
        print_info("已完成所有資料集下載。")
        for path in downloaded:
            print_info(f"已建立：{path}")
        return 0

    downloaded = download_dataset(dataset)
    print_info(f"已下載 `{dataset}` 到：{downloaded}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "list-models":
            return run_list_models()
        if args.command == "download":
            return run_download(getattr(args, "dataset", None), getattr(args, "all", False))
        if args.command == "train":
            return run_train(args.dataset)
        if args.command == "compare":
            return run_compare(args.dataset)
        if args.command == "optimize":
            return run_optimize(args.dataset, args.method)
        parser.error(f"Unknown command: {args.command}")
    except FileNotFoundError as exc:
        print_error(str(exc))
        return 1
    except Exception as exc:
        print_error(str(exc))
        return 1
    return 0
