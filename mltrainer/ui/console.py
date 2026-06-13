from __future__ import annotations

from colorama import just_fix_windows_console
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from mltrainer.core.model_registry import DATASET_SPECS

just_fix_windows_console()
console = Console()


def print_header(title: str) -> None:
    console.print(Panel.fit(f"[bold magenta]{title}[/bold magenta]", border_style="cyan"))


def print_list_models() -> None:
    print_header("ML Trainer CLI")
    table = Table(title="Supported Models", header_style="bold cyan")
    table.add_column("Dataset", style="yellow")
    table.add_column("Task", style="green")
    table.add_column("Models", style="white")
    table.add_column("Ranking Metric", style="magenta")
    table.add_column("Recommended Optimize Model", style="blue")
    for spec in DATASET_SPECS.values():
        table.add_row(
            spec.name,
            spec.task,
            ", ".join(spec.model_names.keys()),
            spec.ranking_metric.upper(),
            spec.model_names[spec.recommended_model_key],
        )
    console.print(table)


def print_training_summary(metrics: dict[str, object], metrics_file: str, model_file: str) -> None:
    print_results_table(metrics, title="Training Results")
    best_model = metrics["best_model"]
    console.print(f"[bold green]Best model:[/bold green] {best_model['name']} ({best_model['key']})")
    console.print(f"[cyan]metrics.json[/cyan]: {metrics_file}")
    console.print(f"[cyan]best_model.joblib[/cyan]: {model_file}")


def print_results_table(metrics: dict[str, object], title: str) -> None:
    print_header("ML Trainer CLI")
    console.print(f"[bold]Dataset:[/bold] {metrics['dataset']}")
    console.print(f"[bold]Task:[/bold] {metrics['task']}")
    console.print(f"[bold]Ranking metric:[/bold] {metrics['ranking_metric']}")

    table = Table(title=title, header_style="bold cyan")
    table.add_column("Rank", justify="right")
    table.add_column("Key", style="yellow")
    table.add_column("Model", style="white")

    task = metrics["task"]
    if task == "classification":
        columns = [("accuracy", "Accuracy"), ("precision", "Precision"), ("recall", "Recall"), ("f1", "F1-score")]
    else:
        columns = [("mae", "MAE"), ("mse", "MSE"), ("rmse", "RMSE"), ("r2", "R2")]

    for _, label in columns:
        table.add_column(label, justify="right")

    for model in metrics["models"]:
        row = [str(model["rank"]), model["key"], model["name"]]
        for key, _ in columns:
            row.append(f"{model['metrics'][key]:.4f}")
        table.add_row(*row)
    console.print(table)


def print_optimization_summary(dataset: str, optimization: dict[str, object], model_file: str) -> None:
    print_header("ML Trainer CLI")
    console.print(f"[bold]Dataset:[/bold] {dataset}")
    console.print(f"[bold]Optimized model:[/bold] {optimization['recommended_model']}")
    console.print(f"[bold]Method:[/bold] {optimization['method']}")
    console.print(f"[bold]CV best score:[/bold] {optimization['best_score']:.4f}")

    params_table = Table(title="Best Parameters", header_style="bold cyan")
    params_table.add_column("Parameter", style="yellow")
    params_table.add_column("Value", style="white")
    for key, value in optimization["best_params"].items():
        params_table.add_row(str(key), str(value))
    console.print(params_table)

    metrics_table = Table(title="Optimized Test Metrics", header_style="bold cyan")
    metrics_table.add_column("Metric", style="yellow")
    metrics_table.add_column("Value", justify="right")
    for key, value in optimization["test_metrics"].items():
        if key == "confusion_matrix":
            metrics_table.add_row(key, str(value))
        else:
            metrics_table.add_row(key, f"{value:.4f}")
    console.print(metrics_table)
    console.print(f"[cyan]best_model.joblib[/cyan]: {model_file}")


def print_error(message: str) -> None:
    console.print(f"[bold red]Error:[/bold red] {message}")


def print_info(message: str) -> None:
    console.print(f"[bold yellow]Info:[/bold yellow] {message}")
