from __future__ import annotations

from mltrainer.core.dataset import load_split_data
from mltrainer.core.evaluator import evaluate_classification, evaluate_regression, sort_results
from mltrainer.core.model_registry import RANDOM_STATE, TEST_SIZE, get_model_builders


def train_dataset(dataset: str) -> tuple[dict[str, object], object]:
    split_data = load_split_data(dataset)
    spec = split_data.spec
    results: list[dict[str, object]] = []
    fitted_models: dict[str, object] = {}

    for key, builder in get_model_builders(dataset).items():
        model = builder()
        model.fit(split_data.X_train, split_data.y_train)
        predictions = model.predict(split_data.X_test)
        if spec.task == "classification":
            metrics = evaluate_classification(split_data.y_test, predictions)
        else:
            metrics = evaluate_regression(split_data.y_test, predictions)
        results.append(
            {
                "key": key,
                "name": spec.model_names[key],
                "metrics": metrics,
            }
        )
        fitted_models[key] = model

    ranked_results = sort_results(results, spec.ranking_metric)
    for index, result in enumerate(ranked_results, start=1):
        result["rank"] = index

    best_result = ranked_results[0]
    metrics_payload = {
        "dataset": spec.name,
        "task": spec.task,
        "ranking_metric": spec.ranking_metric,
        "split": {
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
        },
        "best_model": {
            "key": best_result["key"],
            "name": best_result["name"],
        },
        "models": ranked_results,
    }
    return metrics_payload, fitted_models[best_result["key"]]
