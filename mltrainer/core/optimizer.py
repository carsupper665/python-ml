from __future__ import annotations

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from mltrainer.core.dataset import load_split_data
from mltrainer.core.evaluator import evaluate_classification, evaluate_regression
from mltrainer.core.model_registry import get_dataset_spec, get_optimization_search_space


def optimize_dataset(dataset: str, method: str) -> tuple[dict[str, object], object]:
    if method not in {"grid", "random"}:
        raise ValueError("Optimization method must be 'grid' or 'random'.")

    split_data = load_split_data(dataset)
    spec = get_dataset_spec(dataset)
    estimator, params = get_optimization_search_space(dataset, method)
    if spec.task == "classification":
        scoring = "f1"
    else:
        scoring = "neg_root_mean_squared_error"

    if method == "grid":
        search = GridSearchCV(estimator, params, cv=5, n_jobs=-1, scoring=scoring)
    else:
        search = RandomizedSearchCV(
            estimator,
            params,
            cv=5,
            n_jobs=-1,
            scoring=scoring,
            n_iter=12,
            random_state=42,
        )

    search.fit(split_data.X_train, split_data.y_train)
    best_model = search.best_estimator_
    predictions = best_model.predict(split_data.X_test)
    if spec.task == "classification":
        test_metrics = evaluate_classification(split_data.y_test, predictions)
        best_score = float(search.best_score_)
    else:
        test_metrics = evaluate_regression(split_data.y_test, predictions)
        best_score = float(-search.best_score_)

    result = {
        "recommended_model": spec.model_names[spec.recommended_model_key],
        "recommended_model_key": spec.recommended_model_key,
        "method": method,
        "cv_scoring": scoring,
        "best_params": search.best_params_,
        "best_score": best_score,
        "test_metrics": test_metrics,
    }
    return result, best_model
