from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, LogisticRegression, RANSACRegressor, Ridge
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

RANDOM_STATE = 42
TEST_SIZE = 0.2
DATASET_DIR = Path(__file__).resolve().parents[2] / "dataset"
OUTPUTS_DIR = Path(__file__).resolve().parents[2] / "outputs"


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    file_path: Path
    delimiter: str
    label_column: str
    task: str
    ranking_metric: str
    model_names: dict[str, str]
    recommended_model_key: str
    download_source: str
    download_url: str | None = None
    kaggle_dataset: str | None = None


DATASET_SPECS: dict[str, DatasetSpec] = {
    "heart": DatasetSpec(
        name="heart",
        file_path=DATASET_DIR / "heart.csv",
        delimiter=",",
        label_column="target",
        task="classification",
        ranking_metric="f1",
        model_names={
            "LR": "Logistic Regression",
            "DT": "Decision Tree Classifier",
            "SVM": "Support Vector Machine",
            "RF": "Random Forest Classifier",
            "KNN": "K-Nearest Neighbors",
        },
        recommended_model_key="RF",
        download_source="kaggle",
        kaggle_dataset="johnsmith88/heart-disease-dataset",
    ),
    "winequality-red": DatasetSpec(
        name="winequality-red",
        file_path=DATASET_DIR / "winequality-red.csv",
        delimiter=";",
        label_column="quality",
        task="regression",
        ranking_metric="rmse",
        model_names={
            "Regression": "Linear Regression",
            "RANSAC": "RANSAC Regressor",
            "Ridge": "Ridge Regression",
            "LASSO": "Lasso Regression",
            "ElasticNet": "ElasticNet Regression",
            "DTR": "Decision Tree Regressor",
            "RFR": "Random Forest Regressor",
        },
        recommended_model_key="RFR",
        download_source="http",
        download_url="https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
    ),
}


def get_dataset_spec(dataset: str) -> DatasetSpec:
    try:
        return DATASET_SPECS[dataset]
    except KeyError as exc:
        supported = ", ".join(sorted(DATASET_SPECS))
        raise ValueError(f"Unsupported dataset '{dataset}'. Supported datasets: {supported}") from exc


def get_model_builders(dataset: str) -> dict[str, callable]:
    spec = get_dataset_spec(dataset)
    if spec.task == "classification":
        return {
            "LR": lambda: make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
            ),
            "DT": lambda: DecisionTreeClassifier(random_state=RANDOM_STATE),
            "SVM": lambda: make_pipeline(StandardScaler(), SVC(random_state=RANDOM_STATE)),
            "RF": lambda: RandomForestClassifier(random_state=RANDOM_STATE),
            "KNN": lambda: make_pipeline(StandardScaler(), KNeighborsClassifier()),
        }

    return {
        "Regression": lambda: make_pipeline(StandardScaler(), LinearRegression()),
        "RANSAC": lambda: make_pipeline(
            StandardScaler(),
            RANSACRegressor(random_state=RANDOM_STATE),
        ),
        "Ridge": lambda: make_pipeline(StandardScaler(), Ridge()),
        "LASSO": lambda: make_pipeline(StandardScaler(), Lasso(random_state=RANDOM_STATE, max_iter=5000)),
        "ElasticNet": lambda: make_pipeline(
            StandardScaler(),
            ElasticNet(random_state=RANDOM_STATE, max_iter=5000),
        ),
        "DTR": lambda: DecisionTreeRegressor(random_state=RANDOM_STATE),
        "RFR": lambda: RandomForestRegressor(random_state=RANDOM_STATE),
    }


def get_optimization_search_space(dataset: str, method: str) -> tuple[object, dict[str, list | object]]:
    spec = get_dataset_spec(dataset)
    if spec.task == "classification":
        estimator = RandomForestClassifier(random_state=RANDOM_STATE)
        if method == "grid":
            params = {
                "n_estimators": [100, 200],
                "max_depth": [None, 10, 20],
                "min_samples_split": [2, 5],
                "min_samples_leaf": [1, 2],
                "max_features": ["sqrt", "log2"],
                "class_weight": [None, "balanced"],
            }
        else:
            params = {
                "n_estimators": [100, 150, 200, 300],
                "max_depth": [None, 5, 10, 20, 30],
                "min_samples_split": [2, 4, 6, 8],
                "min_samples_leaf": [1, 2, 4],
                "max_features": ["sqrt", "log2", None],
                "class_weight": [None, "balanced", "balanced_subsample"],
            }
        return estimator, params

    estimator = RandomForestRegressor(random_state=RANDOM_STATE)
    if method == "grid":
        params = {
            "n_estimators": [100, 200],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2],
            "max_features": [1.0, "sqrt", "log2"],
        }
    else:
        params = {
            "n_estimators": [100, 150, 200, 300],
            "max_depth": [None, 5, 10, 20, 30],
            "min_samples_split": [2, 4, 6, 8],
            "min_samples_leaf": [1, 2, 4],
            "max_features": [1.0, "sqrt", "log2"],
        }
    return estimator, params
