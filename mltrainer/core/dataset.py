from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from mltrainer.core.downloader import ensure_dataset_available
from mltrainer.core.model_registry import RANDOM_STATE, TEST_SIZE, DatasetSpec, get_dataset_spec


@dataclass
class SplitData:
    spec: DatasetSpec
    feature_names: list[str]
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def load_split_data(dataset: str) -> SplitData:
    spec = get_dataset_spec(dataset)
    ensure_dataset_available(dataset)
    frame = pd.read_csv(spec.file_path, sep=spec.delimiter)
    features = frame.drop(columns=[spec.label_column])
    labels = frame[spec.label_column]

    split_kwargs = {
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
    }
    if spec.task == "classification":
        split_kwargs["stratify"] = labels

    X_train, X_test, y_train, y_test = train_test_split(features, labels, **split_kwargs)
    return SplitData(
        spec=spec,
        feature_names=list(features.columns),
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )
