"""Model data helper tests."""

from __future__ import annotations
from typing import List

import numpy as np
from imblearn.over_sampling import SMOTE, SMOTEN, SMOTENC
from imblearn.under_sampling import ClusterCentroids

import os
import sys

ROOT_PATH: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)

from simpml.core.base import Dataset, PredictionType
from simpml.core.data_set import DataSet
from simpml.tabular.adapters_pool import ManipulateAdapter
from simpml.tabular.splitter_pool import RandomSplitter
from simpml.tabular.tabular_data_manager import SupervisedTabularDataManager

def test_get_data() -> None:
    """Test getting data."""
    data_manager = SupervisedTabularDataManager(
        DataSet.load_titanic_dataset(),
        target="Survived",
        prediction_type=PredictionType.BinaryClassification,
    )
    assert (
        data_manager.get_training_data()[0].shape[1]
        == data_manager.get_validation_data()[0].shape[1]
    )


def test_random_state_same() -> None:
    """Test when the random state is the same between data managers."""
    data_manager = SupervisedTabularDataManager(
        DataSet.load_titanic_dataset(),
        target="Survived",
        prediction_type=PredictionType.BinaryClassification,
    )

    data_manager_2 = SupervisedTabularDataManager(
        DataSet.load_titanic_dataset(),
        target="Survived",
        prediction_type=PredictionType.BinaryClassification,
    )

    assert (
        data_manager.get_training_data()[0]["Age"].iloc[0]
        == data_manager_2.get_training_data()[0]["Age"].iloc[0]
    )


def test_random_state_different() -> None:
    """Test when the random state is different between data managers."""
    splitter = RandomSplitter(target="Survived", random_state=5)
    data_manager = SupervisedTabularDataManager(
        DataSet.load_titanic_dataset(),
        target="Survived",
        splitter=splitter,
        prediction_type=PredictionType.BinaryClassification,
    )

    data_manager_2 = SupervisedTabularDataManager(
        DataSet.load_titanic_dataset(),
        target="Survived",
        prediction_type=PredictionType.BinaryClassification,
    )

    assert (
        data_manager.get_training_data()[0]["Age"].iloc[0]
        != data_manager_2.get_training_data()[0]["Age"].iloc[0]
    )


def test_description() -> None:
    """Test the description functionality."""
    data_manager = SupervisedTabularDataManager(
        DataSet.load_titanic_dataset(),
        target="Survived",
        prediction_type=PredictionType.BinaryClassification,
    )
    data_manager.set_description("Test")

    assert data_manager.description == "Test"


def test_clone() -> None:
    """Test the clone functionality."""
    splitter = RandomSplitter(target="Survived", random_state=5)
    data_manager = SupervisedTabularDataManager(
        DataSet.load_titanic_dataset(),
        target="Survived",
        splitter=splitter,
        prediction_type=PredictionType.BinaryClassification,
    )
    data_manager_2 = data_manager.clone()
    assert data_manager_2.id != data_manager.id


def test_undersampling_balance_method() -> None:
    """Test the undersampling balance method."""
    data_manager = SupervisedTabularDataManager(
        DataSet.load_titanic_dataset(),
        target="Survived",
        prediction_type=PredictionType.BinaryClassification,
    )
    pipeline = data_manager.build_pipeline()
    data_manager.load_pipeline(pipeline)
    data_manager.load_and_split_data()
    X_train, _ = data_manager.get_training_data()
    original_size = X_train.shape[0]

    data_manager = SupervisedTabularDataManager(
        DataSet.load_titanic_dataset(),
        target="Survived",
        prediction_type=PredictionType.BinaryClassification,
    )
    pipeline = data_manager.build_pipeline()
    ros_wrapped = ManipulateAdapter(ClusterCentroids(), "fit_resample")  # imblearn step

    pipeline.add_train_step(("sampler", ros_wrapped))

    data_manager.load_pipeline(pipeline)
    data_manager.load_and_split_data()

    X_train, _ = data_manager.get_training_data()

    size_after_balance = X_train.shape[0]

    assert original_size > size_after_balance


def test_oversampling_balance_method_smote() -> None:
    """Test data set that has only numerical features."""
    data_manager = SupervisedTabularDataManager(
        DataSet.load_titanic_dataset(),
        target="Survived",
        prediction_type=PredictionType.BinaryClassification,
    )
    pipeline = data_manager.build_pipeline()
    data_manager.load_pipeline(pipeline)
    data_manager.load_and_split_data()
    X_train, _ = data_manager.get_training_data()
    original_size = X_train.shape[0]

    data_manager = SupervisedTabularDataManager(
        DataSet.load_titanic_dataset(),
        target="Survived",
        prediction_type=PredictionType.BinaryClassification,
    )
    pipeline = data_manager.build_pipeline()
    ros_wrapped = ManipulateAdapter(SMOTE(), "fit_resample")  # imblearn step

    pipeline.add_train_step(("sampler", ros_wrapped))

    data_manager.load_pipeline(pipeline)
    data_manager.load_and_split_data()

    X_train, _ = data_manager.get_training_data()

    size_after_balance = X_train.shape[0]

    assert original_size < size_after_balance
