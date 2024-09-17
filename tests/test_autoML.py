"""Tests for AutoML

In case tests fail after simpML has been updated autoML will fail too.
Should any modifications be made within this file, they must also be updated in AutoML.
https://dev.azure.com/ni/DevCentral/_git/op-data-science-projects?path=/TabularApplication
"""
from __future__ import annotations

import datetime
import inspect
from typing import Any, Callable, Dict, Optional, Union

import pandas as pd
import plotly.graph_objects as go
import pytest

import os
import sys

ROOT_PATH: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)

from simpml.common.metrics_pool import METRICS_POOL
from simpml.core.base import Dataset, MetricName, PredictionType, SplitterBase
from simpml.core.data_set import DataSet
from simpml.core.experiment_manager import ExperimentManager
from simpml.tabular.inference import TabularInferenceManager
from simpml.tabular.interpreter import TabularInterpreter
from simpml.tabular.splitter_pool import DateTimeSplitter, GroupSplitter, RandomSplitter
from simpml.tabular.tabular_data_manager import CrossValidationSupervisedTabularDataManager
from simpml.tabular.tabular_data_manager import SupervisedTabularDataManager

metrics_for_BinaryClassification = [
    'Accuracy',
    'AUC',
    'Recall',
    'Precision',
    'Balanced Accuracy',
    'F1'
]
args_for_BinaryClassification = {
    'target': 'Survived',
    'add_missing_indicator': True,
    'infinity_to_nan': True,
    'mean_median_imputer': True,
    'categorical_imputer': True,
    'nan_column_dropper': True,
    'one_hot_encoder': True,
    'datetime_features': True,
    'smote': False,
    'ordinal_encode_target': True,
    'high_cardinality_dropper': True,
    'min_max_scaler': True,
    'waveforms_feature_extractor': False,
    'match_variables': True,
    'remove_special_json_characters': True,
    'id_label_encoder': False,
    'drop_cols': None,
    'inernal_drop_cols': None,
    'step_params': None
}
targets_for_BinaryClassification = [1, 0]
args_for_Regression = {
    'target': 'AveBedrms',
    'add_missing_indicator': True,
    'infinity_to_nan': True,
    'mean_median_imputer': True,
    'categorical_imputer': False,
    'nan_column_dropper': True,
    'one_hot_encoder': False,
    'datetime_features': True,
    'smote': False,
    'ordinal_encode_target': False,
    'high_cardinality_dropper': False,
    'min_max_scaler': True,
    'waveforms_feature_extractor': False,
    'match_variables': True,
    'remove_special_json_characters': False,
    'id_label_encoder': False,
    'drop_cols': None,
    'inernal_drop_cols': None,
    'step_params': None
}
metrics_for_Regression = [
    'MSE',
    'RMSE',
    'R2'
]
targets_for_Regression = [
    0.9717138103161398,
    6.071428571428571,
    1.1954887218045114,
    1.3350634371395618,
    1.9238461538461538,
    1.5113636363636365,
    1.0388841927303465,
    2.017391304347826,
    1.278150633855332,
    1.460984393757503,
    2.7252906976744184,
    1.306787330316742,
    1.5584269662921348,
    1.4513513513513514,
]
Filtered_Dict = {
    'add_missing_indicator': True,
    'infinity_to_nan': True,
    'mean_median_imputer': True,
    'categorical_imputer': True,
    'nan_column_dropper': True,
    'one_hot_encoder': True,
    'datetime_features': True,
    'smote': False,
    'ordinal_encode_target': True,
    'high_cardinality_dropper': True,
    'min_max_scaler': True,
    'waveforms_feature_extractor': False,
    'match_variables': True,
    'remove_special_json_characters': True,
    'id_label_encoder': False
}
Filtered_Kwargs = {
    'add_missing_indicator': True,
    'infinity_to_nan': True,
    'mean_median_imputer': True,
    'categorical_imputer': True,
    'nan_column_dropper': True,
    'one_hot_encoder': True,
    'datetime_features': True,
    'smote': False,
    'ordinal_encode_target': True,
    'high_cardinality_dropper': True,
    'min_max_scaler': True,
    'waveforms_feature_extractor': False,
    'match_variables': True,
    'remove_special_json_characters': True,
    'id_label_encoder': False
}

'###################### AutoML Code ###############################'


def train_model(
    df: pd.DataFrame,
    target: str,
    prediction_type: str,
    optimize_metric: Union[str, MetricName],
    split_dict: Dict[str, float],
    positive_class: str | int | None = None,
    add_missing_indicator: bool = False,
    infinity_to_nan: bool = False,
    mean_median_imputer: bool = False,
    categorical_imputer: bool = False,
    nan_column_dropper: bool = False,
    one_hot_encoder: bool = False,
    datetime_features: bool = False,
    smote: bool = False,
    ordinal_encode_target: bool = False,
    high_cardinality_dropper: bool = False,
    min_max_scaler: bool = False,
    match_variables: bool = False,
    remove_special_json_characters: bool = False,
    drop_cols: Optional[list] = None,
    split_type: bool | str = False,
    n_folds: bool = False,
    is_rca: bool = False,
    is_cross_validation: bool = False,
    waveforms_feature_extractor: bool = False,
    id_label_encoder: bool = False,
    group_columns: str = '',
    column_id: Optional[str] = None,
    column_sort: str = "",
    date_column: str = ""
) -> tuple[TabularInferenceManager, Dict[str, str], TabularInterpreter]:
    """Train Function"""
    def replace_keys(split_sets: Dict[str, float]) -> Dict[Dataset, float]:
        """Replace the keys in the split_sets dictionary with Dataset enum members.

        Args:
        split_sets (dict): A dictionary with string keys to be replaced.

        Returns:
        dict: A dictionary with keys replaced by Dataset enum members.
        """
        return {Dataset[key]: value for key, value in split_sets.items()}

    def convert_date_column_to_timestamp(
        df: pd.DataFrame,
        date_column: Optional[str]
    ) -> pd.DataFrame:
        """Convert a date column in a DataFrame to timestamps.

        Args:
        df (pd.DataFrame): DataFrame containing the date column.
        date_column (str): Name of the column to be converted.

        Returns:
        pd.DataFrame: DataFrame with the date column converted to timestamps.
        """
        if isinstance(df[date_column].iloc[0], datetime.datetime):
            df[date_column] = df[date_column].apply(lambda date: date.timestamp())
        else:
            df[date_column] = df[date_column].apply(
                lambda date_str: pd.to_datetime(date_str).timestamp()
            )

        return df

    def select_splitter(is_rca: bool,
                        split_type: bool | str,
                        group_columns: str,
                        df: pd.DataFrame,
                        date_column: str,
                        target: str,
                        split_sets: Dict[str, float],
                        prediction_type: str) -> Union[str, SplitterBase]:
        """Selects the appropriate splitter based on given conditions.

        Args:
        is_rca (bool): Indicates if RCA splitter is to be used.
        split_type (str): Type of splitting ("Random", "By Date", or other).
        group_columns (list): List of columns for group splitting.
        df (pd.DataFrame): The DataFrame to be used.
        date_column (str): Name of the date column (if applicable).
        target (str): Target column name.
        split_sets (dict): Splitting ratios.
        prediction_type (PredictionType): The type of prediction.

        Returns:
        splitter: The selected splitter based on the conditions.
        """
        if is_rca:
            return "RCA"

        by_time = split_type == "By Date"

        if group_columns:
            return GroupSplitter(
                split_sets=replace_keys(split_sets),
                group_columns=group_columns
            )
        # add by_time=by_time, target=target

        elif by_time:
            df = convert_date_column_to_timestamp(df, date_column)
            return DateTimeSplitter(
                target=target,
                split_sets=replace_keys(split_sets),
                time_column=date_column
            )

        elif split_type == "Random":
            return RandomSplitter(
                target=target,
                split_sets=replace_keys(split_sets),
                stratify=prediction_type != PredictionType.Regression.value
            )

        raise ValueError("Invalid split type or conditions.")

    label_encoding = None
    if prediction_type == PredictionType.BinaryClassification.value and ordinal_encode_target:
        negative_class = [val for val in df[target].unique() if val != positive_class][0]
        label_encoding = {negative_class: 0, positive_class: 1}

    splitter = select_splitter(
        is_rca,
        split_type,
        group_columns,
        df,
        column_sort,
        target,
        split_dict,
        prediction_type)
    data_manager = None

    if not is_cross_validation:
        data_manager = SupervisedTabularDataManager(
            data=df,
            target=target,
            prediction_type=PredictionType[prediction_type],
            splitter=splitter
        )
    else:
        data_manager = CrossValidationSupervisedTabularDataManager(
            data=df,
            target=target,
            prediction_type=PredictionType[prediction_type],
            n_folds=n_folds
        )

    data_manager.build_pipeline(
        add_missing_indicator=add_missing_indicator,
        infinity_to_nan=infinity_to_nan,
        mean_median_imputer=mean_median_imputer,
        categorical_imputer=categorical_imputer,
        nan_column_dropper=nan_column_dropper,
        one_hot_encoder=one_hot_encoder,
        datetime_features=datetime_features,
        smote=smote,
        ordinal_encode_target=ordinal_encode_target,
        high_cardinality_dropper=high_cardinality_dropper,
        min_max_scaler=min_max_scaler,
        match_variables=match_variables,
        remove_special_json_characters=remove_special_json_characters,
        drop_cols=drop_cols,
        waveforms_feature_extractor=waveforms_feature_extractor,
        id_label_encoder=id_label_encoder,
        step_params={
            'column_id': column_id,
            'encoding_dict': label_encoding
        }
    )

    condition = prediction_type == PredictionType.BinaryClassification.value

    exp_mang = ExperimentManager(data_manager, optimize_metric=optimize_metric)
    exp_mang.run_experiment(
        metrics_kwargs={"pos_label" : 1} if condition and ordinal_encode_target else None
    )

    interp = TabularInterpreter(
        model=exp_mang.get_best_model(),
        model_data=exp_mang.get_data_model_of_best_model(),
        opt_metric=exp_mang.opt_metric,
        prediction_type=prediction_type,
        pos_class={'pos_class' : 1 if condition and ordinal_encode_target else None},
        verbose=False,
        enable_shap=False)
    inference_manager = TabularInferenceManager(data_manager, exp_mang.get_best_model())
    columns_to_drop = [
        'Experiment ID',
        'Data Version',
        'Data Description',
        'Model Params',
        'Metric Params'
    ]
    models = exp_mang.df_metrics.drop(
        columns_to_drop,
        axis=1).sort_values(
            optimize_metric,
            ascending=False).to_json(
                orient='records') if exp_mang.df_metrics is not None else None
    best_model = exp_mang.get_best_model().name
    return inference_manager, {"models": models, "best_model": best_model}, interp


def get_train_args(df: pd.DataFrame, target: str, prediction_type: PredictionType) -> Any:
    """Gets correct arguments for the selected prediction type"""
    data_manager = SupervisedTabularDataManager(
        data=df,
        target=target,
        prediction_type=prediction_type,
        splitter=RandomSplitter(stratify=False))

    sig = inspect.signature(data_manager.build_pipeline)
    params = sig.parameters
    params_dict: Dict[str, Optional[str]] = {}

    for name, param in params.items():
        if param.default is param.empty:
            params_dict[name] = None
        else:
            params_dict[name] = param.default

    metrics_list = []

    for metric in METRICS_POOL[data_manager.data_type][data_manager.prediction_type]:
        metrics_list.append(metric.name)
    target = df[target].value_counts(ascending=True).index.tolist()

    return params_dict, metrics_list, target


def filter_kwargs(func: Callable[..., Any], kwargs: Dict) -> Dict:
    """Filter a dictionary of keyword arguments to include only those
    that are accepted by a given function.
    """
    from inspect import signature

    # Get the signature of the function
    sig = signature(func)

    # Filter the kwargs based on the function's parameters
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}

    return filtered_kwargs


'###################### AutoML Tests ###############################'


@pytest.mark.parametrize(
    "data, target, prediction_type, Args, Metrics, Targets",
    [(DataSet.load_titanic_dataset(),
      "Survived",
      PredictionType.BinaryClassification.value,
      args_for_BinaryClassification,
      metrics_for_BinaryClassification,
      targets_for_BinaryClassification),
     (DataSet.load_fetch_california_housing_dataset(),
      "AveBedrms",
      PredictionType.Regression.value,
      args_for_Regression,
      metrics_for_Regression,
      targets_for_Regression)]
)
def test_get_train_args(
    data: pd.DataFrame,
    target: str,
    prediction_type: PredictionType,
    Args: dict,
    Metrics: list,
    Targets: list
) -> None:
    """Test Get Train Args Function"""
    args, metrics, targets = get_train_args(data, target, prediction_type)
    assert args == Args
    assert metrics == Metrics
    for i in range(len(Targets)):
        assert Targets[i] == targets[i]


@pytest.mark.parametrize(
    "func, filtered_dict, Filtered_kwargs",
    [(train_model,
      Filtered_Dict,
      Filtered_Kwargs)]
)
def test_filter_kwargs(
    func: Callable[..., Any],
    filtered_dict: dict,
    Filtered_kwargs: dict
) -> None:
    """Test filter_kwargs"""
    filtered_kwargs = filter_kwargs(func, filtered_dict)
    assert filtered_kwargs == Filtered_kwargs


@pytest.mark.parametrize(
    "data, target, optimize_metric, poss_class, prediction_type, add_TSFResh, group_columns, split_type, column_sort",
    [
        (
            DataSet.load_titanic_dataset(),
            'Survived',
            'AUC',
            '1',
            PredictionType.BinaryClassification.value,
            False,
            None,
            'Random',
            ''
        ),
        (
            DataSet.load_fetch_california_housing_dataset(),
            'AveBedrms',
            'MSE',
            '',
            PredictionType.Regression.value,
            False,
            None,
            'Random',
            ''
        ),
        (
            DataSet.load_time_series_classification_dataset()[::100],
            'target',
            'AUC',
            1,
            PredictionType.BinaryClassification.value,
            False,
            'ID',
            '',
            ''
        )
    ]
)
def test_train(
    data: pd.DataFrame,
    target: str,
    optimize_metric: str,
    poss_class: str,
    prediction_type: PredictionType,
    add_TSFResh: bool,
    group_columns: str,
    split_type: str,
    column_sort: str
) -> None:
    """Test Train Function, Get Train Arguments and Filter Kwargs"""
    args, metrics, targets = get_train_args(data, target, prediction_type)
    if add_TSFResh:
        args["waveforms_feature_extractor"] = True
    filtered_dict = {k: v for k, v in args.items() if isinstance(v, bool)}
    defaults = [k for k, v in filtered_dict.items() if v is True]
    steps = defaults
    for key in filtered_dict:
        if key in steps:
            filtered_dict[key] = True
        else:
            filtered_dict[key] = False
    filtered_kwargs = filter_kwargs(train_model, filtered_dict)
    inference_manager, results, interp = train_model(
        df=data,
        target=target,
        prediction_type=str(prediction_type),
        optimize_metric=optimize_metric,
        positive_class=poss_class,
        split_type=split_type,
        split_dict={'Train': 0.8, 'Valid': 0.2},
        column_sort=column_sort,
        **filtered_kwargs,
        group_columns=group_columns
    )

    assert isinstance(inference_manager, TabularInferenceManager)
    assert results["best_model"]
    fig = interp.main_fig()  # type: ignore
    assert isinstance(fig, go.Figure)
