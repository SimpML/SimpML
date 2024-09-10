"""File to implement the ShapManager class"""

from __future__ import annotations

from typing import Any, cast, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor

from simpml.core.base import PandasModelManagerBase


def is_tree_based(model: Any) -> bool:
    """Helper function to check if the model is a tree-based model."""
    return isinstance(
        model,
        (
            DecisionTreeClassifier,
            DecisionTreeRegressor,
            RandomForestClassifier,
            RandomForestRegressor,
            GradientBoostingClassifier,
            GradientBoostingRegressor,
            XGBClassifier,
            XGBRegressor,
            LGBMClassifier,
            LGBMRegressor,
        ),
    )


class ShapManager:
    """ShapManager class
    This class is instantiated by the interpreter and handles
    all SHAP related calculations and funcionality
    """

    def __init__(
        self,
        model: "PandasModelManagerBase",  # Assuming custom type
        X_train: pd.DataFrame,
        X_valid: pd.DataFrame,
        verbose: Optional[bool] = True,
    ):
        """Initialize the ShapManager class."""
        self.model = model.model
        self.X_train = X_train
        self.X_valid = X_valid
        self.verbose = verbose
        self.cache: Dict[str, shap.Explanation] = {}  # Properly type-annotated cache
        self.initialize_explainer_and_cache(model)

    def initialize_explainer_and_cache(self, model: "PandasModelManagerBase") -> None:
        """Helper function to initialize the SHAP explainer and cache SHAP values."""
        try:
            if is_tree_based(self.model) and hasattr(self.model, "predict_proba"):
                self.explainer = shap.TreeExplainer(
                    self.model, self.X_train, model_output="probability"
                )
            else:
                self.initialize_general_explainer()

            self.cache_shap_values()

        except Exception as e:
            if self.verbose:
                print(f"{model.name}: Failed to initialize SHAP Explainer with error: {e}")

    def initialize_general_explainer(self) -> None:
        """Initialize a general SHAP explainer when the model is not tree-based.
        Attempts to use KernelExplainer if 'predict_proba' is available.
        """
        try:
            self.explainer = shap.Explainer(self.model, self.X_train)
        except Exception:
            self.try_kernel_explainer()

    def try_kernel_explainer(self) -> None:
        """Attempt to initialize a KernelExplainer for models with 'predict_proba'.
        Uses a sample of the training data for efficiency.
        """
        try:
            if hasattr(self.model, "predict_proba"):
                self.explainer = shap.KernelExplainer(
                    self.model.predict_proba, self.X_train.sample(50)
                )
        except Exception as e:
            if self.verbose:
                print(f"{self.model.name}: Failed to initialize KernelExplainer with error: {e}")

    def cache_shap_values(self) -> None:
        """Calculate SHAP values for the training and validation data and cache them.
        Does not use caching if calculating values to avoid recursive caching issues.
        """
        self.cache["X_train"] = self.get_shap_values(self.X_train, cache=False)
        self.cache["X_valid"] = self.get_shap_values(self.X_valid, cache=False)

    def get_shap_values(
        self,
        x: Optional[pd.DataFrame] = None,
        single_mat_abs: bool = False,
        as_df: bool = False,
        cache: bool = True,
    ) -> Union[shap.Explanation, pd.DataFrame]:
        """Returns the SHAP values for the given data based
        on the interpreters model and explainer
        """
        explainer = self.explainer
        if x is None:
            x = self.X_valid
        if x.equals(self.X_train) and cache and "X_train" in self.cache.keys():
            vals = self.cache["X_train"]
        elif x.equals(self.X_valid) and cache and "X_valid" in self.cache.keys():
            vals = self.cache["X_valid"]
        else:
            try:
                vals = explainer(x)
            except Exception as e:
                if "ExplainerError" in str(type(e)):
                    if self.verbose:
                        print("Caught an ExplainerError:", e)
                        print(
                            "Running without check_additivity which\
                            might lead to inaccurate results!"
                        )
                    vals = explainer(x, check_additivity=False)
                else:
                    raise
        if single_mat_abs:
            if isinstance(vals, list):
                vals = np.sum(np.abs(np.array(vals)), axis=0)
            elif hasattr(vals, "values"):
                vals = vals.values
                if len(vals.shape) > 2:
                    vals = np.sum(np.abs(np.array(vals)), axis=2)
                else:
                    vals = np.abs(np.array(vals))
            else:
                vals = np.abs(np.array(vals))
            if as_df:
                return pd.DataFrame(vals, columns=x.columns)
        return vals

    def plot_summary_shap(
        self, x: Optional[pd.DataFrame] = None, size: Tuple[int, int] = (8, 6)
    ) -> None:
        """Create a SHAP beeswarm plot, colored by feature values when they are provided.

        Args:
            X: Input Dataset.
            size: Tuple of width, height of the plot.

        Returns:
            A plot of the shap values for each feature.
        """
        if x is None:
            x = self.X_valid
        shap_values = self.get_shap_values(x)
        shap.summary_plot(shap_values, feature_names=x.columns)

    def plot_shap_on_row(
        self,
        X: Optional[Union[pd.DataFrame, shap.Explainer]] = None,
        row_number: int = 0,
        plot_type: str = "force_plot",
    ) -> Union[shap.force_plot, shap.decision_plot]:
        """Visualize the given SHAP values with an additive force layout or Visualize model
        decisions using cumulative SHAP values.

        Args:
            X: Input Dataset.
            row_number: Index of row to display shap values for.
            plot_type: Type of plot to display, either "force_plot" (default) or "decision_plot".

        Returns:
            A plot of each features shap value on a single row of data
        """
        if X is None:
            X = self.X_valid
        assert X is not None
        shap_values = self.get_shap_values(X)
        if isinstance(self.explainer.expected_value, (int, float, np.float32)):
            expected_value = self.explainer.expected_value
        else:
            expected_value = self.explainer.expected_value[0]

        if plot_type == "force_plot":
            return shap.force_plot(
                base_value=expected_value,
                shap_values=np.array(shap_values.data[row_number]),
                features=X.iloc[row_number],
            )
        elif plot_type == "decision_plot":
            return shap.decision_plot(
                base_value=expected_value,
                shap_values=np.array(shap_values.data[row_number]),
                feature_names=X.columns.to_list(),
                link="logit",
            )
        else:
            raise ValueError(f"Invalid plot_type: {plot_type}")

    def plot_shap_dependence(self, feature_name: str, X: pd.DataFrame = None) -> None:
        """Create a SHAP dependence plot, colored by an interaction feature.

        Args:
            feature_name: Name of the feature for which the SHAP values are plotted.
            X: Data frame of input features. If None, uses a predefined validation set.
        """
        if X is None:
            X = self.X_valid
        vals = self.get_shap_values(X)
        shap.dependence_plot(
            ind=feature_name,
            shap_values=vals.values,
            features=X,
            feature_names=X.columns.tolist(),
            interaction_index="auto",
        )

    def calculate_shap_fi(
        self, X: pd.DataFrame = None, shap_values: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """Produces an importance table based on SHAP (much less biased than native feature
        importance implementations that exist for individual algorithms).

        Args:
            x: The local population relevant to the calculation.
            shap_values: The function may have the Shap values to save running time, otherwise it
                will perform the calculation.

        Returns:
            A data frame with the importance table.
        """
        if X is None:
            X = self.X_valid
        if shap_values is None:
            shap_values = self.get_shap_values(X, single_mat_abs=True, as_df=False)
        feature_importance = pd.DataFrame(
            list(zip(X.columns, cast(np.ndarray, sum(shap_values)))),
            columns=["col_name", "feature_importance_vals"],
        )
        feature_importance.sort_values(
            by=["feature_importance_vals"], ascending=False, inplace=True
        )
        feature_importance["feature_importance_vals"] = (
            feature_importance["feature_importance_vals"]
            / feature_importance["feature_importance_vals"].sum()
        )
        return feature_importance.reset_index(drop=True)
