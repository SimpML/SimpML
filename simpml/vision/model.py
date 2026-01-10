"""Vision model based classes."""

from __future__ import annotations

import copy
import warnings
from os import PathLike
from types import ModuleType
from typing import Any, Dict, Optional, Tuple, Union

import cv2
import dill
import numpy as np
import pandas as pd
import torch
from fastai.data.core import DataLoaders
from fastai.learner import Learner
from fastai.vision.learner import vision_learner
from sklearn.ensemble import RandomForestClassifier
from typing_extensions import Self

from simpml.vision.base import FastaiModelManagerBase


class FastaiModelClassificationManager(FastaiModelManagerBase):
    """Model manager for FastAI vision classification models."""

    def __init__(self, arch: Any, name: str, desc: str) -> None:
        """Initializes the FastaiModelClassificationManager class.

        Args:
            arch: The model for the Vision architecture (e.g. resnet50).
            name: The name of the model view.
            desc: Description of the model.
        """
        self.arch: Any = arch
        self.name: str = name
        self.desc: str = desc
        self.model: Optional[Learner] = None
        self.dls: Optional[DataLoaders] = None
        self.batch_size: int = 32

    def set_batch_size(self, batch_size: int) -> None:
        """Set the batch size for predictions."""
        if batch_size < 1:
            raise ValueError("Batch size must be at least 1")
        self.batch_size = batch_size

    def __repr__(self) -> str:
        if self.model is not None:
            return f"Model: {self.model}, Description: {self.desc}"
        return f"Model: {self.name}, Description: {self.desc}"

    def fit(self, data: DataLoaders, num_epocs: int = 5, **kwargs: Any) -> Self:
        """Fit the model."""
        if kwargs:
            raise RuntimeError(f"Unrecognized kwargs: {kwargs}")
        self.dls = data
        self.model = vision_learner(data, self.arch)
        self.model.fine_tune(num_epocs)
        return self

    def get_preds(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Use the model to make a prediction (detailed results).

        Args:
            X: The feature data.

        Returns:
            A tuple containing:
            - probabilities: The prediction probabilities
            - predictions: The predicted classes
        """
        assert self.dls is not None, "Model hasn't been fitted yet. Call fit() first."
        assert self.model is not None, "Model hasn't been fitted yet. Call fit() first."

        dl = self.dls.test_dl(X)
        preds = self.model.get_preds(dl=dl)
        probs = preds[0].cpu().numpy()
        pred_class = probs.argmax(axis=1)
        
        return probs, pred_class

    def get_preds_batched(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Use the model to make predictions in batches.
        
        Args:
            X: The feature data.
            
        Returns:
            A tuple containing:
            - probabilities: The prediction probabilities
            - predictions: The predicted classes
        """
        assert self.dls is not None, "Model hasn't been fitted yet. Call fit() first."
        assert self.model is not None, "Model hasn't been fitted yet. Call fit() first."

        all_probs = []
        
        try:
            # Process data in batches
            for i in range(0, len(X), self.batch_size):
                batch = X[i:i + self.batch_size]
                dl = self.dls.test_dl(batch)
                preds = self.model.get_preds(dl=dl)
                probs = preds[0]
                
                # Move to CPU and convert to numpy
                all_probs.append(probs.cpu().numpy())

                # Optional: clear GPU cache if using CUDA
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Concatenate results
            final_probs = np.concatenate(all_probs)
            final_preds = final_probs.argmax(axis=1)
            
            return final_probs, final_preds
            
        except Exception as e:
            raise RuntimeError(f"Error during batch prediction: {str(e)}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Use the model to make a prediction.

        Args:
            X: The feature data.

        Returns:
            The prediction results.
        """
        if len(X) <= self.batch_size:
            probs, preds = self.get_preds(X)
        else:
            probs, preds = self.get_preds_batched(X)
        
        return preds

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Get the prediction probabilities.

        Args:
            X: The feature data.

        Returns:
            The prediction probabilities.
        """
        if len(X) <= self.batch_size:
            probs, preds = self.get_preds(X)
        else:
            probs, preds = self.get_preds_batched(X)
            
        return probs

    def export(
        self,
        path: Union[str, PathLike],
        pickle_module: ModuleType = dill,
        pickle_protocol: int = 2,
        include_data: bool = True,
        **kwargs: Any,
    ) -> None:
        """Export model."""
        if kwargs:
            raise RuntimeError(f"Unrecognized kwargs: {kwargs}")
            
        if not include_data:
            if self.model is None:
                raise RuntimeError("There is no model to export.")
            dls = self.model.dls
            self.model.dls = self.model.dls.new_empty()

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                torch.save(self, path, pickle_module=pickle_module, pickle_protocol=pickle_protocol)
        finally:
            if not include_data:
                assert self.model is not None
                self.model.dls = dls

    def clone(self) -> Self:
        """Creates a copy of this class instance."""
        cloned_object = self.__class__(self.arch, self.name, self.desc)
        cloned_object.batch_size = self.batch_size

        if self.model:
            cloned_object.model = copy.deepcopy(self.model)
            cloned_object.model.load_state_dict(self.model.state_dict())
            cloned_object.dls = self.dls

        return cloned_object


class VisionBaselineClassification(FastaiModelManagerBase):
    """Baseline for modeling where we try to predict according to classes' distribution."""

    def __init__(self, name: str, desc: str) -> None:
        """Initializes the VisionBaselineClassification class.

        Args:
            name: The name of the model view.
            desc: Description of the model.
        """
        self.name: str = name
        self.desc: str = desc
        self.target: Optional[Dict[str, float]] = None
        self.classes_: Optional[pd.Series] = None

    def fit(self, data: DataLoaders, num_epocs: int = 5, **kwargs: Any) -> Self:
        """Fit the model.

        Args:
            data: The training data.
            num_epocs: The number of epocs to train. Ignored.
            **kwargs: For compatibility with the base class.

        Returns:
            This class instance.
        """
        del num_epocs

        if kwargs:
            raise RuntimeError(f"Unrecognized kwargs: {kwargs}")
        y_train = pd.Series([int(ins[1]) for ins in data.train_ds])
        self.target = dict((y_train.value_counts() / len(y_train)))
        self.classes_ = y_train.unique()
        self.classes_ = y_train.unique()

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Use the model to make a prediction.

        Args:
            X: The feature data.

        Returns:
            The prediction results.
        """
        assert self.target is not None
        items = self.target.items()
        classes = [item[0] for item in items]
        ps = [item[1] for item in items]
        return np.random.choice(classes, len(X), p=ps)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get the prediction probabilities.

        Args:
            X: The feature data.

        Returns:
            The prediction probabilities.
        """
        assert self.classes_ is not None
        return np.full((len(X), len(self.classes_)), 0.5, dtype=np.float_)

    def export(
        self,
        path: Union[str, PathLike],
        pickle_module: ModuleType = dill,
        pickle_protocol: int = 2,
        include_data: bool = False,
        **kwargs: Any,
    ) -> None:
        """Export the model to a file.

        Args:
            path (Union[str, PathLike]): The file path to export the model into.
            pickle_module (ModuleType, optional): The module used for pickling.
            Defaults to Python's built-in `pickle`.
            pickle_protocol (int, optional): The pickle protocol version to use.
            Defaults to 2.
            include_data (bool, optional): Whether to include data when exporting.
            Defaults to False.
            **kwargs (Any): For compatibility with the base class; raises an error if
            any are provided.

        Raises:
            RuntimeError: If unrecognized kwargs are passed.
        """
        if kwargs:
            raise RuntimeError(f"Unrecognized kwargs: {kwargs}")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            torch.save(self, path, pickle_module=pickle_module, pickle_protocol=pickle_protocol)

    def clone(self) -> Self:
        """Creates a copy of this class instance.

        Returns:
            A copy of this class instance.
        """
        return copy.deepcopy(self)


class ClassicalImageClassifier(FastaiModelManagerBase):
    """Classical Image Classifier."""

    def __init__(self, name: str, desc: str, n_estimators: int = 100) -> None:
        """Initializes the ClassicalImageClassifier class.

        Args:
            name: The name of the model view.
            desc: Description of the model.
            n_estimators: The number of estimators.
        """
        self.name: str = name
        self.desc: str = desc
        self.clf = RandomForestClassifier(n_estimators=n_estimators)

    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """Extract features.

        Args:
            image: The numpy array containing the image data.

        Returns:
            A numpy array of features.
        """
        # Resize the image to a fixed size
        image = cv2.resize(image, (128, 128))

        # Calculate color histograms for each channel
        hist_r = cv2.calcHist([image], [0], None, [16], [0, 256])
        hist_g = cv2.calcHist([image], [1], None, [16], [0, 256])
        hist_b = cv2.calcHist([image], [2], None, [16], [0, 256])

        # Concatenate the histograms into a single feature vector
        features = np.concatenate([hist_r, hist_g, hist_b]).flatten()

        return features

    def fit(self, data: DataLoaders, num_epocs: int = 5, **kwargs: Any) -> Self:
        """Fit the model.

        Args:
            data: The training data.
            num_epocs: The number of epocs to train. Ignored.
            **kwargs: For compatibility with the base class.

        Returns:
            This class instance.
        """
        del num_epocs

        if kwargs:
            raise RuntimeError(f"Unrecognized kwargs: {kwargs}")
        features, labels = [self.extract_features(np.array(i[0])) for i in data.train_ds], np.array(
            [i[1] for i in data.train_ds]
        )
        self.features = features
        self.clf.fit(features, labels)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Use the model to make a prediction.

        Args:
            X: The feature data.

        Returns:
            The prediction results.
        """
        # Extract features for all images in X
        feature_vectors = [self.extract_features(np.array(img)) for img in X]

        # Predict the labels using the classifier
        return self.clf.predict(feature_vectors)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get the prediction probabilities.

        Args:
            X: The feature data.

        Returns:
            The prediction probabilities.
        """
        feature_vectors = [self.extract_features(np.array(img)) for img in X]
        return self.clf.predict_proba(feature_vectors)

    def export(
        self,
        path: Union[str, PathLike],
        pickle_module: ModuleType = dill,
        pickle_protocol: int = 2,
        include_data: bool = False,
        **kwargs: Any,
    ) -> None:
        """Export the model to a file.

        Args:
            path (Union[str, PathLike]): The file path to export the model into.
            pickle_module (ModuleType, optional): The module used for pickling.
            Defaults to Python's built-in `pickle`.
            pickle_protocol (int, optional): The pickle protocol version to use.
            Defaults to 2.
            include_data (bool, optional): Whether to include data when exporting.
            Defaults to False.
            **kwargs (Any): For compatibility with the base class; raises an error
            if any are provided.

        Raises:
            RuntimeError: If unrecognized kwargs are passed.
        """
        if kwargs:
            raise RuntimeError(f"Unrecognized kwargs: {kwargs}")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            torch.save(self, path, pickle_module=pickle_module, pickle_protocol=pickle_protocol)

    def clone(self) -> Self:
        """Creates a copy of this class instance.

        Returns:
            A copy of this class instance.
        """
        return copy.deepcopy(self)


def load_fastai_model(
    path: Union[str, PathLike], cpu: Optional[bool] = None, pickle_module: ModuleType = dill
) -> Any:
    """Load a FastAI model.

    Args:
        path: String or PathLike of file path to export the model into.
        cpu: Whether to use CPU when using the model (vs GPU).
        pickle_module: Which module to use for pickle functionality.

    Returns:
        The FastAI model.
    """
    if cpu is not None:
        pass
    elif torch.cuda.is_available():
        cpu = False
    else:
        cpu = True
    model = torch.load(path, pickle_module=dill, map_location="cpu" if cpu else None)

    if cpu:
        model.model.dls = model.model.dls.cpu()
        model.model.dls = model.model.dls.cpu()
    return model
