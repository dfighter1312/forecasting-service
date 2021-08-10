"""Abstract Model."""

from abc import ABC, abstractmethod


class BaseModel(ABC):
    """Abstract class for model implementation."""

    @abstractmethod
    def __init__(self, __C) -> None:
        """
        Initialization.

        Args:
        -------
            __C: Configs
                Configuration object.
        """
        raise NotImplementedError

    @abstractmethod
    def compile(self, optimizer='RMSprop', loss=None, metrics=None) -> None:
        """
        Compile the model.

        Args:
        ------
            optimizer: tf.keras.optimizers (default: RMSprop)
                Optimizer that is used for minimizing the loss function. (e.g., Adam, SGD, RMSprop)
            loss: tf.keras.losses
                Loss function. (e.g., Mean Squared Error, Mean Absolute Error)
            metric: List<tf.keras.metrics>
                Metrics.
        """
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    @abstractmethod
    def fit(self, X_train, y_train, X_val=None, y_val=None, validation_split=0.2):
        """
        Fit the data into the model.

        Args:
        -------
            X_train: array-like
                Training data.
            y_train: array-like that is consistant with X_train
                Training labels.
            X_val: None or array-like
                Validation data.
            y_val: None or array-like that is consistant with X_val
                Validation data.
            validation_split: float (default: 0.2)
                In case (X_val, y_val) is empty, split (X_train, y_train) into train
                and validation datasets. Where validation takes validation_split (%)
                of rows. If validation_split equals 0, we will not evaluate
                validation-related metrics (e.g., val_loss).

        Returns:
        -------
            history: tf.keras.callbacks.History
                A History object. Its History.history attribute is a record of training
                loss values and metrics values at successive epochs, as well as
                validation loss values and validation metrics values (if applicable).
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, X):
        """
        Predict new data.

        Args:
        -------
            X: array-like
                Prediction data.

        Returns:
        -------
            y: array-like
                Predicted labels.
        """
        raise NotImplementedError