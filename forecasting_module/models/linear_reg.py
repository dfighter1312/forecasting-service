"""Implementation of basic Linear Regression model."""

from models.base_model import BaseModel
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten


class LinearRegression(BaseModel):
    """Simple Linear Regression implementation."""

    def __init__(self, __C):
        self.__C = __C
        self.model = Sequential()
        self.model.add(Flatten())
        self.model.add(Dense(1, activation='linear'))

    def compile(self, optimizer='RMSprop', loss="mse", metrics="loss"):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def fit(self, X_train, y_train, X_val=None, y_val=None, validation_split=0.2, \
            batch_size=20, epochs=10, callbacks=None):
        try:
            if (X_val is None) or (y_val is None):
                history = self.model.fit(X_train, y_train, validation_split=validation_split, \
                    batch_size=batch_size, epochs=epochs, callbacks=callbacks)
            else:
                history = self.model.fit(X_train, y_train, validation_data=(X_val, y_val), \
                    batch_size=batch_size, epochs=epochs, callbacks=callbacks)
            return history
        except Exception() as e:
            print('Problem during fit method of Linear Regression')
            print(e)

    def predict(self, X):
        return self.model.predict(X)