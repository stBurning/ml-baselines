import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.base import BaseEstimator

# Creating FastAPI instance
app = FastAPI()


# Класс, описывающий входные данные
class RequestData(BaseModel):
    x: float
    y: float


class BasicModel(BaseEstimator):
    """Реализация модели (для примера)"""
    def __init__(self, a: float):
        self.a = a
        self.b = None
        self.is_fitted_ = False

    def fit(self, X, y=None):
        self.is_fitted_ = True
        self.b = 2
        return self

    def predict(self, X):
        return np.sum(X, axis=1) * self.a / self.b


model = BasicModel(2)
model.fit([[1, 1]])


@app.post('/predict')
def predict(data: RequestData):
    test_data = [[
        data.x,
        data.y
    ]]
    print(test_data)

    result = model.predict(test_data)[0]
    return {'predict': result}


