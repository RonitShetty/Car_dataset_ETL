import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

FEATURES = [
    "normalized_losses",
    "wheel_base",
    "engine_size",
    "bore",
    "stroke",
    "compression_ratio",
    "horsepower",
    "peak_rpm"
]

def train_model(df):
    df = df.copy()

    numeric_cols = FEATURES + ["price"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].mean())

    X = df[FEATURES]
    y = df["price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=5
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    return model

def predict_car_price(
    model,
    normalized_losses,
    wheel_base,
    engine_size,
    bore,
    stroke,
    compression_ratio,
    horsepower,
    peak_rpm
):
    time.sleep(1.5)
    input_data = np.array([[normalized_losses, wheel_base, engine_size,
                            bore, stroke, compression_ratio,
                            horsepower, peak_rpm]])
    return round(model.predict(input_data)[0], 2)
