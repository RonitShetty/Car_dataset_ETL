import pandas as pd
import numpy as np

COLUMNS = [
    "symboling", "normalized_losses", "make", "fuel_type", "aspiration",
    "num_doors", "body_style", "drive_wheels", "engine_location",
    "wheel_base", "length", "width", "height", "curb_weight",
    "engine_type", "num_cylinders", "engine_size", "fuel_system",
    "bore", "stroke", "compression_ratio", "horsepower", "peak_rpm",
    "city_mpg", "highway_mpg", "price"
]

def load_data(path):
    return pd.read_csv(path, names=COLUMNS)

def basic_cleaning(df):
    df = df.copy()
    df.replace("?", np.nan, inplace=True)

    numeric_cols = ["normalized_losses", "bore", "stroke", "horsepower", "peak_rpm", "price"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].mean())

    return df
