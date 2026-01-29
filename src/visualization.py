import matplotlib.pyplot as plt
import seaborn as sns
import os

PLOT_DIR = "data/processed/plots"
os.makedirs(PLOT_DIR, exist_ok=True)

def save_plot(fig, name):
    path = f"{PLOT_DIR}/{name}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path

def price_distribution(df):
    fig, ax = plt.subplots()
    sns.histplot(df["price"], kde=True, ax=ax, color="orange")
    ax.set_title("Car Price Distribution")
    return save_plot(fig, "price_distribution")

def correlation_heatmap(df, features):
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df[features + ["price"]].corr(), annot=True, cmap="Oranges", ax=ax)
    ax.set_title("Feature Correlation Heatmap")
    return save_plot(fig, "correlation_heatmap")

def fueltype_price_box(df):
    fig, ax = plt.subplots()
    sns.boxplot(data=df, x="fuel_type", y="price", palette="Oranges", ax=ax)
    ax.set_title("Fuel Type vs Price")
    return save_plot(fig, "fueltype_price_box")
