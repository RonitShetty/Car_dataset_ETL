import gradio as gr
import pandas as pd

from src.eda import describe, missing_values
from src.visualization import (
    price_distribution,
    correlation_heatmap,
    fueltype_price_box
)
from src.ml import train_model, predict_car_price

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

def launch_gradio(df):

    model = train_model(df)

    def show_describe():
        return describe(df)

    def show_missing():
        mv = missing_values(df)
        return mv.reset_index().rename(
            columns={"index": "column", 0: "missing"}
        )

    def filter_cars(min_price, max_price, fuel_type, body_style):
        filtered = df[(df["price"] >= min_price) & (df["price"] <= max_price)]
        if fuel_type != "All":
            filtered = filtered[filtered["fuel_type"] == fuel_type]
        if body_style != "All":
            filtered = filtered[filtered["body_style"] == body_style]

        stats = {
            "Total Cars": len(filtered),
            "Petrol Cars": len(filtered[filtered["fuel_type"] == "gas"]),
            "Diesel Cars": len(filtered[filtered["fuel_type"] == "diesel"]),
        }

        return (
            filtered[["make", "body_style", "fuel_type", "engine_size", "horsepower", "price"]],
            pd.DataFrame(stats.items(), columns=["Category", "Count"])
        )

    with gr.Blocks(theme=gr.themes.Soft(primary_hue="orange")) as app:

        gr.Markdown("""
        <div style='text-align:center;'>
            <h1 style='color:#FF7F00;'>🚗 Car Price Prediction Dashboard</h1>
        </div>
        """)

        with gr.Tab("🔮 Predict Car Price"):
            with gr.Row():
                nl = gr.Number(label="Normalized Losses", value=100)
                wb = gr.Number(label="Wheel Base", value=95)
                es = gr.Number(label="Engine Size", value=130)
                bore = gr.Number(label="Bore", value=3.2)

            with gr.Row():
                stroke = gr.Number(label="Stroke", value=3.0)
                cr = gr.Number(label="Compression Ratio", value=9.0)
                hp = gr.Number(label="Horsepower", value=120)
                rpm = gr.Number(label="Peak RPM", value=5200)

            predict_btn = gr.Button("🔍 Predict Now", variant="primary")
            output_price = gr.Number(label="Predicted Price (USD)")

            predict_btn.click(
                fn=lambda *args: predict_car_price(model, *args),
                inputs=[nl, wb, es, bore, stroke, cr, hp, rpm],
                outputs=output_price
            )

        with gr.Tab("📊 Visual Insights"):
            chart = gr.Dropdown(
                choices=["Price Distribution", "Correlation Heatmap", "Fuel Type vs Price"],
                value="Price Distribution"
            )
            plot = gr.Image()

            chart.change(
                fn=lambda c: (
                    price_distribution(df) if c == "Price Distribution"
                    else correlation_heatmap(df, FEATURES)
                    if c == "Correlation Heatmap"
                    else fueltype_price_box(df)
                ),
                inputs=chart,
                outputs=plot
            )

        with gr.Tab("🎯 Filter Cars"):
            min_p = gr.Number(value=5000, label="Minimum Price")
            max_p = gr.Number(value=20000, label="Maximum Price")
            fuel = gr.Dropdown(["All"] + sorted(df["fuel_type"].dropna().unique()), value="All")
            body = gr.Dropdown(["All"] + sorted(df["body_style"].dropna().unique()), value="All")
            btn = gr.Button("Filter")

            table = gr.Dataframe()
            stats = gr.Dataframe()

            btn.click(
                fn=filter_cars,
                inputs=[min_p, max_p, fuel, body],
                outputs=[table, stats]
            )

    app.launch()
