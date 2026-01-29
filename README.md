# 🚗 Car Price Prediction Dashboard (EDA + ML + Gradio)

An end-to-end **Exploratory Data Analysis (EDA) and Machine Learning dashboard** built using the **UCI Automobile Dataset (`imports-85.data`)**.

The project was initially prototyped in Jupyter notebooks and later **refactored into a modular, production-style Python codebase**, exposed through a fully interactive **Gradio web application**.

---

## 🔍 Project Overview

This application performs the complete ML pipeline **inside a single Gradio dashboard**:

* Data cleaning and preprocessing of raw automobile data
* Exploratory Data Analysis (EDA) with interactive visualizations
* Training a **Linear Regression** model for car price prediction
* Real-time user interaction for predictions, insights, and filtering

No static plots. No console output. Everything is rendered through the web UI.

---

## 🧠 Machine Learning Details

* **Algorithm:** Linear Regression (`sklearn.linear_model.LinearRegression`)
* **Train/Test Split:** 75% / 25% (`random_state=5`)
* **Target Variable:** `price`

### Features Used

(Identical to the original notebook implementation)

* `normalized_losses`
* `wheel_base`
* `engine_size`
* `bore`
* `stroke`
* `compression_ratio`
* `horsepower`
* `peak_rpm`

Users manually input these values in the Gradio UI to generate live predictions.

---

## 🗂 Project Structure

```
Car_dataset_ETL/
│
├── data/
│   └── raw/
│       └── imports-85.data
│
├── src/
│   ├── __init__.py
│   ├── utils.py              # Data loading & cleaning
│   ├── eda.py                # EDA logic
│   ├── visualization.py      # Gradio-safe visualizations
│   ├── gradio_app.py         # Gradio UI
│   └── ml/
│       ├── __init__.py
│       └── model.py          # Linear Regression model
│
├── main.py                   # Application entry point
├── requirements.txt
└── README.md
```

---

## 📊 Gradio Dashboard Features

### 🔮 Car Price Prediction

* Manual input of vehicle specifications
* Instant price prediction using the trained model

### 📈 Visual Insights

* Price distribution
* Feature correlation heatmap
* Fuel type vs price comparison

### 🎯 Car Filtering

* Filter vehicles by:

  * Price range
  * Fuel type
  * Body style
* View filtered results with summary statistics

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository

```bash
git clone https://github.com/ksn056/2_EDA-with-Pandas-Matplotlib-Seaborn.git
cd Car_dataset_ETL
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the application

```bash
python main.py
```

The Gradio dashboard will launch locally in your browser.

---

## 📁 Dataset

* **Source:** UCI Machine Learning Repository
* **File:** `imports-85.data`
* **Missing values:** Represented as `?`
* **Column names:** Defined manually in code

---

## ✅ Design Choices

* Notebook logic preserved without unnecessary rewrites
* Clear separation of **EDA**, **ML**, and **UI** layers
* Pandas 2.x–safe preprocessing (no chained assignments)
* No local plotting—Gradio-only rendering
* Single entry point (`main.py`) for execution

---

## 🚀 Potential Enhancements

* Display model evaluation metrics in the UI
* Support CSV upload for batch predictions
* Add multiple ML models for comparison
* Save/load trained models

---

## 👤 Author

**Ronit Shetty**
Aspiring AI/ML Engineer

⭐ If you find this project useful, consider starring the repository.

