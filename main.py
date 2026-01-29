from src.utils import load_data, basic_cleaning
from src.gradio_app import launch_gradio

RAW_DATA_PATH = "data/raw/imports-85.data"

def main():
    df = load_data(RAW_DATA_PATH)
    df = basic_cleaning(df)
    launch_gradio(df)

if __name__ == "__main__":
    main()
