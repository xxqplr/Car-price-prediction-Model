import gradio as gr
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# ---- Load & preprocess data ----
def load_and_train():
    df = pd.read_csv("car_prices_jordan.csv")

    # Extract brand
    df["brand"] = df["Model"].str.split(" ").str[0]
    df.rename(columns={'Unnamed: 0': 'Index', 'Property': 'Trans'}, inplace=True)

    # Fix Power column
    df['Power'] = df['Power'].str.extract(r'(\d+)', expand=False)
    avg_power = int(round(df['Power'].dropna().astype(float).mean()))
    df['Power'] = df['Power'].replace("0", avg_power).astype(int)

    # Extract year
    df["Year"] = df["Model"].str.extract(r'(\d{4})$', expand=False)
    avg_year = int(df["Year"].dropna().astype(int).mean())
    df["Year"] = df["Year"].fillna(avg_year)

    # Convert price to numeric
    df["Price"] = df["Price"].str.replace(",", "").astype(int)

    # Clean transmission
    df["Trans"] = df["Trans"].str.capitalize().str.split(" ").str[0]

    # Encode
    le_trans = LabelEncoder()
    le_brand = LabelEncoder()

    df["Trans"] = le_trans.fit_transform(df["Trans"])
    df["brand"] = le_brand.fit_transform(df["brand"])

    # Train model
    X = df[["Trans", "Power", "Year", "brand"]]
    Y = df["Price"]

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X, Y)

    return df, model, le_trans, le_brand

df, model, le_trans, le_brand = load_and_train()

# ---- Prediction Function ----
def predict_car_price(brand, power, year, trans):
    try:
        brand_encoded = le_brand.transform([brand])[0]
        trans_encoded = le_trans.transform([trans])[0]

        new_data = pd.DataFrame([[trans_encoded, power, year, brand_encoded]],
                                columns=["Trans", "Power", "Year", "brand"])
        prediction = model.predict(new_data)[0]
        return f"Estimated price: {round(prediction)} JD"
    except:
        return "Invalid input or unseen category."

# ---- Build UI ----
brand_choices = sorted(df["brand"].unique())
trans_choices = sorted(df["Trans"].unique())
trans_labels = le_trans.inverse_transform(trans_choices)

inputs = [
    gr.Dropdown(brand_choices, label="Brand"),
    gr.Number(label="Engine Power (CC)", value=1600),
    gr.Number(label="Year", value=2021),
    gr.Dropdown(list(trans_labels), label="Transmission")
]

output = gr.Textbox(label="Predicted Price")

app = gr.Interface(
    fn=predict_car_price,
    inputs=inputs,
    outputs=output,
    title="🚗 Car Price Prediction (Jordan Market)",
    description="AI model trained using real car market data.",
    theme="soft",
)

app.launch()
