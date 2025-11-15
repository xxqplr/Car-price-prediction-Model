import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import streamlit as st

# --- Load dataset ---
df_train = pd.read_csv("temp_file.csv")

# Encode categorical features for model training
le_trans = LabelEncoder()
df_train["Trans_encoded"] = le_trans.fit_transform(df_train["Trans"])

le_brand = LabelEncoder()
df_train["brand_encoded"] = le_brand.fit_transform(df_train["brand"])

X = df_train[["Trans_encoded", "Power", "Year", "brand_encoded"]]
y = df_train["Price"]

# Train Random Forest
rf_model = RandomForestRegressor(n_estimators=200, random_state=42)
rf_model.fit(X, y)

# --- Streamlit UI ---
st.title("🚗 Car Price Predictor")

brand = st.selectbox("Select Brand:", df_train["brand"].unique())
trans = st.selectbox("Transmission:", df_train["Trans"].unique())
power = st.number_input("Power (CC):", min_value=500, max_value=5000, value=1600)
year = st.number_input("Year:", min_value=2000, max_value=2025, value=2023)

if st.button("Predict Price"):
    # Encode user input
    input_df = pd.DataFrame({
        "Trans_encoded": [le_trans.transform([trans])[0]],
        "Power": [power],
        "Year": [year],
        "brand_encoded": [le_brand.transform([brand])[0]]
    })

    predicted_price = rf_model.predict(input_df)[0]
    st.success(f"Predicted Car Price: {predicted_price:,.0f} JD")
