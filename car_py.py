import pandas as pd 
import numpy as np 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
import matplotlib.pyplot as plt


def data_process():    
    
    df_unf = pd.read_csv('car_prices_jordan.csv')

    # Data Cleaning and Preprocessing
    df_unf["brand"]=df_unf["Model"].str.split(" ").str[0]
    df_unf.rename(columns={'Unnamed: 0': 'Index'}, inplace=True)
    df_unf.rename(columns={'Property': 'Trans'}, inplace=True)



    # Clean Power column to keep only numeric values + 0 to mean

    #df_unf["Power"]=df_unf["Power"].str.replace(" CC","") # remoce CC from Power col 

    #avg_pwr=round(df_unf["Power"].dropna().astype(float).mean())

    # 1. Extract the number using Pandas's str.extract()
    # The pattern r'(\d+)' means: find and capture the FIRST sequence of one or more digits.
    df_unf['Power'] = df_unf['Power'].str.extract(r'(\d+)', expand=False)

    # 2. Impute the mean and convert to final integer type
    # This handles the remaining missing values (like those that originally said '0 CC')
    avg_power = int(round(df_unf['Power'].dropna().astype(float).mean()))
    print(avg_power)
    df_unf['Power'] = df_unf['Power'].replace("0",avg_power).astype(int)





    # Extract Year from Model column + plaaaceholder to be AVG of all years
    df_unf["Year"]=df_unf["Model"].str.extract(r'(\d{4})$', expand=False)
    avg_year = int(df_unf["Year"].dropna().astype(int).mean())
    df_unf["Year"] = df_unf["Year"].fillna(avg_year)



    # Convert Price to numeric, removing any non-numeric characters
    df_unf["Price"] = df_unf["Price"].str.replace(",","").astype(int)

    #Trans to only have Automatic and Manual + Capitalize first letter
    df_unf["Trans"]=df_unf["Trans"].str.capitalize()
    df_unf["Trans"]=df_unf["Trans"].str.split(" ").str[0]

    print(avg_power)
    print(df_unf.head())




    # Save the new dataSet to temp CSV File! 
    df_unf.to_csv("temp_file.csv", index=False)  
    return df_unf


def encode_df(df_unf):
    df=df_unf.drop(columns=["Index","Model"])

    # start encoding categorical features
    le=LabelEncoder()
    for col in ["Trans","brand"]:
        df[col]=le.fit_transform(df[col])
    print(df.head())
    return df

def model(df1):

    X = df1[["Trans","Power","Year","brand"]]
    Y = df1["Price"]

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Linear Regression
    lin = LinearRegression()
    lin.fit(x_train, y_train)
    y_pred_lin = lin.predict(x_test)

    # Random Forest
    rf = RandomForestRegressor(n_estimators=200, random_state=42)
    rf.fit(x_train, y_train)
    y_pred_rf = rf.predict(x_test)

    # Compare performance
    print("\n Linear Regression:")
    print("MSE:", mean_squared_error(y_test, y_pred_lin))
    print("R2:", r2_score(y_test, y_pred_lin))
    print("MAE:", mean_absolute_error(y_test, y_pred_lin))

    print("\n Random Forest:")
    print("MSE:", mean_squared_error(y_test, y_pred_rf))
    print("R2:", r2_score(y_test, y_pred_rf))
    print("MAE:", mean_absolute_error(y_test, y_pred_rf))


    return lin, rf


# CHAT GPT ADDED THIS CLASS TO HANDLE PREDICTIONS
class CarPricePredictor:
    def __init__(self, df_train, rf_model):
        self.df_train = df_train
        self.rf_model = rf_model
        # Fit encoders once
        self.le_trans = LabelEncoder()
        self.le_trans.fit(df_train["Trans"])
        self.le_brand = LabelEncoder()
        self.le_brand.fit(df_train["brand"])
    
    def predict(self, brand, power, year, trans):
        # Create input DataFrame
        new_car = pd.DataFrame({
            "Trans": [trans],
            "Power": [power],
            "Year": [year],
            "brand": [brand]
        })
        # Encode
        new_car["Trans"] = self.le_trans.transform(new_car["Trans"])
        new_car["brand"] = self.le_brand.transform(new_car["brand"])
        # Predict
        price = self.rf_model.predict(new_car)[0]
        return round(price, 0)


# Main function call
df0=data_process()
df1=encode_df(df0)
final=model(df1)
final

# Initialize predictor once (after training)   CHAT GPT 
predictor = CarPricePredictor(df0, final[1])  # final[1] is Random Forest

# Predict any car
price = predictor.predict(brand="Changan", power=1600, year=2021, trans="Automatic")
print(f"Predicted price: {price} JD")


print(df1.describe())



