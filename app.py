import streamlit as st
import pandas as pd
import joblib

# Load saved model & columns
model = joblib.load('house_price_model.pkl')
model_columns = joblib.load('model_columns.pkl')

st.title("🏡 House Price Prediction App")
st.write("Enter house features to get an estimated market price:")

# Inputs for property details
bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=3)
bathrooms = st.number_input("Bathrooms", min_value=1.0, max_value=10.0, value=2.0)
sqft_living = st.number_input("Living Area (sqft)", min_value=300, max_value=10000, value=2000)
sqft_lot = st.number_input("Lot Size (sqft)", min_value=500, max_value=50000, value=5000)
floors = st.number_input("Floors", min_value=1.0, max_value=4.0, value=1.0)
waterfront = st.selectbox("Waterfront", [0, 1])
view = st.slider("View Rating (0-4)", 0, 4, 0)
condition = st.slider("Condition Rating (1-5)", 1, 5, 3)
sqft_above = st.number_input("Above Ground (sqft)", min_value=300, max_value=10000, value=1500)
sqft_basement = st.number_input("Basement (sqft)", min_value=0, max_value=5000, value=500)
yr_built = st.number_input("Year Built", min_value=1900, max_value=2026, value=1995)
city = st.text_input("City", value="Seattle")

if st.button("Predict Price"):
    input_data = pd.DataFrame([{
        'bedrooms': bedrooms, 'bathrooms': bathrooms, 'sqft_living': sqft_living,
        'sqft_lot': sqft_lot, 'floors': floors, 'waterfront': waterfront,
        'view': view, 'condition': condition, 'sqft_above': sqft_above,
        'sqft_basement': sqft_basement, 'yr_built': yr_built, 'city': city
    }])
    
    input_encoded = pd.get_dummies(input_data, columns=['city'], drop_first=True)
    input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)
    
    predicted_price = model.predict(input_encoded)[0]
    st.success(f"Estimated Price: **${predicted_price:,.2f}**")
