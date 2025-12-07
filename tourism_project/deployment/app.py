import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(repo_id="armakar123/tourism-taken", filename="best_tourism_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI for Tourism Package Prediction
st.title("Tourism Package Prediction")
st.write("""
This application predicts whether a customer will purchase the newly introduced Wellness Tourism Package before contacting them.
Please enter the Customer and tourism data below to get a prediction.

""")

# User input
Age = st.number_input("Age", min_value=18, max_value=65, value=18)
TypeofContact = st.selectbox("TypeofContact", ['Self Enquiry','Company Invited'])
CityTier = st.selectbox("CityTier", min_value=1, max_value=3, value=324.0, step=1)
DurationOfPitch = st.number_input("DurationOfPitch", min_value=5, max_value=130, value=5)
Occupation = st.selectbox("Occupation", ['Salaried','Self Employed','Business','Free Lancer'])
NumberOfPersonVisiting = st.number_input("NumberOfPersonVisiting", min_value=1, max_value=10, value=1)
NumberOfFollowups = st.number_input("NumberOfFollowups", min_value=1, max_value=10, value=1)
ProductPitched = st.selectbox("ProductPitched", ['Basic','Deluxe','Standard'.'King','Super Deluxe'])
PreferredPropertyStar= st.number_input("PreferredPropertyStar", min_value=1, max_value=5, value=1)
MaritalStatus= st.selectbox("MaritalStatus", ['Married','Single','Divorced'])
NumberOfTrips= st.number_input("NumberOfTrips", min_value=1, max_value=10, value=1)
Passport= st.selectbox("Passport", ['Yes','No'])
PitchSatisfactionScore= st.number_input("PitchSatisfactionScore", min_value=1, max_value=5, value=1)
OwnCar= st.selectbox("OwnCar", ['Yes','No'])
NumberOfChildrenVisiting= st.number_input("NumberOfChildrenVisiting", min_value=0, max_value=10, value=0)
Designation= st.selectbox("Designation", ['Executive','Senior Manager','Manager','AVP','VP']
MonthlyIncome= st.number_input("Monthly Income",min_value=0, max_value=500000, value=1000]


# Assemble input into DataFrame
input_data = pd.DataFrame([{
    'Age' : Age,
    'TypeofContact': TypeofContact,
    'CityTier': CityTier,
    'DurationOfPitch': DurationOfPitch,
    'Occupation': Occupation,
    'NumberOfPersonVisiting': NumberOfPersonVisiting,
    'NumberOfFollowups': NumberOfFollowups
    'ProductPitched': ProductPitched
    'PreferredPropertyStar': PreferredPropertyStar,
    'MaritalStatus': MaritalStatus,
    'NumberOfTrips': NumberOfTrips,
    'Passport': Passport,
    'PitchSatisfactionScore': PitchSatisfactionScore,
    'OwnCar': OwnCar,
    'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
    'Designation': Designation,
    'MonthlyIncome': MonthlyIncome
}])


if st.button("Tourism Product Taken"):
    prediction = model.predict(input_data)[0]
    result = "Product Taken" if prediction == 1 else " Product Not Taken"
    st.subheader("Tourism Package Prediction")
    st.success(f"The model predicts: **{result}**")
