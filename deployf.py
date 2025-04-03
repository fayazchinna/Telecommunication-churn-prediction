import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pickle import load

# Set page title and icon
st.set_page_config(page_title="Telecom Churn Prediction", page_icon="📞")

# Load the trained model
model = load(open('classify.sav', 'rb'))

# App header
st.markdown(
    """
    <div style="background-color:#f63350;padding:10px">
    <h2 style="color:white;text-align:center;">
    Telecom Churn Prediction App</h2>
    </div>
    """, unsafe_allow_html=True
)

# Function to make predictions
def predict_churn(account_length, voice_messages, voice_plan, intl_plan, intl_calls, intl_mins, intl_charge, 
                  day_calls, day_mins, day_charge, eve_calls, eve_mins, eve_charge, 
                  night_calls, night_mins, night_charge, customer_calls):
    input_data = np.array([[account_length, voice_messages, voice_plan, intl_plan, intl_calls, intl_mins, intl_charge,
                            day_calls, day_mins, day_charge, eve_calls, eve_mins, eve_charge,
                            night_calls, night_mins, night_charge, customer_calls]]).astype(np.float64)
    prediction = model.predict(input_data)
    return prediction

# Streamlit UI
def main():
    st.sidebar.header("User Input Features")

    account_length = st.sidebar.number_input("Account Length", min_value=0, step=1)
    voice_messages = st.sidebar.number_input("Voice Messages", min_value=0, step=1)
    voice_plan = st.sidebar.selectbox("Voice Plan Subscription", [0, 1])
    intl_plan = st.sidebar.selectbox("International Plan Subscription", [0, 1])
    intl_calls = st.sidebar.number_input("Number of International Calls", min_value=0, step=1)
    intl_mins = st.sidebar.number_input("International Minutes", min_value=0.0, format="%.2f")
    intl_charge = st.sidebar.number_input("International Call Charges ($)", min_value=0.0, format="%.2f")
    day_calls = st.sidebar.number_input("Day Calls", min_value=0, step=1)
    day_mins = st.sidebar.number_input("Daytime Minutes Used", min_value=0.0, format="%.2f")
    day_charge = st.sidebar.number_input("Daytime Charge ($)", min_value=0.0, format="%.2f") 
    eve_calls = st.sidebar.number_input("Evening Calls", min_value=0, step=1)
    eve_mins = st.sidebar.number_input("Evening Minutes Used", min_value=0.0, format="%.2f")
    eve_charge = st.sidebar.number_input("Evening Charge ($)", min_value=0.0, format="%.2f")
    night_calls = st.sidebar.number_input("Night Calls", min_value=0, step=1)
    night_mins = st.sidebar.number_input("Night Minutes Used", min_value=0.0, format="%.2f")
    night_charge = st.sidebar.number_input("Night Charge ($)", min_value=0.0, format="%.2f")
    customer_calls = st.sidebar.number_input("Customer Service Calls", min_value=0, step=1)
    
    if st.sidebar.button("Predict Churn"):
        result = predict_churn(account_length, voice_messages, voice_plan, intl_plan, intl_calls, intl_mins, intl_charge,
                               day_calls, day_mins, day_charge, eve_calls, eve_mins, eve_charge,
                               night_calls, night_mins, night_charge, customer_calls)
        
        if result[0] == 1:
            st.error("⚠️ This customer is likely to churn.")
        else:
            st.success("✅ This customer is not likely to churn.")

        # Visualization: Feature Importance Placeholder
        st.subheader("Feature Importance Analysis")
        features = ["Account Length", "Voice Messages", "Voice Plan", "International Plan", "International Calls", 
                    "International Minutes", "International Call Charges", "Day Calls", "Daytime Minutes", "Daytime Charge", 
                    "Evening Calls", "Evening Minutes", "Evening Charge", "Night Calls", "Night Minutes", "Night Charge", "Customer Calls"]
        
        feature_importance = np.random.rand(len(features))  # Placeholder for real feature importance values
        sorted_indices = np.argsort(feature_importance)[::-1]
        sorted_features = [features[i] for i in sorted_indices]
        sorted_importance = feature_importance[sorted_indices]
        
        plt.figure(figsize=(10, 5))
        plt.barh(sorted_features, sorted_importance, color='skyblue')
        plt.xlabel("Importance")
        plt.ylabel("Features")
        plt.title("Feature Importance in Churn Prediction")
        st.pyplot(plt)

if __name__ == '__main__':
    main()
