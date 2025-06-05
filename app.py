import streamlit as st
import pandas as pd
import joblib

# Load model and encoders
model = joblib.load("carbon_footprint_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")
feature_encoders = joblib.load("feature_encoders.pkl")

# Streamlit app
st.title("🌍 Personal Carbon Footprint Chatbot")
st.write("Answer a few questions to estimate your carbon footprint and get personalized tips.")

# User inputs
miles = st.selectbox("How many miles do you drive per week?", [0, 25, 100, 200, 300])
meat = st.selectbox("How often do you eat red meat?", ["Never", "1-3x/week", "4-7x/week", "Daily"])
heating = st.selectbox("How do you heat your home?", ["None", "Electric", "Gas", "Wood"])
recycles = st.selectbox("Do you recycle or compost?", ["Always", "Sometimes", "Never"])
efficient = st.selectbox("Do you own energy-efficient appliances?", ["Yes", "No"])

if st.button("Predict My Carbon Footprint"):
    # Prepare input DataFrame
    input_data = pd.DataFrame([{
        "miles_per_week": miles,
        "meat_freq": meat,
        "heating_type": heating,
        "recycles": recycles,
        "efficient_appliances": efficient
    }])

    # Encode input data
    for col in input_data.columns:
        le = feature_encoders.get(col)
        if le:
            input_data[col] = le.transform(input_data[col])

    # Predict
    prediction = model.predict(input_data)[0]
    category = label_encoder.inverse_transform([prediction])[0]

    st.subheader(f"🧾 Your Carbon Footprint: **{category}**")

    tips = {
        "Low": ["You're doing great! Consider helping others reduce theirs."],
        "Medium": ["Try cutting meat 2x/week.", "Walk or bike once a week instead of driving."],
        "High": ["Switch to public transport if possible.", "Reduce heating usage or switch to electric."]
    }

    st.write("✅ Recommended Actions:")
    for tip in tips.get(category, []):
        st.write(f"- {tip}")
