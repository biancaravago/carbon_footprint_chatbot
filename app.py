import streamlit as st
import pandas as pd
import joblib
import re

# Load model and encoders
model = joblib.load("carbon_footprint_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")
feature_encoders = joblib.load("feature_encoders.pkl")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Store user inputs here
inputs = {}

# Define expected questions and keys
questions = [
    ("How many miles do you drive per week?", "miles_per_week"),
    ("How often do you eat red meat?", "meat_freq"),
    ("How do you heat your home?", "heating_type"),
    ("Do you recycle or compost?", "recycles"),
    ("Do you own energy-efficient appliances?", "efficient_appliances")
]

# Track question progress
if "question_index" not in st.session_state:
    st.session_state.question_index = 0

st.title("🌍 Personal Carbon Footprint Chatbot")

# Display past messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Ask the next question
if st.session_state.question_index < len(questions):
    q_text, q_key = questions[st.session_state.question_index]
    with st.chat_message("assistant"):
        st.markdown(q_text)
    
    user_input = st.chat_input("Your answer:")
    
    if user_input:
        # Show user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Save answer
        inputs[q_key] = user_input.strip()
        st.session_state.messages.append({"role": "assistant", "content": f"Got it: {user_input}"})
        
        # Move to next question
        st.session_state.question_index += 1
        st.rerun()
else:
    # All questions answered
    st.markdown("Thanks! Calculating your carbon footprint...")

    # Build input DataFrame
    input_df = pd.DataFrame([inputs])

    # Encode input data
    for col in input_df.columns:
        le = feature_encoders.get(col)
        if le:
            try:
                input_df[col] = le.transform(input_df[col])
            except:
                input_df[col] = le.transform([le.classes_[0]])  # fallback to default if unseen value

    # Predict
    prediction = model.predict(input_df)[0]
    category = label_encoder.inverse_transform([prediction])[0]

    with st.chat_message("assistant"):
        st.subheader(f"🧾 Your Carbon Footprint: **{category}**")

        tips = {
            "Low": ["You're doing great! Consider helping others reduce theirs."],
            "Medium": ["Try cutting meat 2x/week.", "Walk or bike once a week instead of driving."],
            "High": ["Switch to public transport if possible.", "Reduce heating usage or switch to electric."]
        }

        st.write("✅ Recommended Actions:")
        for tip in tips.get(category, []):
            st.write(f"- {tip}")
