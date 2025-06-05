import streamlit as st
import pandas as pd
import joblib

# Load model and encoders
model = joblib.load("carbon_footprint_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")
feature_encoders = joblib.load("feature_encoders.pkl")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "question_index" not in st.session_state:
    st.session_state.question_index = 0
if "answers" not in st.session_state:
    st.session_state.answers = {}

# Define questions
questions = [
    ("How many miles do you drive per week?", "miles_per_week"),
    ("How often do you eat red meat?", "meat_freq"),
    ("How do you heat your home?", "heating_type"),
    ("Do you recycle or compost?", "recycles"),
    ("Do you own energy-efficient appliances?", "efficient_appliances")
]

st.title("🌍 Personal Carbon Footprint Chatbot")

# Display message history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Ask next question
if st.session_state.question_index < len(questions):
    q_text, q_key = questions[st.session_state.question_index]
    with st.chat_message("assistant"):
        st.markdown(q_text)

    user_input = st.chat_input("Your answer:")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.answers[q_key] = user_input.strip()
        st.session_state.question_index += 1
        st.rerun()

# All questions answered
else:
    with st.chat_message("assistant"):
        st.markdown("Thanks! Calculating your carbon footprint...")

        df = pd.DataFrame([st.session_state.answers])

        # Normalization map
        normalized_map = {
            "meat_freq": {
                "never": "Never", "1-3": "1-3x/week", "few": "1-3x/week", 
                "4-7": "4-7x/week", "daily": "Daily"
            },
            "heating_type": {
                "gas": "Gas", "with gas": "Gas", "natural gas": "Gas", 
                "electric": "Electric", "wood": "Wood", "none": "None"
            },
            "recycles": {
                "always": "Always", "sometimes": "Sometimes", "never": "Never"
            },
            "efficient_appliances": {
                "yes": "Yes", "no": "No"
            }
        }

        # Encoding loop
        for col in df.columns:
            if col == "miles_per_week":
                try:
                    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
                except:
                    df[col] = 0
                continue

            le = feature_encoders.get(col)
            if le:
                try:
                    raw_val = df[col].values[0].strip().lower()
                    choices = normalized_map.get(col, {})
                    matched_val = next((v for k, v in choices.items() if k in raw_val), le.classes_[0])
                    df[col] = le.transform([matched_val])
                except:
                    df[col] = le.transform([le.classes_[0]])

        # Predict
        prediction = model.predict(df)[0]
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
