import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
from scipy import stats

st.set_page_config(page_title="Health Predictor App", layout="wide")

# Load the data
@st.cache_data
def load_data():
    #data = pd.read_csv('new_merged.csv')

    url = "https://raw.githubusercontent.com/LaxmiVatsalyaDaita/CMSE830/main/new_merged.csv"
    data = pd.read_csv(url, delimiter=",")
    df = data.copy()
    df.drop(['id', 'age', 'gender', 'age_years', 'BMI', 'height', 'weight'], axis=1, inplace=True)
    df = df.rename(columns={"cardio": "heart_risk"})
    return df

df = load_data()

# Create tabs for navigation
with st.container():
    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Risk Prediction", "BMI Calculator", "Tips for a better"])

    with tab1:
        st.markdown("""
        # 💤 Why Sleep and Nutrition Matter for Overall Well-Being 🥦

        Sleep and nutrition are two fundamental pillars of health that significantly impact our physical, mental, and emotional well-being. Understanding the relationship between sleep and various health factors can empower individuals to make informed lifestyle choices that enhance their quality of life.

        ## The Role of Sleep

        Sleep is a vital process that allows the body to recover, rejuvenate, and maintain optimal function. Adequate sleep is essential for various reasons:

        - **Physical Health:** Quality sleep supports immune function, hormonal balance, and metabolic health. It plays a crucial role in tissue repair, muscle growth, and overall bodily recovery.

        - **Mental Clarity:** Sleep is integral to cognitive processes, including memory consolidation, problem-solving, and decision-making. A well-rested brain is more alert and capable of focusing on tasks.

        - **Emotional Regulation:** Sleep influences mood and emotional resilience. Insufficient sleep can lead to irritability, anxiety, and increased stress levels, making it harder to cope with daily challenges.

        - **Chronic Disease Prevention:** Poor sleep patterns are linked to a range of chronic conditions, including obesity, diabetes, cardiovascular diseases, and mental health disorders. Prioritizing sleep can help mitigate these risks.

        ## The Importance of Nutrition

        Nutrition provides the essential nutrients needed for the body to function effectively. A balanced diet is crucial for overall health and well-being:

        - **Energy and Vitality:** The food we consume fuels our bodies, providing the energy required for daily activities. Proper nutrition ensures that we have the stamina to engage in physical and mental tasks.

        - **Disease Prevention:** A well-rounded diet rich in fruits, vegetables, whole grains, and lean proteins can lower the risk of chronic diseases and promote overall health. Nutrient-dense foods support the immune system and reduce inflammation.

        - **Mental Health:** Emerging research suggests a strong connection between nutrition and mental health. Nutrients such as omega-3 fatty acids, vitamins, and minerals have been linked to improved mood and cognitive function.

        - **Healthy Sleep Patterns:** Certain foods and nutrients can influence sleep quality. For example, foods rich in magnesium and tryptophan can promote better sleep, while excessive caffeine or sugar can disrupt it.

        The synergy between sleep and nutrition is crucial for achieving and maintaining overall well-being. By recognizing the importance of these elements, individuals can take proactive steps to improve their health and enhance their quality of life. This dashboard serves as a resource to explore insights related to sleep and nutrition, empowering users to make informed choices that foster a healthier lifestyle.
        """
        )

    with tab2:
        import streamlit as st
        import numpy as np

        # Set page configuration and custom styles
        #st.set_page_config(page_title="Health Predictor App", layout="wide")

        # Custom CSS for styling
        st.markdown("""
            <style>
                .stApp {
                    background-color: #f0f4f8;
                    color: #2c3e50;
                }
                .header-title {
                    text-align: center;
                    font-size: 36px;
                    color: #34495e;
                    margin-bottom: 10px;
                }
                .sub-header {
                    font-size: 28px;
                    color: #2980b9;
                }
                .container {
                    background-color: #ffffff;
                    padding: 20px;
                    border-radius: 15px;
                    box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
                    margin-bottom: 20px;
                }
                .prediction-section {
                    background-color: #ecf0f1;
                    padding: 15px;
                    border-radius: 10px;
                    text-align: center;
                }
                .stButton>button {
                    background-color: #3498db;
                    color: white;
                    border-radius: 10px;
                    padding: 10px 20px;
                }
                .stButton>button:hover {
                    background-color: #2980b9;
                }
                .emoji {
                    font-size: 40px;
                }
            </style>
        """, unsafe_allow_html=True)

        # App Title
        st.markdown('<div class="header-title">🩺 Health Predictor App</div>', unsafe_allow_html=True)
        st.write("Welcome to the **Health Predictor App**! 🌟 Let's dive in and unlock some insights about your health. Ready? Let's go! 🚀")

        # User Inputs for Risk Predictions
        with st.container():
            st.markdown('<div class="sub-header">Health Risk Predictions 🚀</div>', unsafe_allow_html=True)

            col1, col2 = st.columns(2)
            with col1:
                age = st.number_input('🔹 Age', min_value=10, max_value=100, value=30, help="How old are you? 🎂")
                gender = st.selectbox('🔹 Gender', ['Female', 'Male'], help="Select your gender. 👩👨")
                sleep_duration = st.number_input('🔹 Sleep Duration (hours)', min_value=0.0, max_value=12.0, value=7.0, help="How long do you sleep each night? 😴")
                stress_level = st.slider('🔹 Stress Level (1-10)', min_value=1, max_value=10, value=5, help="Rate your stress level. 🧘‍♂️")
                physical_activity = st.number_input('🔹 Physical Activity Level', min_value=0, max_value=100, value=50, help="Enter your physical activity level. 🏃‍♀️")

            with col2:
                daily_steps = st.number_input('🔹 Daily Steps', min_value=0, max_value=30000, value=5000, help="How many steps do you take daily? 👟")
                heart_rate = st.number_input('🔹 Heart Rate (bpm)', min_value=40, max_value=180, value=72, help="Enter your resting heart rate. ❤️")
                bp_upper = st.number_input('🔹 Systolic BP', min_value=80, max_value=200, value=120, help="Your systolic blood pressure. 🩸")
                bp_lower = st.number_input('🔹 Diastolic BP', min_value=60, max_value=130, value=80, help="Your diastolic blood pressure. 💉")
                cholesterol = st.number_input('🔹 Cholesterol Level', min_value=0.0, max_value=10.0, value=2.0, help="Your cholesterol level. 🥩")

            if st.button("🌟 Predict Health Risks"):
                # Placeholder for custom model predictions
                sleep_pred = "Custom Sleep Model Output"  # Replace with your model logic
                heart_pred = "Custom Heart Model Output"  # Replace with your model logic
                st.markdown(f'<div class="prediction-section">🛌 Predicted Sleep Disorder: <b>{sleep_pred}</b></div>', unsafe_allow_html=True)
                st.markdown(f'<div class="prediction-section">❤️ Predicted Heart Risk: <b>{heart_pred}</b></div>', unsafe_allow_html=True)

    with tab3:
        with st.container():
            st.markdown('<div class="sub-header">BMI Calculator 📏</div>', unsafe_allow_html=True)
            st.write("Let’s calculate your **Body Mass Index (BMI)** and see how you measure up! 📊")

            col3, col4 = st.columns(2)
            with col3:
                height = st.number_input('📏 Height (cm)', min_value=100.0, max_value=250.0, value=170.0, help="Enter your height in centimeters. 📏")
            with col4:
                weight = st.number_input('⚖️ Weight (kg)', min_value=30.0, max_value=200.0, value=70.0, help="Enter your weight in kilograms. ⚖️")

            if st.button("📊 Calculate BMI"):
                bmi = weight / ((height / 100) ** 2)
                st.markdown(f'<div class="prediction-section">📝 Your BMI: <b>{bmi:.2f}</b></div>', unsafe_allow_html=True)
                if bmi < 18.5:
                    st.write("⚠️ Category: Underweight")
                elif 18.5 <= bmi < 24.9:
                    st.write("✅ Category: Normal weight")
                elif 25 <= bmi < 29.9:
                    st.write("⚠️ Category: Overweight")
                else:
                    st.write("⚠️ Category: Obesity")
                st.success("Remember, health is a journey! Keep going strong! 💪")
                st.toast("BMI calculation complete! 🎉")
