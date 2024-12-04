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
    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Risk Prediction", "BMI Calculator", "Ask for Nutrition Tips"])

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

        # Custom CSS for styling
        st.markdown("""
            <style>
                /* Modern Color Palette */
                :root {
                    --primary-color: #2c3e50;      /* Deep Blue-Gray */
                    --secondary-color: #3498db;    /* Bright Blue */
                    --accent-color: #2ecc71;       /* Vibrant Green */
                    --background-color: #f4f6f7;   /* Soft Gray */
                    --card-color: #ffffff;         /* Pure White */
                    --text-color: #34495e;         /* Charcoal */
                }

                /* Global Styling */
                .stApp {
                    background-color: var(--background-color);
                    color: var(--text-color);
                    font-family: 'Inter', 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                }

                /* Typography */
                .header-title {
                    text-align: center;
                    font-size: 2.5rem;
                    color: var(--primary-color);
                    font-weight: 700;
                    margin-bottom: 1.5rem;
                    letter-spacing: -1px;
                }

                /* Section Headers */
                .group-header {
                    font-size: 1.4rem;
                    color: var(--secondary-color);
                    margin-bottom: 1rem;
                    padding-bottom: 10px;
                    border-bottom: 3px solid var(--accent-color);
                    font-weight: 600;
                }

                /* Container Styling */
                .stContainer {
                    background-color: var(--card-color);
                    border-radius: 15px;
                    padding: 20px;
                    box-shadow: 0 6px 12px rgba(0, 0, 0, 0.08);
                    margin-bottom: 15px;
                    transition: all 0.3s ease;
                }

                .stContainer:hover {
                    transform: translateY(-3px);
                    box-shadow: 0 8px 15px rgba(0, 0, 0, 0.1);
                }

                /* Input Styling */
                .stTextInput > div > div > input, 
                .stNumberInput > div > div > input, 
                .stSelectbox > div > div > div {
                    border-radius: 10px !important;
                    border: 1.5px solid #e0e0e0 !important;
                    padding: 10px 15px !important;
                    transition: all 0.3s ease;
                    font-size: 0.9rem;
                }

                .stTextInput > div > div > input:focus, 
                .stNumberInput > div > div > input:focus, 
                .stSelectbox > div > div > div:focus {
                    border-color: var(--secondary-color) !important;
                    box-shadow: 0 0 0 3px rgba(52, 152, 219, 0.15) !important;
                }

                /* Button Styling */
                .stButton > button {
                    background-color: var(--secondary-color) !important;
                    color: white !important;
                    border-radius: 12px !important;
                    padding: 12px 25px !important;
                    font-weight: 600 !important;
                    transition: all 0.3s ease !important;
                    text-transform: uppercase;
                    letter-spacing: 1px;
                }

                .stButton > button:hover {
                    background-color: var(--primary-color) !important;
                    transform: translateY(-2px);
                    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
                }

                /* Prediction Section */
                .prediction-section {
                    background-color: var(--accent-color);
                    color: white;
                    border-radius: 12px;
                    padding: 15px;
                    margin-top: 20px;
                    text-align: center;
                    font-weight: 600;
                    letter-spacing: 0.5px;
                }
            </style>
        """, unsafe_allow_html=True)

        def main():
            # App Title
            st.markdown('<div class="header-title">🩺 Comprehensive Health Profile</div>', unsafe_allow_html=True)
            st.markdown("**Unlock personalized health insights with our advanced predictor! 🌟**", unsafe_allow_html=True)

            # Personal Demographic Profile
            with st.container():
                st.markdown('<div class="group-header">👤 Demographic Overview</div>', unsafe_allow_html=True)
                col1, col2 = st.columns(2)
                with col1:
                    gender = st.selectbox('Gender', ['Female', 'Male'], help="Select your gender")
                    #age = st.number_input('Age', min_value=10, max_value=100, value=30, help="What is your current age in years?")
                    age = st.slider('Age', 0,100,1,help="What is your current age in years?")
                
                with col2:
                    occupation = st.selectbox('Occupation', ['Software Engineer', 'Doctor', 'Sales Representative', 'Teacher',
                    'Nurse', 'Engineer', 'Accountant', 'Scientist', 'Lawyer','Salesperson', 'Manager'], help="Select the occupation type that best matches your professional field")
                    bmi_category = st.selectbox('BMI Category', ['Underweight', 'Normal weight', 'Overweight', 'Obese'], help="Your current BMI classification")
                    #cholesterol = st.selectbox('Cholesterol Levels', ['Normal', 'Above Normal', 'High'], help="Your Current Cholesterol levels")
            # Physical Fitness Metrics
            with st.container():
                st.markdown('<div class="group-header">💪 Physical Fitness Assessment</div>', unsafe_allow_html=True)
                col1, col2 = st.columns(2)
                with col1:
                    #bmi_category = st.selectbox('BMI Category', ['Normal weight', 'Overweight', 'Obese'], help="Your current BMI classification. Please use our BMI Calculator to know what category you fall under.")
                    cholesterol = st.selectbox('Cholesterol Levels', ['Normal', 'Above Normal', 'High'], help="Your Current Cholesterol levels")
                    #bmi_category = st.selectbox('BMI Category', ['Normal weight', 'Overweight', 'Obese'], help="Your current BMI classification. Please use our BMI Calculator to know what category you fall under.")
                    physical_activity = st.number_input('Physical Activity (%)', min_value=0, max_value=100, value=50, help="Activity level percentage")
                    heart_rate = st.number_input('Heart Rate (bpm)', min_value=40, max_value=180, value=72, help="Resting heart rate")
                
                with col2:
                    daily_steps = st.number_input('Daily Steps', min_value=0, max_value=30000, value=5000, help="Average number of steps per day")
                    bp_upper = st.number_input('Systolic BP', min_value=80, max_value=200, value=120, help="Systolic blood pressure")
                    bp_lower = st.number_input('Diastolic BP', min_value=60, max_value=130, value=80, help="Diastolic blood pressure")

            # Mental Wellness Indicators
            with st.container():
                st.markdown('<div class="group-header">🧠 Mental Wellness Profile</div>', unsafe_allow_html=True)
                col1, col2 = st.columns(2)
                with col1:
                    sleep_duration = st.number_input('Average Sleep Duration (hrs)', min_value=0.0, max_value=12.0, value=7.0, help="Average hours of sleep per day")
                    sleep_quality = st.slider('Sleep Quality', min_value=1, max_value=10, value=5, help="Rate your sleep quality on a scale of 1 to 10")
                
                with col2:
                    stress_level = st.slider('Stress Level', min_value=1, max_value=10, value=5, help="Rate your overall stress on a scale of 1 to 10")

            # Lifestyle Choices
            with st.container():
                st.markdown('<div class="group-header">🌿 Lifestyle Evaluation</div>', unsafe_allow_html=True)
                col1, col2 = st.columns(2)
                with col1:
                    smoking = st.selectbox('Smoking', ['No', 'Yes'], help="Smoking status")
                
                with col2:
                    alcohol = st.selectbox('Alcohol Consumption', ['No', 'Yes'], help="Alcohol consumption")

            # Prediction Button
            if st.button("🔮 Analyze Health Profile"):
                # Placeholder for model predictions
                st.markdown('<div class="prediction-section">Analyzing your comprehensive health data...</div>', unsafe_allow_html=True)
                
                # TODO: Replace with actual model predictions
                sleep_pred = "Moderate Sleep Risk"
                heart_pred = "Low Cardiovascular Risk"
                
                st.markdown(f'<div class="prediction-section">💤 Sleep Health: <b>{sleep_pred}</b></div>', unsafe_allow_html=True)
                st.markdown(f'<div class="prediction-section">❤️ Cardiovascular Health: <b>{heart_pred}</b></div>', unsafe_allow_html=True)

        if __name__ == "__main__":
            main()



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

    with tab4:
        import streamlit as st
        import re

        class NutritionChatbot:
            def __init__(self):
                self.nutrition_rules = {
                    r'\b(weight|lose weight|weight loss)\b': self.weight_loss_advice,
                    r'\b(muscle|build muscle|gain muscle)\b': self.muscle_building_advice,
                    r'\b(diet|healthy eating|nutrition)\b': self.general_nutrition_advice,
                    r'\b(protein|proteins)\b': self.protein_advice,
                    r'\b(vegetables|fruits|greens)\b': self.produce_advice,
                    r'\b(breakfast|morning meal)\b': self.breakfast_advice,
                    r'\b(hydration|water|drink)\b': self.hydration_advice
                }
                
                self.default_response = (
                    "I'm your nutrition assistant! Ask me about weight loss, muscle building, "
                    "healthy eating, proteins, fruits and vegetables, breakfast, or hydration."
                )
            
            def weight_loss_advice(self):
                return (
                    "For weight loss, focus on:\n"
                    "• Calorie deficit (burn more than you consume)\n"
                    "• High protein intake to preserve muscle\n"
                    "• Balanced diet with whole foods\n"
                    "• Regular exercise, mixing cardio and strength training"
                )
            
            def muscle_building_advice(self):
                return (
                    "To build muscle effectively:\n"
                    "• Consume 1.6-2.2g of protein per kg of body weight\n"
                    "• Eat in a slight calorie surplus\n"
                    "• Focus on progressive resistance training\n"
                    "• Include complex carbs and lean proteins"
                )
            
            def general_nutrition_advice(self):
                return (
                    "Healthy eating tips:\n"
                    "• Eat a variety of whole foods\n"
                    "• Balance macronutrients (proteins, carbs, fats)\n"
                    "• Include fruits, vegetables, whole grains\n"
                    "• Minimize processed foods and added sugars"
                )
            
            def protein_advice(self):
                return (
                    "Protein is crucial for:\n"
                    "• Muscle repair and growth\n"
                    "• Metabolic health\n"
                    "• Feeling full and satisfied\n"
                    "Best sources: chicken, fish, eggs, legumes, tofu"
                )
            
            def produce_advice(self):
                return (
                    "Fruits and vegetables are essential:\n"
                    "• Rich in vitamins and minerals\n"
                    "• High in fiber\n"
                    "• Low in calories\n"
                    "Aim for variety and color in your selections"
                )
            
            def breakfast_advice(self):
                return (
                    "Best breakfast strategies:\n"
                    "• Include protein to stay full\n"
                    "• Add complex carbohydrates for energy\n"
                    "• Consider eggs, oatmeal, Greek yogurt\n"
                    "• Don't skip breakfast"
                )
            
            def hydration_advice(self):
                return (
                    "Hydration tips:\n"
                    "• Drink 8-10 glasses of water daily\n"
                    "• More if you exercise or in hot weather\n"
                    "• Water aids metabolism and energy\n"
                    "• Herbal teas count towards hydration"
                )
            
            def get_response(self, user_input):
                user_input = user_input.lower()
                
                for pattern, response_func in self.nutrition_rules.items():
                    if re.search(pattern, user_input):
                        return response_func()
                
                return self.default_response

        def main():
            st.title("🥗 Nutrition Insights Chatbot")
            
            # Initialize chatbot
            if 'chatbot' not in st.session_state:
                st.session_state.chatbot = NutritionChatbot()
            
            # Chat history
            if 'messages' not in st.session_state:
                st.session_state.messages = []
            
            # Display chat messages
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            
            # User input
            if prompt := st.chat_input("Ask me about nutrition..."):
                # Add user message to chat history
                st.session_state.messages.append({"role": "user", "content": prompt})
                
                # Display user message
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                # Get chatbot response
                response = st.session_state.chatbot.get_response(prompt)
                
                # Add bot response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
                
                # Display bot response
                with st.chat_message("assistant"):
                    st.markdown(response)

        if __name__ == "__main__":
            main()