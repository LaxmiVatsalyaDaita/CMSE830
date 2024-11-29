import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
from scipy import stats

st.set_page_config(
    page_title="Nourish and Rest",
    page_icon="🥕",
)

# Sidebar selection
sidebar_option = st.sidebar.radio(
    "Select an option",
    ("Overview", "something 1", "something 2")
)

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

# About Cardiovascular Health - IDA with outputs
if sidebar_option == "Overview":
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

