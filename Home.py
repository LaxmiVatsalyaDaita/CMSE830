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
sidebar_option = st.sidebar.selectbox(
    "Select an option",
    ("Overview", "About Heart Health", "About Nutrition Data")
)

# Load the data
@st.cache_data
def load_data():
    #data = pd.read_csv('new_merged.csv')

    url = "https://raw.githubusercontent.com/LaxmiVatsalyaDaita/CMSE830/blob/main/new_merged.csv"
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

Sleep and nutrition are two fundamental pillars of health that significantly impact our physical, mental, and emotional well-being. Understanding the relationship between these two components can empower individuals to make informed lifestyle choices that enhance their quality of life.

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

elif sidebar_option == "About Heart Health":
    st.title("🫀 About Heart Health Dashboard 🩺")

    st.markdown("""
    ## 1. Data Collection and Importation
    The data was sourced from multiple Kaggle datasets on cardiovascular and sleep health. These datasets were merged to provide a comprehensive view of heart health and sleep, and the resulting dataset was imported into the Python environment for analysis.
                
    The datasets used can be accessed here: 
    
    1. [Cardiovascular Disease dataset](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset/data) by Svetlana Ulianova
    
    2. [Sleep Health and Lifestyle Dataset](https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset) by Laksika Tharmalingam
    
    """)

    st.markdown("""
    ## 2. Data Cleaning and Preprocessing
    Initial steps included removing unnecessary columns like identifiers in both the datasets. In the Sleep health and Lifestyle Dataset, the next steps involved in preprocessing was to correct the datatypes. I split the Blood Pressure reading into Systolic and Diastolic Pressure readings (which were later used for the merge with the cardiovascular health dataset) and typecasted them to integers. There were redundant labels present (such as Normal and Normal Weight in BMI Category) which were combined for a better quality dataset. In the Cardiovascular Disease dataset, the columns have to be renamed to match with the column names in the other dataset. The BMI was calculated using the height and weight columns.
                
    After the datasets were ready, they were merged based on the Systolic and Diastolic Blood Pressure values along with the Age.The jon resulted in a lot of missing values in the final dataset. 
    The final dataset has 808 rows and 26 columns.
    """)
    
    
    st.markdown("""
    ## 3. Variable Identification and Classification
    Variables were identified as either dependent or independent and classified as numeric or categorical. Key variables include:
    - **heart_risk**: Indicator of cardiovascular risk (dependent variable).
    - **Sleep Disorder**: Indicator of the presence of Sleep Disorder (dependent variable).
    - **Numeric Variables**: Age, sleep duration, physical activity, heart rate, daily steps, systolic and diastolic blood pressures, etc.
    - **Categorical Variables**: Gender, occupation, and sleep disorder, , BMI category  status.
    - **Binary Variables**: smoking conditions, daily activity, alcohol consumption, heart risk
    - **Ordinal Variables**: quality of sleep, stress level.
    
    The columns in Cardiovascular Disease dataset were already encoded. However, the columns in Sleep Health and Lifestyle dataset had to be encoded. I used a **LabelEncoder** keeping in mind that using a One-Hot encoder would result in a very high-dimensional dataset due to the presence of multiple labels in each column.
    """)
    
    st.markdown("""
    ## 4. Basic Descriptive Statistics
    Descriptive statistics such as mean, median, and standard deviation were calculated for each numeric variable.
    """)
    sleep = pd.read_csv('sleep_health_and_lifestyle_dataset.csv')
    heart = pd.read_csv('cardio_train.csv', delimiter=";")
    st.write(sleep.describe())
    st.write(heart.describe())

    ## 5. Missing Data Analysis
    st.markdown("""
    ## 5. Missing Data Analysis
    A correlation matrix explored any patterns in missing values. This analysis guided the choice of imputation method and provided insight into the relationships between variables. After performing a missing value analysis, it was found out that the data was **MAR** as the mssing values were correlated in the columns of the cardiovascular disease dataset. 
    """)
    # Plot missing values heatmap
    st.markdown("### Missing Values Heatmap")
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
    st.pyplot(plt)
    
    # # Show the data info
    # st.write("### Data Information")
    # buffer = st.text_area("Data Info", df.info(buf=None))
    # st.write(buffer)

    # Encoding categorical variables
    le_gender = LabelEncoder()
    le_occupation = LabelEncoder()
    le_bmi_category = LabelEncoder()
    le_sleep_disorder = LabelEncoder()

    df1 = df.copy()
    df1['Gender'] = le_gender.fit_transform(df1['Gender'])
    df1['Occupation'] = le_occupation.fit_transform(df1['Occupation'])
    df1['BMI Category'] = le_bmi_category.fit_transform(df1['BMI Category'])
    df1['Sleep Disorder'] = le_sleep_disorder.fit_transform(df1['Sleep Disorder'])

    st.write("### Encoded Categorical Variables")
    st.write(df1[['Gender', 'Occupation', 'BMI Category', 'Sleep Disorder']].head())

    
    # Plot correlation of missing values
    missing_corr = df1.isnull().corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(missing_corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    st.pyplot(plt)
    

    st.markdown("""
    ## 6. Data Quality Assessment
    The absolute difference between the correlation of features before and after imputation being the least for KNN Imputation compelled me to choose KNNs over other imputation techniques such as MICE.
    """)
    
    # KNN Imputation
    imputer = KNNImputer(n_neighbors=5)
    df_imputed = pd.DataFrame(imputer.fit_transform(df1), columns=df1.columns)

    st.markdown("### Correlation Heatmap after Imputation")
    plt.figure(figsize=(20, 20))
    sns.heatmap(df_imputed.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    st.pyplot(plt)
    
    st.markdown("""
    ## 7. Outlier Detection
    Outliers were identified using both visual (box plots) and statistical (Z-score) methods. 
    """)
    
    numeric_columns = [
        'Age', 'Sleep Duration', 'Quality of Sleep', 'Physical Activity Level', 
        'Stress Level', 'Heart Rate', 'Daily Steps', 'bp_upper', 
        'bp_lower', 'cholesterol', 'gluc'
    ]

    # Box plot for numeric columns
    st.markdown("### Box Plots for Outlier Detection")
    num_columns = len(numeric_columns)
    num_rows = (num_columns // 3) + (num_columns % 3 > 0)

    plt.figure(figsize=(20, 5 * num_rows))
    for i, column in enumerate(numeric_columns):
        plt.subplot(num_rows, 3, i + 1)
        sns.boxplot(data=df_imputed, x=column)
        plt.title(f'Box Plot for {column}')
    st.pyplot(plt)

    # Z-score outliers
    z_scores = np.abs(stats.zscore(df_imputed[numeric_columns]))
    threshold = 3
    outliers_z = (z_scores > threshold)

    st.write("### Outliers Detected based on Z-Score")
    st.write(df_imputed[outliers_z.any(axis=1)])

    st.markdown("""
The outliers were treated, along with the rounding-off of the newly imputed values. The final dataframe was saved as **new_merged.csv**
    """)

elif sidebar_option == "About Nutrition Data":
    st.title("Nutrition Data - Initial Data Analysis")



    # Display the dataset
    st.subheader("Dataset Preview")
    st.markdown("""
    In this section, we take an initial look at the nutrition dataset. The dataset used in this dashboard is the cleaned version of the dataset by Niharika Pandit from Kaggle, and can be accessed [here](https://www.kaggle.com/datasets/niharika41298/nutrition-details-for-most-common-foods). It contains information on various food items, including nutritional values such as Calories, Protein, Fat, Saturated Fat, Fiber, and Carbohydrates. Each food item is categorized to facilitate comparative analysis across different food groups. The dataset was scraped from Wikipedia. The preview below shows the first few rows, 
    giving a sense of the structure of the data, including column names and sample values.
    """)
    nutrition_data = pd.read_csv("nutrients_csvfile.csv")
    st.write(nutrition_data.head())

    # Summary Statistics
    st.subheader("Summary Statistics")
    st.markdown("""
    To understand the data better, we compute basic summary statistics for numerical columns, such as mean, median, 
    minimum, and maximum values. These provide insights into the general range of values and any potential anomalies.
    """)
    st.write(nutrition_data.describe())

    # Data Types
    st.subheader("Data Types")
    st.markdown("""
    Knowing the data types of each column is essential for data preprocessing. This section shows the data type of each 
    column in the dataset (e.g., numerical, categorical), which helps us understand what transformations might be needed 
    for analysis and visualization.
    """)
    st.write(nutrition_data.dtypes)


else:
    st.markdown("""
    """)
