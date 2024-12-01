import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
from scipy import stats

# Create tabs
#tabs = st.tabs(["Tab 1: Sleep and Cardiovascular Health", "Tab 2: Nutrition"])

def tab1_options():
    sidebar_option = st.radio(
        "Select an option",
        ("About Heart Health", "About Nutrition Data")
    )
    return sidebar_option

def tab2_options():
    st.title('Navigation')
    options = st.selectbox('Select a section:', 
        ['Univariate Analysis', 'Bivariate Analysis', 'Multivariate Analysis', 
        'Correlation Analysis', 'Dimensionality Assessment', 
        'Pattern and Trend Identification', 'Hypothesis Generation'])
    

    # Sidebar filters for dataset slicing
    min_age, max_age = st.slider('Select Age Range', int(data['Age'].min()), int(data['Age'].max()), (25, 50))
    filtered_data = data[(data['Age'] >= min_age) & (data['Age'] <= max_age)]

    st.subheader('Color Palette for Plots')
    palette = st.selectbox('Select color palette:', sns.color_palette().as_hex(), index=2)
    return filtered_data, palette, min_age, max_age, options

def tab3_options():
    st.header("Navigation")
    analysis_type = st.selectbox("Select Analysis Type", 
                                        ["Univariate Analysis", "Bivariate Analysis", "Multivariate Analysis", 
                                        "Correlation Analysis", "Dimensionality Assessment", 
                                        "Pattern and Trend Identification"])
    return analysis_type

tab1, tab2, tab3 = st.tabs(["About Datasets", "Sleep and Cardiovascular Health", "Nutrition"])

with tab1:
    # Sidebar selection
    # sidebar_option = st.sidebar.radio(
    #     "Select an option",
    #     ("About Heart Health", "About Nutrition Data")
    # )

    
    sidebar_option = tab1_options()

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
    
    if sidebar_option == "About Heart Health":
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
        sleep = pd.read_csv('https://raw.githubusercontent.com/LaxmiVatsalyaDaita/CMSE830/main/Sleep_health_and_lifestyle_dataset.csv')
        heart = pd.read_csv('https://raw.githubusercontent.com/LaxmiVatsalyaDaita/CMSE830/main/cardio_train.csv', delimiter=";")
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


with tab2:
    import streamlit as st
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder
    from scipy import stats
    import plotly.express as px
    import statsmodels.api as sm

    # Load the data with mappings for categorical variables

    @st.cache_resource
    def load_data():
        url = "https://raw.githubusercontent.com/LaxmiVatsalyaDaita/CMSE830/main/pages/imputed_cardio_sleep.csv"
        #data = pd.read_csv('imputed_cardio_sleep.csv')
        data = pd.read_csv(url, delimiter=",")
        
        # Round off the values to the nearest integer for categorical variables
        data['gluc'] = data['gluc'].round().astype(int)
        data['smoke'] = data['smoke'].round().astype(int)
        data['alco'] = data['alco'].round().astype(int)
        data['active'] = data['active'].round().astype(int)
        data['heart_risk'] = data['heart_risk'].round().astype(int)
        
        # Map categorical variable values to their respective labels
        data['Gender'] = data['Gender'].map({1: 'Male', 0: 'Female'})
        data['Occupation'] = data['Occupation'].map({0: 'Accountant', 1: 'Doctor', 2: 'Engineer', 3: 'Lawyer', 
                                                    4: 'Manager', 5: 'Nurse', 6: 'Sales Representative', 
                                                    7: 'Salesperson', 8: 'Scientist', 9: 'Software Engineer', 
                                                    10: 'Teacher'})
        data['BMI Category'] = data['BMI Category'].map({0: 'Normal', 1: 'Obese', 2: 'Overweight'})
        data['gluc'] = data['gluc'].map({1: 'normal', 2: 'above normal', 3: 'high'})
        data['smoke'] = data['smoke'].map({1: 'yes', 0: 'no'})
        data['alco'] = data['alco'].map({1: 'yes', 0: 'no'})
        data['active'] = data['active'].map({1: 'yes', 0: 'no'})
        data['heart_risk'] = data['heart_risk'].map({1: 'yes', 0: 'no'})
        data['Sleep Disorder'] = data['Sleep Disorder'].map({1: 'No Disorder', 0: 'Insomnia', 2:'Sleep Apnea'})

        data = data.rename(columns={
            'gluc': 'Glucose',
            'smoke': 'Smoking',
            'alco': 'Alcohol',
            'active': 'Daily Activity',
            'bp_upper': 'Systolic Blood Pressure',
            'bp_lower': 'Diastolic Blood Pressure'
            # Add more columns as needed
        })
        return data


    data = load_data()

    # Streamlit app layout and configuration
    st.title('Sleep Data Analysis Dashboard')

    # Sidebar for navigation and filters
    # st.sidebar.title('Navigation')
    # options = st.sidebar.selectbox('Select a section:', 
    #     ['Univariate Analysis', 'Bivariate Analysis', 'Multivariate Analysis', 
    #     'Correlation Analysis', 'Dimensionality Assessment', 
    #     'Pattern and Trend Identification', 'Hypothesis Generation'])

    # # Sidebar filters for dataset slicing
    # min_age, max_age = st.sidebar.slider('Select Age Range', int(data['Age'].min()), int(data['Age'].max()), (25, 50))
    # filtered_data = data[(data['Age'] >= min_age) & (data['Age'] <= max_age)]

    # st.sidebar.subheader('Color Palette for Plots')
    # palette = st.sidebar.selectbox('Select color palette:', sns.color_palette().as_hex(), index=2)


    
    filtered_data, palette, min_age, max_age, options = tab2_options()

    # Univariate Analysis
    if options == 'Univariate Analysis':
        st.header('Univariate Analysis')
        
        # Numeric variables
        st.subheader('Histograms and Box Plots for Numeric Variables')
        numeric_cols = filtered_data.select_dtypes(include=[np.number]).columns
        selected_numeric = st.selectbox('Select a numeric variable:', numeric_cols)
        
        col1, col2 = st.columns(2)
        with col1:
            fig1, ax1 = plt.subplots()
            sns.histplot(filtered_data[selected_numeric], kde=True, ax=ax1, color=palette)
            ax1.set_title(f'Histogram of {selected_numeric}')
            st.pyplot(fig1)
        with col2:
            fig2, ax2 = plt.subplots()
            sns.boxplot(y=filtered_data[selected_numeric], ax=ax2, color=palette)
            ax2.set_title(f'Box Plot of {selected_numeric}')
            st.pyplot(fig2)
        
        # Categorical variables
        st.subheader('Bar Charts for Categorical Variables')
        categorical_cols = filtered_data.select_dtypes(include=['object']).columns
        selected_categorical = st.selectbox('Select a categorical variable:', categorical_cols)
        
        fig, ax = plt.subplots()
        filtered_data[selected_categorical].value_counts().plot(kind='bar', ax=ax, color=palette)
        ax.set_title(f'Bar Chart of {selected_categorical}')
        st.pyplot(fig)
        
        # Summary statistics
        st.subheader('Summary Statistics')
        st.write(filtered_data.describe())

    # Bivariate Analysis
    elif options == 'Bivariate Analysis':
        st.header('Bivariate Analysis')
        
        # Scatter plots
        st.subheader('Scatter Plots')
        x_var = st.selectbox('Select X variable:', filtered_data.columns)
        y_var = st.selectbox('Select Y variable:', filtered_data.columns)
        
        fig = px.scatter(filtered_data, x=x_var, y=y_var, color='Gender', title=f'Scatter Plot: {x_var} vs {y_var}')
        st.plotly_chart(fig)
        
        # Cross-tabulations
        st.subheader('Cross-tabulations')
        categorical_vars = ['Gender', 'Occupation', 'Sleep Duration', 'Quality of Sleep',
                        'Physical Activity Level', 'Stress Level', 'BMI Category', 
                        'Sleep Disorder', 'Smoking', 'Alcohol', 'Daily Activity', 'heart_risk']

        # Create a new selectbox for categorical variables
        cat_var1 = st.selectbox('Select first categorical variable:', categorical_vars)
        cat_var2 = st.selectbox('Select second categorical variable:', categorical_vars)
        
        st.write(pd.crosstab(filtered_data[cat_var1], filtered_data[cat_var2]))

        
    # Multivariate Analysis
    elif options == 'Multivariate Analysis':
        st.header('Multivariate Analysis')

        # PCA with interactive components selection
        st.subheader('Principal Component Analysis (PCA)')
        numeric_data = filtered_data.select_dtypes(include=[np.number])
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_data)
        
        n_components = st.slider('Select number of PCA components', 1, len(numeric_data.columns), 2)
        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(scaled_data)
        
        fig, ax = plt.subplots()
        ax.plot(np.cumsum(pca.explained_variance_ratio_), color=palette)
        ax.set_xlabel('Number of Components')
        ax.set_ylabel('Cumulative Explained Variance')
        ax.set_title('PCA: Cumulative Explained Variance')
        st.pyplot(fig)
        
        # Pair plot
        st.subheader('Pair Plot')
        numeric_vars = filtered_data.select_dtypes(include=[np.number]).columns.tolist()
        selected_vars = st.multiselect('Select numerical variables for pairplot:', numeric_vars, default=numeric_vars)
        if len(selected_vars) > 1:
            fig = sns.pairplot(filtered_data[selected_vars], height=2, palette=palette)
            st.pyplot(fig)
        else:
            st.write('Please select at least two variables for the pair plot.')

    # Correlation Analysis
    elif options == 'Correlation Analysis':
        st.header('Correlation Analysis')

        encoded_data = filtered_data.copy()

        categorical_cols = ['Gender', 'Occupation', 'BMI Category', 'Glucose', 'Smoking', 'Alcohol', 'Daily Activity', 'heart_risk']

        # Apply Label Encoding to categorical columns
        label_encoder = LabelEncoder()
        for col in categorical_cols:
            encoded_data[col] = label_encoder.fit_transform(encoded_data[col])

        # Ensure only numeric data is used
        numeric_data = encoded_data.select_dtypes(include=[np.number])

        # Check for missing values and handle them (e.g., fill them or drop)
        if numeric_data.isnull().values.any():
            st.warning("Warning: There are missing values in the dataset. Filling missing values with column means.")
            numeric_data = numeric_data.fillna(numeric_data.mean())
        
        # Correlation matrix
        corr_matrix = numeric_data.corr()

        # Color palette selection for heatmap
        heatmap_palette = st.sidebar.selectbox('Select a color palette for the heatmap:', 
                                            ['coolwarm', 'viridis', 'plasma', 'inferno', 'magma'])

        # Correlation heatmap plot
        st.subheader('Correlation Heatmap')
        fig, ax = plt.subplots(figsize=(20, 10))
        sns.heatmap(corr_matrix, annot=True, cmap=heatmap_palette, ax=ax, vmin=-1, vmax=1, center=0)
        ax.set_title('Correlation Heatmap')
        st.pyplot(fig)

        # Strong correlations
        st.subheader('Strongly Correlated Variables')
        threshold = st.slider('Correlation threshold:', 0.0, 1.0, 0.8, 0.05)
        
        # Identify pairs of strongly correlated variables
        strong_corr = np.where(np.abs(corr_matrix) > threshold)
        strong_corr_pairs = [(corr_matrix.index[x], corr_matrix.columns[y], corr_matrix.iloc[x, y]) 
                            for x, y in zip(*strong_corr) if x != y and x < y]

        # Display strongly correlated pairs
        if strong_corr_pairs:
            st.write(pd.DataFrame(strong_corr_pairs, columns=['Variable 1', 'Variable 2', 'Correlation']))
        else:
            st.write('No strong correlations found at the current threshold.')

    # Dimensionality Assessment
    elif options == 'Dimensionality Assessment':
        st.header('Dimensionality Assessment')

        # Display number of features and observations
        st.write(f'Number of features: {data.shape[1]}')
        st.write(f'Number of observations: {data.shape[0]}')

        # Feature importance using Random Forest
        st.subheader('Feature Importance (using Random Forest)')
        
        if 'Sleep Disorder' not in data.columns:
            st.error("Error: 'Sleep Disorder' column not found in the dataset.")
        else:
            X = data.drop('Sleep Disorder', axis=1)
            y = data['Sleep Disorder']

            categorical_cols = X.select_dtypes(include=['object']).columns
            if len(categorical_cols) > 0:
                from sklearn.preprocessing import OneHotEncoder
                encoder = OneHotEncoder(sparse_output=False, drop='first')
                encoded_features = encoder.fit_transform(X[categorical_cols])
                encoded_feature_names = encoder.get_feature_names_out(categorical_cols)
                X = pd.concat([X.drop(categorical_cols, axis=1), 
                            pd.DataFrame(encoded_features, columns=encoded_feature_names)], axis=1)
            
            rf = RandomForestClassifier()
            rf.fit(X, y)
            importances = rf.feature_importances_
            feature_importances = pd.Series(importances, index=X.columns).sort_values(ascending=False)
            
            st.bar_chart(feature_importances[:10])

    # Pattern and Trend Identification
    elif options == 'Pattern and Trend Identification':
        st.header('Pattern and Trend Identification')

        # Trend lines for time-based or age-based analysis
        time_var = st.selectbox('Select variable for x-axis (time/age-based):', filtered_data.columns)
        y_var = st.selectbox('Select variable for y-axis:', filtered_data.columns)
        
        fig = px.scatter(filtered_data, x=time_var, y=y_var, color='Gender', title=f'Trend Plot: {time_var} over {y_var}')
        st.plotly_chart(fig)

    elif options == 'Hypothesis Generation':
        st.subheader('Distribution of Age by Heart Risk Categories')
        
        # Plot distribution of Age for different heart_risk categories
        fig, ax = plt.subplots()
        sns.histplot(data=filtered_data, x='Age', hue='heart_risk', multiple='stack', kde=True, ax=ax)
        ax.set_title('Older individuals are more prone to cardiovascular issues')
        st.pyplot(fig)

        st.markdown("""
        The plot shows the age distribution of individuals based on their heart risk. It reveals that the risk of heart disease increases with age, with the highest proportion of individuals with heart risk found in the age group of 45-50. Younger individuals are more likely to have no heart risk, but as age increases, the proportion of individuals without heart risk decreases. The density curves overlaid on the histograms provide a smoother representation of the age distributions, helping to visualize the underlying probability distributions and identify patterns or trends. Overall, the plot suggests a strong association between age and heart risk, highlighting the importance of targeting preventive measures and interventions for heart disease in high-risk populations.
        """)

        # Explore relationship between Physical Activity Level, BMI Category, and Quality of Sleep
        st.subheader('Physical Activity vs BMI Category and Quality of Sleep')
        
        #st.subheader("Chi-Square Test of Independence")
        st.write("Testing whether there is a significant relationship between Physical Activity, BMI, and Sleep Quality.")
        
        # Perform chi-square test of independence
        contingency_table = pd.crosstab(filtered_data['Physical Activity Level'], 
                                        [filtered_data['BMI Category'], filtered_data['Quality of Sleep']])
        chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
        
        st.write(f"Chi-Square Statistic: {chi2:.4f}")
        st.write(f"P-value: {p:.4f}")
        st.write(f"Degrees of Freedom: {dof}")
        
        # Interpret the p-value
        if p < 0.05:
            st.write("Result: The p-value is less than 0.05, so we reject the null hypothesis. This indicates that there is a significant relationship between Physical Activity, BMI, and Sleep Quality.")
        else:
            st.write("Result: The p-value is greater than 0.05, so we fail to reject the null hypothesis. This suggests that there is no significant relationship between Physical Activity, BMI, and Sleep Quality.")
        
        st.subheader('Sleep Disorder vs Heart Risk')
        st.write("2. Testing whether there is a significant relationship between Sleep Disorder and Heart Risk.")
        
        # Perform chi-square test of independence
        sleep_heart_risk_ct = pd.crosstab(filtered_data['Sleep Disorder'], filtered_data['heart_risk'])
        chi2, p, dof, expected = stats.chi2_contingency(sleep_heart_risk_ct)
        
        st.write(f"Chi-Square Statistic: {chi2:.4f}")
        st.write(f"P-value: {p:.4f}")
        st.write(f"Degrees of Freedom: {dof}")
        
        # Interpret the p-value
        if p < 0.05:
            st.write("Result: The p-value is less than 0.05, so we reject the null hypothesis. This indicates that there is a significant relationship between Sleep Disorder and Heart Risk.")
        else:
            st.write("Result: The p-value is greater than 0.05, so we fail to reject the null hypothesis. This suggests that there is no significant relationship between Sleep Disorder and Heart Risk.")



with tab3:
    import streamlit as st
    import pandas as pd
    import numpy as np
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from scipy import stats
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

        # Set page config
    #st.set_page_config(page_title="Nutrition EDA Dashboard", page_icon="🍎", layout="wide")

    # Load the data
    @st.cache_data
    def load_data():
        # url = "https://raw.githubusercontent.com/LaxmiVatsalyaDaita/CMSE830/pages/nutrition_cleaned.csv"
        # #data = pd.read_csv('nutrition_cleaned.csv')
        # data = pd.read_csv(url, delimiter=",")
        # return data

        try:
            url = "https://raw.githubusercontent.com/LaxmiVatsalyaDaita/CMSE830/main/nutrition_cleaned.csv"
            return pd.read_csv(url)
        except (urllib.error.HTTPError, pd.errors.EmptyDataError) as e:
            st.warning(f"Could not load data from primary URL. Trying alternate source...")
            try:
                # Try alternate URL (master branch instead of main)
                url_alt = "https://raw.githubusercontent.com/LaxmiVatsalyaDaita/CMSE830/master/nutrition_cleaned.csv"
                return pd.read_csv(url_alt)
            except Exception as e:
                try:
                    # Try local file as last resort
                    return pd.read_csv('nutrition_cleaned.csv')
                except Exception as e:
                    st.error("""
                    Failed to load data from all sources. Please ensure:
                    1. The file exists in the repository
                    2. The file path is correct
                    3. The repository is public
                    4. You have the file locally if all else fails
                    
                    Error details: """ + str(e))
                    return None

    df = load_data()

    # Check if 'Category' and 'Food' columns exist
    if 'Category' not in df.columns or 'Food' not in df.columns:
        st.error("The dataset does not contain required columns: 'Category' or 'Food'. Please check the dataset.")
    else:
        # Title
        st.title("🍎 Comprehensive Nutrition EDA Dashboard")

        # Sidebar
        # st.sidebar.header("Navigation")
        # analysis_type = st.sidebar.selectbox("Select Analysis Type", 
        #                                     ["Univariate Analysis", "Bivariate Analysis", "Multivariate Analysis", 
        #                                     "Correlation Analysis", "Dimensionality Assessment", 
        #                                     "Pattern and Trend Identification"])
        
        
        analysis_type = tab3_options()


        # Main content
        if analysis_type == "Univariate Analysis":
            st.header("1. Univariate Analysis")
            
            # Numeric variables
            st.subheader("Numeric Variables")
            # numeric_cols = list(df.select_dtypes(include=[np.number]).columns)  # Convert to list
            numeric_cols = ['Calories', 'Protein', 'Fat', 'Sat.Fat', 'Fiber', 'Carbs']
            selected_numeric = st.multiselect("Select numeric variables", numeric_cols)

            if selected_numeric:
            # Adjust the subplot titles: histogram and box plot for each variable side by side
                subplot_titles = []
                for col in selected_numeric:
                    subplot_titles.append(f"{col} Histogram")
                    subplot_titles.append(f"{col} Box Plot")
                
                fig = make_subplots(rows=len(selected_numeric), cols=2, subplot_titles=subplot_titles)
                
                for i, col in enumerate(selected_numeric, start=1):
                    # Add histogram to the first column
                    fig.add_trace(go.Histogram(x=df[col], name=f"{col} Histogram"), row=i, col=1)
                    # Add box plot to the second column
                    fig.add_trace(go.Box(y=df[col], name=f"{col} Box Plot"), row=i, col=2)
                
                # Adjust layout height to accommodate multiple rows
                fig.update_layout(height=300*len(selected_numeric), showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            
            # Categorical variables
            # Categorical variables
            st.subheader("Categorical Variables")

            # Check if 'Category' exists in the DataFrame
            if 'Category' in df.columns:
                # Create a DataFrame with category counts
                cat_count_df = df['Category'].value_counts().reset_index()
                cat_count_df.columns = ['Category', 'Count']  # Rename columns for clarity

                # Plot the bar chart
                fig = px.bar(cat_count_df, x='Category', y='Count', color='Category',
                            labels={'Category': 'Category', 'Count': 'Count'},
                            title="Distribution of Food Categories")
                
                fig.update_layout(showlegend=False)  # Hide legend for cleaner look
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No 'Category' variable found in the dataset.")


            
            # Summary statistics
            st.subheader("Summary Statistics")
            st.dataframe(df.describe())

        elif analysis_type == "Bivariate Analysis":
            st.header("2. Bivariate Analysis")
            
            # Scatter plot
            st.subheader("Scatter Plot")
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if not numeric_cols.empty:
                x_var = st.selectbox("Select X variable", numeric_cols)
                y_var = st.selectbox("Select Y variable", numeric_cols)
                fig = px.scatter(df, x=x_var, y=y_var, color='Category', hover_data=['Food'])
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No numeric variables available for bivariate analysis.")
            
            # # Cross-tabulation
            # st.subheader("Cross-tabulation")
            # cat_cols = df.select_dtypes(include=['object']).columns
            # if not cat_cols.empty:
            #     cat_var1 = st.selectbox("Select first categorical variable", cat_cols)
            #     cat_var2 = st.selectbox("Select second categorical variable", cat_cols)
            #     st.dataframe(pd.crosstab(df[cat_var1], df[cat_var2]))
            # else:
            #     st.warning("No categorical variables available for cross-tabulation.")
            
            # Correlation
            st.subheader("Correlation")
            corr = df.select_dtypes(include=[np.number]).corr()
            fig = px.imshow(corr, text_auto=True, aspect="auto")
            fig.update_layout(title="Correlation Heatmap")
            st.plotly_chart(fig, use_container_width=True)

        elif analysis_type == "Multivariate Analysis":
            st.header("3. Multivariate Analysis")
            
            # PCA
            st.subheader("Principal Component Analysis (PCA)")
            numeric_df = df[['Calories', 'Protein', 'Fat', 'Sat.Fat', 'Fiber', 'Carbs']]  # Ensure we select the right columns
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(numeric_df)
            pca = PCA()
            pca_data = pca.fit_transform(scaled_data)
            
            fig = px.scatter(x=pca_data[:, 0], y=pca_data[:, 1], color=df['Category'],
                            labels={'x': 'First Principal Component', 'y': 'Second Principal Component'})
            fig.update_layout(title="PCA Visualization")
            st.plotly_chart(fig, use_container_width=True)
            
            # Pair plot
            st.subheader("Pair Plot")
            numeric_columns_list = numeric_df.columns.tolist()  # Convert to list
            selected_vars = st.multiselect("Select variables for pair plot", numeric_columns_list)
            if selected_vars:
                fig = px.scatter_matrix(df, dimensions=selected_vars, color='Category')
                st.plotly_chart(fig, use_container_width=True)

            else:
                st.warning("Please select the numerical variables.")

        elif analysis_type == "Correlation Analysis":
            st.header("4. Correlation Analysis")
            
            corr = df.select_dtypes(include=[np.number]).corr()
            
            # Correlation heatmap
            st.subheader("Correlation Heatmap")
            fig = px.imshow(corr, text_auto=True, aspect="auto")
            fig.update_layout(title="Correlation Heatmap")
            st.plotly_chart(fig, use_container_width=True)
            
            # Strongly correlated variables
            st.subheader("Strongly Correlated Variables")
            threshold = st.slider("Correlation Threshold", 0.0, 1.0, 0.8, 0.05)
            strong_corr = (corr.abs() > threshold) & (corr != 1.0)
            if not strong_corr.empty:
                st.dataframe(strong_corr[strong_corr].stack().reset_index())
            else:
                st.warning("No strongly correlated variables found above the threshold.")

        elif analysis_type == "Dimensionality Assessment":
            st.header("5. Dimensionality Assessment")
            
            st.subheader("Feature vs Observations")
            st.write(f"Number of features: {df.shape[1]}")
            st.write(f"Number of observations: {df.shape[0]}")
            
            st.subheader("Variance Explained by Principal Components")
            numeric_df = df.select_dtypes(include=[np.number])
            if not numeric_df.empty:
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(numeric_df)
                pca = PCA()
                pca.fit(scaled_data)
                
                fig = px.line(x=range(1, len(pca.explained_variance_ratio_)+1), y=np.cumsum(pca.explained_variance_ratio_))
                fig.update_layout(title="Cumulative Variance Explained by Principal Components",
                                xaxis_title="Number of Components", yaxis_title="Cumulative Variance Explained")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No numeric variables available for dimensionality assessment.")

        elif analysis_type == "Pattern and Trend Identification":
            st.header("6. Pattern and Trend Identification")
            
            st.subheader("Distribution of Foods Across Categories")
            fig = px.histogram(df, x='Category')
            fig.update_layout(title="Distribution of Foods Across Categories")
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Nutrient Patterns Across Categories")
            nutrient = st.selectbox("Select a nutrient", ['Calories', 'Protein', 'Fat', 'Carbs', 'Fiber'])
            fig = px.box(df, y='Category', x=nutrient)
            fig.update_layout(title=f"{nutrient} Distribution Across Categories")
            st.plotly_chart(fig, use_container_width=True)

        # elif analysis_type == "Hypothesis Generation":
        #     st.header("7. Hypothesis Generation")
            
        #     st.write("Based on the exploratory data analysis, we can generate the following hypotheses:")
            
        #     st.write("1. There might be a strong positive correlation between calorie content and fat content in foods.")
        #     st.write("2. Certain food categories may have significantly higher protein content than others.")
        #     st.write("3. The fiber content might be inversely related to the calorie density of foods.")
        #     st.write("4. There could be distinct clusters of foods based on their nutritional profiles.")
            
        #     st.write("To investigate these hypotheses, we would need to perform more detailed statistical analyses, "
        #              "such as hypothesis tests, regression analyses, or clustering algorithms.")

        # Footer
        st.markdown("---")
        st.markdown("By: Laxmi Vatsalya Daita")
