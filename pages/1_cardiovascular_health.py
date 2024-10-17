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
    data = pd.read_csv('imputed_cardio_sleep.csv')
    
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
st.sidebar.title('Navigation')
options = st.sidebar.selectbox('Select a section:', 
    ['Univariate Analysis', 'Bivariate Analysis', 'Multivariate Analysis', 
     'Correlation Analysis', 'Dimensionality Assessment', 
     'Pattern and Trend Identification', 'Hypothesis Generation'])

# Sidebar filters for dataset slicing
min_age, max_age = st.sidebar.slider('Select Age Range', int(data['Age'].min()), int(data['Age'].max()), (25, 50))
filtered_data = data[(data['Age'] >= min_age) & (data['Age'] <= max_age)]

st.sidebar.subheader('Color Palette for Plots')
palette = st.sidebar.selectbox('Select color palette:', sns.color_palette().as_hex(), index=2)

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

# Hypothesis Generation
elif options == 'Hypothesis Generation':

    st.subheader('Scatter Plot of Sleep Duration vs Heart Risk')
    fig, ax = plt.subplots()
    sns.regplot(x='Sleep Duration', y='heart_risk', data=filtered_data, ax=ax)
    ax.set_title('Sleep Duration vs Heart Risk with Regression Line')
    st.pyplot(fig)

    # Correlation Heatmap Example
    st.subheader('Correlation Heatmap of Sleep Health Metrics and Heart Risk')
    sleep_health_cols = ['Sleep Duration', 'Quality of Sleep', 'heart_risk']  # Add relevant columns
    correlation_matrix = filtered_data[sleep_health_cols].corr()
    
    fig, ax = plt.subplots()
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
    ax.set_title('Correlation Heatmap of Sleep Health and Heart Risk')
    st.pyplot(fig)

# Regression Analysis
