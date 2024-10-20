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
st.set_page_config(page_title="Nutrition EDA Dashboard", page_icon="🍎", layout="wide")

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
    st.sidebar.header("Navigation")
    analysis_type = st.sidebar.selectbox("Select Analysis Type", 
                                         ["Univariate Analysis", "Bivariate Analysis", "Multivariate Analysis", 
                                          "Correlation Analysis", "Dimensionality Assessment", 
                                          "Pattern and Trend Identification"])
    


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
    st.markdown("Data source: nutrients_cleaned.csv")
