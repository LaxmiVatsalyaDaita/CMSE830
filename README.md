The web application has been deployed [here](https://cmse830-snugfit.streamlit.app/) using *streamlit.io*

#### To run the code on local machine:
1. Clone the Git Repository.
2. Install the required libraries by running `pip install -r requirements.txt` in terminal.
3. Use `streamlit run Home.py` to run the app on localhost.
4. Incase the port is busy and an exception occurs, try running `streamlit run Home.py --server.port 8080`.

#### Description of the files included:
1. **Home.py** is the main app file.
2. The app uses 2 pages which can be found in the "pages" folder.
3. The initial and cleaned datasets used can be found in Datasets repository.
4. The models build are saved in the Models folder.
5. The plots for the performance metrics can be found in Results

---
<p align="center"> 
  <h1> 🌟 SnugFit </h1> 
</p>

Welcome to the **SnugFit**! A Holistic Approach to Better Sleep, Nutrition, and Cardiovascular Health. This Streamlit app is your one-stop platform for personalized health predictions, BMI calculations, and actionable nutrition advice. Whether you're curious about your heart health, want insights into sleep disorders, or need help with healthy eating, we've got you covered.


## 🎯 Features

### 1. Sleep Disorder and Heart Risk Prediction
- **Sleep Disorder Prediction**:  
  Enter a few key inputs to discover if you're at risk of common sleep disorders like insomnia or sleep apnea.
  
- **Heart Risk Prediction**:  
  Gain insights into your cardiovascular health and determine potential risks based on your lifestyle and health metrics.

- **Advanced Ensemble Modeling**:  
  These predictions are powered by advanced machine learning models that provide accurate and reliable results.



### 2. BMI Calculator
Easily calculate your Body Mass Index (BMI) to understand whether you're in a healthy weight range. Just input your height and weight, and the app will provide a quick and clear result.


### 3. NutriBuddy – Your Nutrition Assistant (Beta) 🥗  
NutriBuddy is an AI-powered chatbot that provides customized nutrition advice.  
Ask NutriBuddy about:
- Weight loss tips
- Muscle-building strategies
- Healthy eating
- Protein sources
- Fruits and vegetables
- Hydration tips
- Breakfast ideas

NutriBuddy is here to make nutrition fun and easy to understand! 🤖✨  


## 🌈 Why Use This Dashboard?
- **User-Friendly Interface**:  
  Designed for simplicity and ease of use, with all input elements on a single page to reduce scrolling.  
   
- **Interactive and Insightful**:  
  Combines actionable insights, clear visualizations, and fun animations to keep you engaged.  
   
- **Self-Contained and Informative**:  
  No external knowledge is required—each section provides clear instructions and explanations for first-time users.



## 💻 How to Use
1. Open the app and navigate using the tabs at the top:  
   - **Welcome**: A brief overview of the app.  
   - **Risk Prediction**: Predict sleep disorders and heart risk based on your inputs.  
   - **BMI Calculator**: Quickly calculate your BMI.  
   - **Ask for Nutrition Tips**: Chat with NutriBuddy for personalized nutrition advice.  

2. Input your details in the respective sections and explore the predictions or advice tailored to you.

3. For a bit of fun, enjoy the polished animations and beautiful design while exploring the app!



## 🚀 Tech Stack
- **Streamlit**: For interactive and responsive UI development.  
- **Python**: Core language for machine learning models and backend logic.  
- **Machine Learning**: Regression, Ensemble Modeling and Neural Network for heart and sleep predictions.  
- **Chatbot**: Regex Expressions - still under development.


## 📈 Future Enhancements
- Improve NutriBuddy to have more complex and insightful conversations.
- Include more predictive models for mental health insights.  
- Integrate a calorie tracker and daily meal planner.  
- Add time-series analysis for tracking sleep and cardiovascular health trends over time.



## 🙌 Acknowledgments
This app was developed as a project for **CMSE 830**, with a focus on providing actionable health insights in a professional, user-friendly format. Special thanks to the amazing resources and datasets that made this app possible.


### 🛠️ **Created by Vatsalya Daita**  
Let SnugFit guide you on your journey to a healthier and happier life! 🌱✨ 
