import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import streamlit as st
import joblib

# Load the trained model and dataset
model = joblib.load('best_regression_model.pkl')
df = pd.read_csv('Startups.csv')

# Preprocessor setup for scaling numerical and encoding categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['R&D Spend', 'Administration', 'Marketing Spend']),
        ('cat', OneHotEncoder(), ['State'])
    ]
)

# Pipeline for preprocessing and model prediction
pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', model)])
pipeline.fit(df[['R&D Spend', 'Administration', 'Marketing Spend', 'State']], df[['Profit']])

# Prediction function
def price_pred(rd, administration, marketing, state):
    input_data = pd.DataFrame({
        'R&D Spend': [rd],
        'Administration': [administration],
        'Marketing Spend': [marketing],
        'State': [state]
    })
    prediction = pipeline.predict(input_data)[0]
    return prediction

# Main function for Streamlit app layout
def main():
    st.set_page_config(page_title="Profit Prediction", page_icon="💰", layout="wide")
    
    # Header Section
    st.title("💼 Profit Prediction Tool")
    st.markdown("""
    Welcome to the **Profit Prediction Tool**! This tool uses machine learning to predict the profit of a company 
    based on different financial parameters. Please input the values for **R&D Spend**, **Administration Spend**, 
    and **Marketing Spend** along with the **State** to get the predicted profit.
    """)
    
    # Sidebar Section for better organization
    st.sidebar.header("Enter Your Company Details")
    state = st.sidebar.selectbox('Select your State', df['State'].unique())
    rd = st.sidebar.number_input('R&D Spend Amount ($)', 0, int(df['R&D Spend'].max()), step=1000)
    administration = st.sidebar.number_input('Administration Spend Amount ($)', 0, int(df['Administration'].max()), step=1000)
    marketing = st.sidebar.number_input('Marketing Spend Amount ($)', 0, int(df['Marketing Spend'].max()), step=1000)

    # Display entered data for user confirmation
    st.markdown("### You entered:")
    st.write(f"**State**: {state}")
    st.write(f"**R&D Spend**: ${rd:,.2f}")
    st.write(f"**Administration Spend**: ${administration:,.2f}")
    st.write(f"**Marketing Spend**: ${marketing:,.2f}")
    
    # Prediction Button
    if st.sidebar.button('Predict Profit'):
        profit = price_pred(rd, administration, marketing, state)
        profit = float(profit)
        st.markdown(f"### The Predicted Profit is: **${profit:,.2f}**")
    
    # Add some styling with markdown and display information
    st.markdown("""
    ---
    ### About the Model
    The model is built using a **regression algorithm** that takes into account the R&D spend, administration expenses, 
    and marketing spend for a company to predict its profit. The model was trained using real-world company data.
    """)

if __name__ == '__main__':
    main()
