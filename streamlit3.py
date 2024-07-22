import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Generate sample data
np.random.seed(42)
data = {
    'feature1': np.random.rand(100),
    'feature2': np.random.rand(100),
    'final_grade': np.random.randint(60, 100, 100)
}
df = pd.DataFrame(data)

# Function to train model and return predictions
def train_model(df):
    X = df[['feature1', 'feature2']]
    y = df['final_grade']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    return model, mae, y_pred, y_test

# Set up Streamlit app
st.set_page_config(page_title="PROJECT Demo", layout="wide")

# CSS to set background color
page_bg_color = '''
<style>
[data-testid="stAppViewContainer"] {
    background-color: #BFDDDF;  /* Change this to your desired background color */
    color: black;
}
</style>
'''
st.markdown(page_bg_color, unsafe_allow_html=True)
# App title and description
st.markdown(
    """
    <div style="background-color: #70BDC2; padding: 5px;">
        <h1 style="text-align: center; color: white;">PROJECT DEMO</h1>
    </div>
    """,
    unsafe_allow_html=True,
)

# Display sample dataset
st.write("## Sample Dataset")
st.write(df.head())

# Train and display model accuracy on the sample dataset
model, mae, y_pred, y_test = train_model(df)
st.write(f"## Model Mean Absolute Error on Sample Dataset: {mae:.2f}")

# Display some sample predictions
st.write("### Sample Predictions")
sample_predictions = pd.DataFrame({
    'Predicted Grade': y_pred[:5],
    'Actual Grade': y_test[:5].values
})
st.write(sample_predictions)

# File uploader for user dataset
st.write("## Upload Your Dataset")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    user_df = pd.read_csv(uploaded_file)
    st.write("### Uploaded Dataset")
    st.write(user_df.head())
    
    # Check if the uploaded dataset has the necessary columns
    if set(df.columns).issubset(user_df.columns):
        user_model, user_mae, user_y_pred, user_y_test = train_model(user_df)
        st.write(f"### Model Mean Absolute Error on Uploaded Dataset: {user_mae:.2f}")
        
        # Display some sample predictions on user data
        user_sample_predictions = pd.DataFrame({
            'Predicted Grade': user_y_pred[:5],
            'Actual Grade': user_y_test[:5].values
        })
        st.write("### Sample Predictions on Uploaded Data")
        st.write(user_sample_predictions)
    else:
        st.write("The uploaded dataset does not have the required columns.")

# Add a button to train the model on sample data
if st.button("Train model on sample data"):
    model, mae, y_pred, y_test = train_model(df)
    st.write(f"Model Mean Absolute Error on Sample Dataset: {mae:.2f}")

# Add a button to train the model on user data
if uploaded_file is not None:
    if st.button("Train model on uploaded data"):
        user_model, user_mae, user_y_pred, user_y_test = train_model(user_df)
        st.write(f"Model Mean Absolute Error on Uploaded Dataset: {user_mae:.2f}")
        
        # Display some sample predictions on user data
        user_sample_predictions = pd.DataFrame({
            'Predicted Grade': user_y_pred[:5],
            'Actual Grade': user_y_test[:5].values
        })
        st.write("### Sample Predictions on Uploaded Data")
        st.write(user_sample_predictions)
