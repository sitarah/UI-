import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Function to load data
def load_data():
    data = load_iris()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    
    df['target'] = data.target
    return df

# Function to train model and return predictions
def train_model(df):
    X = df.drop('target', axis=1)
    y = df['target']
    if len(y.unique()) < 2:
        raise ValueError("This solver needs samples of at least 2 classes in the data, but the data contains only one class.")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    print(X_test.head(5), y_test[:5])
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = LogisticRegression(random_state=42)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    # y_pred_just_one = model.predict(X_test_scaled[:1])
    
    accuracy = accuracy_score(y_test, y_pred)
    print(accuracy)
    print(y_pred)
    return model, accuracy, y_pred, y_test

# Set up the Streamlit app
st.set_page_config(page_title="Machine Learning Pipeline Demo", layout="wide")

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
df = load_data()
st.write("## Sample Dataset")
st.write(df.head())

# Train and display model accuracy on the sample dataset
model, accuracy, y_pred, y_test = train_model(df)

st.write(f"## Model accuracy on sample dataset: {accuracy:.2f}")

# Map the numerical labels to species names
species_mapping = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}
y_pred_names = [species_mapping[label] for label in y_pred]
y_test_names = [species_mapping[label] for label in y_test]

# Display some sample predictions
st.write("### Sample Predictions")
for i in range(5):
    st.write(f"Predicted: {y_pred_names[i]}, Actual: {y_test_names[i]}")

# File uploader for user dataset
st.write("## Upload Your Dataset")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    user_df = pd.read_csv(uploaded_file)
    st.write("### Uploaded Dataset")
    st.write(user_df.head())
    
    # Check if the uploaded dataset has the necessary columns
    if set(df.columns).issubset(user_df.columns):
        try:
            user_model, user_accuracy, user_y_pred, user_y_test = train_model(user_df)
            user_y_pred_names = [species_mapping[label] for label in user_y_pred]
            user_y_test_names = [species_mapping[label] for label in user_y_test]
            st.write(f"### Model accuracy on uploaded dataset: {user_accuracy:.2f}")
            
            # Display some sample predictions on user data
            st.write("### Sample Predictions on Uploaded Data")
            for i in range(5):
                st.write(f"Predicted: {user_y_pred_names[i]}, Actual: {user_y_test_names[i]}")
        except ValueError as e:
            st.write(f"ValueError: {e}")
    else:
        st.write("The uploaded dataset does not have the required columns.")

# Add a button to train the model on sample data
if st.button("Train model on sample data"):
    model, accuracy, y_pred, y_test = train_model(df)
    st.write(f"Model accuracy on sample dataset: {accuracy:.2f}")

# Add a button to train the model on user data
if uploaded_file is not None:
    if st.button("Train model on uploaded data"):
        try:
            user_model, user_accuracy, user_y_pred, user_y_test = train_model(user_df)
            user_y_pred_names = [species_mapping[label] for label in user_y_pred]
            user_y_test_names = [species_mapping[label] for label in user_y_test]
            st.write(f"Model accuracy on uploaded dataset: {user_accuracy:.2f}")
            
            # Display some sample predictions on user data
            st.write("### Sample Predictions on Uploaded Data")
            for i in range(5):
                st.write(f"Predicted: {user_y_pred_names[i]}, Actual: {user_y_test_names[i]}")
        except ValueError as e:
            st.write(f"ValueError: {e}")
