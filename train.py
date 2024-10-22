import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def load_dataset():
    # Load the California Housing dataset
    housing = fetch_california_housing()
    
    # Convert to pandas DataFrame for easier manipulation
    df = pd.DataFrame(housing.data, columns=housing.feature_names)
    df['MedHouseVal'] = housing.target  # Target variable is median house value
    
    print("Dataset loaded successfully.")
    return df

def train_model(X_train, y_train):
    # Train a linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    print(f"Model trained: coefficients={model.coef_}, intercept={model.intercept_}")
    return model

def evaluate_model(model, X_test, y_test):
    # Make predictions on the test set
    y_pred = model.predict(X_test)
    
    # Calculate Mean Squared Error
    mse = mean_squared_error(y_test, y_pred)
    print(f"Model evaluation completed. Mean Squared Error: {mse:.2f}")


    # Step 1: Load dataset
df = load_dataset()
    
    # Step 2: Split the data into features and target
X = df.drop('MedHouseVal', axis=1)  # Features
y = df['MedHouseVal']  # Target
    
    # Step 3: Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Step 4: Trai the model
model = train_model(X_train, y_train)
    
    # Step 5: Evaluate the model
evaluate_model(model,X_test,y_test)

