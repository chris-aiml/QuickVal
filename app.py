import os
import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Load model and scaler properly
try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("columns.pkl", "rb") as f:
        model_columns = pickle.load(f)
    with open("feature_dict.pkl", "rb") as f:
        nominal_feature_dict = pickle.load(f)  # Load nominal feature dict
except FileNotFoundError as e:
    print(f"Error loading model files: {e}")
    exit(1)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get form data (Fixing naming issue)
        property_age = float(request.form['Property_Age'])
        num_rooms = float(request.form['Num_Rooms'])
        num_bathrooms = float(request.form['Num_Bathrooms'])
        property_type = request.form['Property_Type']
        location_area = request.form['Location_Area']

        # Create input data dictionary
        input_data = {
            'property_type': [property_type],
            'location_area': [location_area],
            'property_age': [property_age],
            'num_rooms': [num_rooms],
            'num_bathrooms': [num_bathrooms]
        }

        # Convert to DataFrame
        input_df = pd.DataFrame(input_data)

        # One-hot encode categorical variables
        processed_df = one_hot_encode(input_df, nominal_feature_dict)

        # Ensure all columns from training exist
        for col in model_columns:
            if col not in processed_df.columns:
                processed_df[col] = 0
        processed_df = processed_df[model_columns]

        # Scale features
        scaled_features = scaler.transform(processed_df)

        # Make prediction
        prediction = model.predict(scaled_features)[0]
        result = "Approved" if prediction == 1 else "Not Approved"

        return render_template("index.html",
                               prediction_text=f"Status: {result}",
                               property_age=property_age,
                               num_rooms=num_rooms,
                               num_bathrooms=num_bathrooms,
                               property_type=property_type,
                               location_area=location_area)

    except Exception as e:
        return render_template("index.html",
                               prediction_text=f"Error: {str(e)}")


def one_hot_encode(df, column_dict):
    """One-hot encode categorical features"""
    df = df.copy()
    for column, prefix in column_dict.items():
        dummies = pd.get_dummies(df[column], prefix=prefix)
        df = pd.concat([df, dummies], axis=1)
        df = df.drop(column, axis=1)
    return df


if __name__ == "__main__":
    # Verify required files exist
    required_files = ['model.pkl', 'scaler.pkl', 'columns.pkl', 'feature_dict.pkl']
    for file in required_files:
        if not os.path.exists(file):
            print(f"Error: Missing required file {file}")
            exit(1)

    app.run(debug=True)
