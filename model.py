import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load data
data = pd.read_csv("csp.csv")


def one_hot_encode(df, column_dict):
    df = df.copy()
    for column, prefix in column_dict.items():
        dummies = pd.get_dummies(df[column], prefix=prefix)
        df = pd.concat([df, dummies], axis=1)
        df = df.drop(column, axis=1)
    return df


nominal_feature_dict = {'property_type': 'Property Type', 'location_area': 'Location Area'}
data = one_hot_encode(data, nominal_feature_dict)

y = data['approval_status']
X = data.drop('approval_status', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5, random_state=123)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

model = RandomForestClassifier()
model.fit(X_train, y_train)

pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
pickle.dump(X.columns, open("columns.pkl", "wb"))
pickle.dump(nominal_feature_dict, open("feature_dict.pkl", "wb"))
