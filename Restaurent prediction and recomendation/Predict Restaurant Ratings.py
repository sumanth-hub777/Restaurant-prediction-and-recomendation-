# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('restaurant_data.csv')  # Change the file name if needed

# Display first few rows
print("First 5 rows of the dataset:")
print(df.head())

# Check missing values
print("\nMissing values in each column:")
print(df.isnull().sum())

# Handle missing values
df['Cuisines'].fillna(df['Cuisines'].mode()[0], inplace=True)

# Encode categorical variables
label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Define features and target
X = df.drop('Aggregate rating', axis=1)  # Features (drop target column)
y = df['Aggregate rating']               # Target

# Split the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Train Decision Tree Regressor
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)

# Predictions
y_pred_lr = lr_model.predict(X_test)
y_pred_dt = dt_model.predict(X_test)

# Evaluate models
print("\nLinear Regression Performance:")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred_lr):.2f}")
print(f"R2 Score: {r2_score(y_test, y_pred_lr):.2f}")

print("\nDecision Tree Regressor Performance:")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred_dt):.2f}")
print(f"R2 Score: {r2_score(y_test, y_pred_dt):.2f}")

# Feature importance - Linear Regression
print("\nFeature Importance from Linear Regression (coefficients):")
feature_importance_lr = pd.Series(lr_model.coef_, index=X.columns)
print(feature_importance_lr.sort_values(ascending=False))

# Plot feature importance for Linear Regression
plt.figure(figsize=(10,6))
feature_importance_lr.sort_values(ascending=False).plot(kind='bar')
plt.title('Feature Importance - Linear Regression')
plt.show()

# Feature importance - Decision Tree
print("\nFeature Importance from Decision Tree Regressor:")
feature_importance_dt = pd.Series(dt_model.feature_importances_, index=X.columns)
print(feature_importance_dt.sort_values(ascending=False))

# Plot feature importance for Decision Tree
plt.figure(figsize=(10,6))
feature_importance_dt.sort_values(ascending=False).plot(kind='bar', color='orange')
plt.title('Feature Importance - Decision Tree Regressor')
plt.show()
