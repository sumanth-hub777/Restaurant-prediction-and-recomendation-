import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score

# Load data
file_path = 'zomato.csv'
df = pd.read_csv(file_path)

# Preview the data
print("Sample Data Preview:")
print(df.head())

# Preprocess the dataset
df = df.dropna(subset=['Cuisines', 'Restaurant Name'])  # Drop rows with missing Cuisines or Restaurant Name

# Encoding categorical variables: 'Cuisines' and 'City'
encoder = LabelEncoder()
df['Cuisine_Label'] = encoder.fit_transform(df['Cuisines'])

# Split the dataset
X = df[['Average Cost for two', 'Aggregate rating']]  # Features: Add more features if needed
y = df['Cuisine_Label']  # Target variable: Cuisines (encoded)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy * 100:.2f}%")

# Classification report
# Fixing the mismatch by specifying 'labels' parameter to match the classes in y_test
unique_classes_in_test = pd.unique(y_test)  # Get unique classes from y_test
print("\nClassification Report:")
print(classification_report(y_test, y_pred, labels=unique_classes_in_test, target_names=encoder.classes_[unique_classes_in_test]))

# Cross-validation
cv_scores = cross_val_score(model, X, y, cv=5)
print(f"\nCross-validation scores: {cv_scores}")
print(f"Average cross-validation score: {cv_scores.mean():.2f}")

# Identify cuisine class imbalances
class_counts = df['Cuisine_Label'].value_counts()
print("\nClass distribution of cuisines in the dataset:")
print(class_counts)
