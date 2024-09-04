# sampleData.py

import pandas as pd

# Sample data
data = {
    'SeismicActivity': [0.1, 0.3, 0.5, 0.7, 0.2, 0.4],
    'WaterLevel': [100, 150, 200, 250, 180, 220],
    'Temperature': [0, 1, 2, 1, 0, 2],
    'GLOF': [0, 0, 1, 1, 0, 1]
}

# Convert the dictionary to a DataFrame
df = pd.DataFrame(data)

# Save DataFrame to a CSV file
df.to_csv('glof_data.csv', index=False)

# Load the data from the CSV file to verify
df_loaded = pd.read_csv('glof_data.csv')
print(df_loaded)

# Features (input data)
X = df[['SeismicActivity', 'WaterLevel', 'Temperature']]

# Labels (output data)
y = df['GLOF']
[22:36, 9/4/2024] Sathvika IT A: # trainingModel.py

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import sampleData as u
import joblib

# Split the data: 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(u.X, u.y, test_size=0.2, random_state=42)

# Initialize the Decision Tree model
model = DecisionTreeClassifier()

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy * 100:.2f}%')

# Example of new sensor data: SeismicActivity=0.6, WaterLevel=210, Temperature=2
new_data = [[0.6, 210, 2]]

# Predict whether a GLOF will occur
prediction = model.predict(new_data)
print(f'Prediction: {"GLOF will occur" if prediction[0] == 1 else "No GLOF"}')

# Save the model to a file
joblib.dump(model, 'glof_model.pkl')
[22:36, 9/4/2024] Sathvika IT A: import joblib
import pandas as pd

# Load the model from the file
loaded_model = joblib.load('glof_model.pkl')

# Example of new sensor data: SeismicActivity=0.6, WaterLevel=210, Temperature=2
new_data = {
    'SeismicActivity': [0.6],
    'WaterLevel': [210],
    'Temperature': [2]
}

# Convert the new data into a DataFrame with the correct feature names
new_data_df = pd.DataFrame(new_data)

# Use the loaded model to make predictions
loaded_prediction = loaded_model.predict(new_data_df)
print(f'Loaded Model Prediction: {"GLOF will occur" if loaded_prediction[0] == 1 else "No GLOF"}')
