import pandas as pd
import joblib

# Step 1: Load the trained model
rf_model = joblib.load('fraud_detection_model.pkl')

# Step 2: Load the original dataset (to extract features)
data = pd.read_csv("C:/Users/Syed Ashari/Desktop/Developer's Hub Internship/Task 3/creditcard.csv")

# Step 3: Extract the feature names (excluding 'Class' and 'Amount')
features = [col for col in data.columns if col not in ['Class', 'Amount']]

# Step 4: Accept user input for each feature
user_input = {}
for feature in features:
    if feature == "Time":  # If the feature is 'Time', handle it specially
        time_input = input(f"Enter value for {feature} (in hh:mm format): ")
        # Convert the time input to seconds (e.g., '23:00' -> 82800 seconds)
        hours, minutes = map(int, time_input.split(":"))
        user_input[feature] = hours * 3600 + minutes * 60  # Convert to seconds
    else:
        user_input[feature] = float(input(f"Enter value for {feature}: "))

# Step 5: Convert user input to a pandas DataFrame (matching the feature names used for training)
user_df = pd.DataFrame([user_input])

# Step 6: Make a prediction using the loaded model
prediction = rf_model.predict(user_df)

# Step 7: Show the result
if prediction == 1:
    print("Fraud detected!")
else:
    print("No fraud detected.")
