import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.under_sampling import RandomUnderSampler
import joblib  # Import joblib for saving the model
import warnings

# Remove warning
warnings.filterwarnings('ignore')

# Load the data
data = pd.read_csv("C:/Users/Syed Ashari/Desktop/Developer's Hub Internship/Task 3/creditcard.csv")

# Step 1: Data Preprocessing
X = data.drop(['Class', 'Amount'], axis=1)  # Exclude 'Class' and 'Amount' from features
y = data['Class']

# Handle imbalanced data using undersampling
undersampler = RandomUnderSampler()
X_resampled, y_resampled = undersampler.fit_resample(X, y)

# Step 2: Model Training
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Step 3: Model Evaluation
y_pred = rf_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print(classification_report(y_test, y_pred))

# Step 4: Save the model to a file
joblib.dump(rf_model, 'fraud_detection_model.pkl')  # Save the trained model
