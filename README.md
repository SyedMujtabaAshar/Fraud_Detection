# Fraud_Detection
# Fraud Detection using Random Forest Model

This project uses a Random Forest machine learning model to detect fraudulent credit card transactions. The model is trained on the `creditcard.csv` dataset and predicts whether a transaction is fraudulent based on the input features.

## Features:
- **Time**: The time of the transaction (in seconds from midnight).
- **V1 to V28**: Anonymized features used for fraud detection.

## How to Use

1. Ensure that you have the required libraries installed: `pandas`, `scikit-learn`, `imbalanced-learn`, and `joblib`.
2. The model has been pre-trained and saved in a `.pkl` file.
3. Run the script, and when prompted, enter the **Time** of the transaction in `hh:mm` format.
4. The model will predict if the transaction is fraudulent or not.

### Example Output:
- **Fraud detected!**
- **No fraud detected.**

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

- The dataset used in this project is based on the `creditcard.csv` dataset for fraud detection.
- Thanks to `scikit-learn` and `imbalanced-learn` for their useful machine learning tools.



