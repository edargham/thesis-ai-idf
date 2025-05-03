import numpy as np
from sklearn.preprocessing import StandardScaler

def standardize_data(X_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    """
    Standardizes the training and test data using StandardScaler.

    Parameters:
    - X_train: np.ndarray, training data
    - X_test: np.ndarray, test data

    Returns:
    - X_train_scaled: np.ndarray, standardized training data
    - X_test_scaled: np.ndarray, standardized test data
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled