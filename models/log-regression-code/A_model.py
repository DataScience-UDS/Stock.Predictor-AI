import numpy as np
import pandas as pd
import pickle

# Loads the stock data of each company into a CSV
def load_stock_data(filePath):
    df = pd.read_csv(filePath)  # Reads stock data and stores into dataframe
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
    return df

class LogisticRegression:
    def __init__(self, learningRate=0.01, numOfIterations=999):
        self.learningRate = learningRate
        self.numOfIterations = numOfIterations
        self.weights = None
        self.bias = None

    def _sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1/(1 + np.exp(-z))

    def fit(self, X, y):
        numOfSamples, numOfFeatures = X.shape
        self.weights = np.zeros(numOfFeatures)
        self.bias = 0

        # Gradient descent
        for _ in range(self.numOfIterations):
            linearModel = np.dot(X, self.weights) + self.bias
            yPred = self._sigmoid(linearModel)

            dw = (1 / numOfSamples) * np.dot(X.T, (yPred - y))
            db = (1 / numOfSamples) * np.sum(yPred - y)

            # update step (this was missing)
            self.weights -= self.learningRate * dw
            self.bias -= self.learningRate * db

    def predict(self, X):
        linearModel = np.dot(X, self.weights) + self.bias
        yPred = self._sigmoid(linearModel)
        return np.array([1 if i > 0.5 else 0 for i in yPred])

if __name__ == "__main__":
    df = load_stock_data("C:/Users/Mabo Giqwa/Documents/Stock Prediction model/Stock.Predictor-AI/data/csv_with_features/A.csv")

    features = ["Close", "High", "Low", "Open", "Volume", "Daily Return(%)", "Monthly Volatility", "Monthly Momentum"]
    X = df[features].values

    # Create target (next day's up/down movement)
    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    y = df["Target"].dropna().values
    X = X[:len(y)]

    # Train-test split
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Train model
    model = LogisticRegression(learningRate=0.001, numOfIterations=9999)
    model.fit(X_train, y_train)

    with open("logistic_A_model.pkl", "wb") as f:
        pickle.dump(model, f)

    # Predict
    predictions = model.predict(X_test)

    # Accuracy
    accuracy = np.mean(predictions == y_test)
    print(f"Accuracy: {accuracy:.2f}")
