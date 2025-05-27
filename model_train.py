import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pickle

# Load dataset
df = pd.read_csv("fakecurrency.csv", header=None)
df.columns = ['variance', 'skew', 'curtosis', 'entropy', 'label']

# Split data
X = df[['variance', 'skew', 'curtosis', 'entropy']]
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Save the model safely
with open("logistic_model.pkl", "wb") as f:
    pickle.dump(model, f)
