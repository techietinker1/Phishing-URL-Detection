import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load data
df = pd.read_csv("PhiUSIIL_Phishing_URL_Dataset.csv")

# Create simple features
df['url_length'] = df['URL'].apply(lambda x: len(x))
df['count_dot'] = df['URL'].apply(lambda x: x.count('.'))
df['count_at'] = df['URL'].apply(lambda x: x.count('@'))
df['count_hyphen'] = df['URL'].apply(lambda x: x.count('-'))

# Prepare features and target
X = df[['url_length', 'count_dot', 'count_at', 'count_hyphen']]
y = df['label'] 

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save model
joblib.dump(model, 'phishing_model.pkl')
print("Model saved as phishing_model.pkl")