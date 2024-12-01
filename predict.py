# Example of training and saving model and scaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

# Load your training data
# df = pd.read_csv('your_training_data.csv') 

# Train the model (RandomForest example)
X_train = df.drop("quality", axis=1)
y_train = df["quality"]
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the trained model and scaler
joblib.dump(model, 'wine_quality_model.pkl')

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
joblib.dump(scaler, 'scaler.pkl')
