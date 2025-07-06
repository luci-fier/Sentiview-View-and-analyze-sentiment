import joblib
import tensorflow as tf
from sklearn.metrics import classification_report, accuracy_score

# Load the preprocessed data and trained model
X_train, X_test, y_train, y_test = joblib.load('data/processed_data.pkl')
model = tf.keras.models.load_model('bi_lstm_attention_model.h5')

# Make predictions
y_pred = model.predict(X_test)
y_pred_classes = y_pred.argmax(axis=1)

# Print classification report and accuracy
print("Model Accuracy:", accuracy_score(y_test, y_pred_classes))
print(classification_report(y_test, y_pred_classes, target_names=['Negative', 'Neutral', 'Positive']))
