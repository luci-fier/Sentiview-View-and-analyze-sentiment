import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout, Input
from tensorflow.keras.layers import Attention
import joblib
import os
import threading
import webbrowser
from datetime import datetime

# Load the preprocessed data
X_train, X_test, y_train, y_test = joblib.load('data/processed_data.pkl')
tokenizer = joblib.load('data/tokenizer.pkl')

# Define model parameters
vocab_size = 5000  # Vocabulary size set in tokenizer
embedding_dim = 128
max_len = 100  # Same as padding length

# Build the Bi-LSTM with Attention model
inputs = Input(shape=(max_len,))
embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len)(inputs)
bi_lstm = Bidirectional(LSTM(64, return_sequences=True))(embedding_layer)

# Attention Mechanism
attention_layer = Attention()([bi_lstm, bi_lstm])
lstm_output = tf.reduce_sum(attention_layer, axis=1)

# Dropout and final output layer
dropout = Dropout(0.5)(lstm_output)
output = Dense(3, activation='softmax')(dropout)

# Compile the model
model = tf.keras.Model(inputs=inputs, outputs=output)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Display the model architecture
model.summary()

# Setup TensorBoard callback for monitoring training
log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")  # Log directory path with timestamp
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# Function to launch TensorBoard in a separate thread
def launch_tensorboard(log_dir):
    os.system(f"tensorboard --logdir={log_dir}")
    # Open TensorBoard in a web browser after a delay to allow TensorBoard server to start
    webbrowser.open("http://localhost:6006")

# Start TensorBoard in a separate thread
tb_thread = threading.Thread(target=launch_tensorboard, args=(log_dir,))
tb_thread.start()

# Train the model with verbose output and TensorBoard callback
model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=5,
    batch_size=64,
    verbose=1,  # Shows detailed output for each epoch
    callbacks=[tensorboard_callback]
)

# Save the trained model
model.save('bi_lstm_attention_model.h5')
print("Model training completed and saved as 'bi_lstm_attention_model.h5'.")
