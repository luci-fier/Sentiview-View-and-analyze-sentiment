from flask import Flask, request, render_template, redirect, url_for, flash
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
import tensorflow as tf
import joblib
from keras.utils import pad_sequences
from tensorflow.keras.layers import Attention
from datetime import datetime
from auth import User
from db_config import searches_collection

# Memory management
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'hellothere'  # Change this to a secure secret key

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Load the trained model with custom objects
model = tf.keras.models.load_model('bi_lstm_attention_model.h5', custom_objects={'Attention': Attention})

# Load the tokenizer
tokenizer = joblib.load('data/tokenizer.pkl')

# Define the maximum length for padding
max_len = 100

@login_manager.user_loader
def load_user(user_id):
    return User.get(user_id)

@app.route('/')
@login_required
def home():
    return render_template('index.html', prediction=None)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        user = User.get_by_email(email)
        if user and user.check_password(password):
            login_user(user)
            flash('Logged in successfully!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Invalid email or password', 'danger')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        
        if password != confirm_password:
            flash('Passwords do not match', 'danger')
            return render_template('register.html')
        
        user = User.create(username, email, password)
        if user:
            login_user(user)
            flash('Registration successful!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Email or username already exists', 'danger')
    
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Logged out successfully!', 'success')
    return redirect(url_for('home'))

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    # Get user input text
    input_text = request.form['text']
    
    if input_text.strip() == "":
        # If the input text is empty or contains only whitespace, don't make a prediction
        return render_template('index.html', prediction=None)

    # Preprocess the input
    sequences = tokenizer.texts_to_sequences([input_text])
    padded_sequences = pad_sequences(sequences, maxlen=max_len)

    # Predict sentiment
    prediction = model.predict(padded_sequences)
    sentiment_class = prediction.argmax(axis=1)[0]

    # Map the predicted class to a sentiment label
    sentiment_mapping = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    sentiment = sentiment_mapping[sentiment_class]

    # Store the search in MongoDB
    search_data = {
        'user_id': current_user.id,
        'username': current_user.username,
        'text': input_text,
        'sentiment': sentiment,
        'timestamp': datetime.utcnow()
    }
    searches_collection.insert_one(search_data)

    # Return the prediction result
    return render_template('index.html', prediction=f'Sentiment: {sentiment}')

if __name__ == '__main__':
    app.run(debug=True)
