# Sentiview — View and Analyze Sentiment

Sentiview is a modern web application for sentiment analysis, powered by a deep learning model (Bi-LSTM with Attention) and a sleek, interactive UI. It allows registered users to analyze the sentiment of text data (e.g., tweets, reviews, messages) and stores each analysis securely for further insights.

---

## 🌟 Features

- **Deep Learning Model:** Uses a custom-trained Bi-LSTM with Attention (TensorFlow/Keras) for robust sentiment classification (Negative, Neutral, Positive).
- **User Authentication:** Secure registration and login using Flask-Login.
- **Beautiful UI:** Responsive, animated interface with Bootstrap 5 and custom theming.
- **Sentiment History:** Each user's predictions are securely logged in MongoDB.
- **Real-Time Feedback:** Instant sentiment prediction with emoji and confetti for positive sentiment.
- **Admin/Analytics Ready:** Easily extendable for dashboards and usage analysis.

---

## 🚀 Live Demo

> Run locally (see below) and visit: [http://localhost:5000](http://localhost:5000)

---

## 🗂️ Project Structure

```
Sentiview---View-and-analyze-sentiment/
│
├── app.py                       # Main Flask server and endpoints
├── db_config.py                 # MongoDB connection and collections
├── auth.py                      # User model and authentication logic
├── bi_lstm_with_attention.py    # Model architecture and training
├── evaluate_bi_lstm_model.py    # Model evaluation script
├── data_preprocessing_for_lstm.py # Data cleaning/prep for model
├── data/
│   ├── processed_data.pkl       # Preprocessed data (generated)
│   └── tokenizer.pkl            # Tokenizer object (generated)
├── templates/
│   ├── base.html
│   ├── index.html
│   ├── login.html
│   └── register.html
├── static/                      # Static assets (CSS, JS, images)
└── README.md
```

---

## ⚙️ How to Run

### 1. **Clone the repository**

```sh
git clone https://github.com/luci-fier/Sentiview---View-and-analyze-sentiment.git
cd Sentiview---View-and-analyze-sentiment
```

### 2. **Set up Python Environment**

It's recommended to use a virtual environment:

```sh
python -m venv venv
source venv/bin/activate          # On Linux/macOS
venv\Scripts\activate             # On Windows
```

### 3. **Install Dependencies**

```sh
pip install -r requirements.txt
```
**(If `requirements.txt` is missing, install manually: Flask, Flask-Login, tensorflow, keras, joblib, pymongo, etc.)**

### 4. **Set up MongoDB**

- The app uses MongoDB Atlas by default (see `db_config.py`).  
- You can edit `MONGO_URI` in `db_config.py` for your own database.

### 5. **Train or Download the Model**

- **To train:** Run `bi_lstm_with_attention.py` after preparing your dataset with `data_preprocessing_for_lstm.py`.
- **Or:** Use the provided model files (`bi_lstm_attention_model.h5` and `tokenizer.pkl`) in the root directory.

### 6. **Run the Application**

```sh
python app.py
```

- The app will be available at [http://localhost:5000](http://localhost:5000)

---

## 🧑‍💻 Usage

- **Register/Login:** Create an account to access the sentiment analysis features.
- **Analyze Sentiment:** Enter any text on the home page and submit to get instant feedback.
- **View Results:** Positive, Negative, and Neutral results are shown with emojis and confetti for positive predictions.
- **Your Analysis History:** All analyses are securely saved for your account.

---

## 📈 Model Details

- **Type:** Bi-directional LSTM with Attention
- **Framework:** TensorFlow/Keras
- **Training:** See `bi_lstm_with_attention.py` and `data_preprocessing_for_lstm.py` for full pipeline.
- **Evaluation:** Use `evaluate_bi_lstm_model.py` to view metrics (accuracy, precision, recall).

---

## 🤝 Credits

- **UI/UX:** [Bootstrap 5](https://getbootstrap.com/), Animate.css, Confetti.js
- **Backend:** Flask, Flask-Login, PyMongo/MongoDB
- **ML/DL:** TensorFlow, Keras

---

## 📄 License

Distributed under the MIT License. See [LICENSE](LICENSE) for more information.

---

