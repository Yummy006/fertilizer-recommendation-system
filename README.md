# 🌾 Fertilizer Recommendation System

A smart web app to recommend the best fertilizer based on soil conditions, crop type, nutrients, and weather. Built with **Flask**, **scikit-learn**, and a **Support Vector Machine (SVM)** model.

---

## 🚀 Features

- Predicts fertilizer based on:
  - Nitrogen (N), Phosphorous (P), Potassium (K)
  - Moisture
  - Soil type
  - Crop type
  - Real-time temperature (auto-fetched using OpenWeatherMap API)
- Frontend: HTML, CSS, JS
- Backend: Flask + SVM
- Trained with one-hot encoded features (`drop_first=True`)
- Uses `StandardScaler` for preprocessing

---

## 📁 Project Structure

```
fertilizer-recommendation/
├── app.py                  # Flask backend
├── templates/
│   └── index.html          # Frontend form
├── svm_model.pkl           # Trained SVM model
├── scaler.pkl              # Trained StandardScaler
├── requirements.txt        # Python dependencies
└── README.md
```

---

## 💻 Local Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/ramyasravanthi/fertilizer-recommendation.git
cd fertilizer-recommendation
```

### 2️⃣ (Optional) Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Get OpenWeatherMap API Key
- Sign up at [https://openweathermap.org/api](https://openweathermap.org/api)
- Copy your API key
- Replace the value in `index.html`:

```javascript
const apiKey = 'YOUR_API_KEY_HERE';
```

### 5️⃣ Run the App
```bash
python app.py
```

Open browser: [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## 📦 Kaggle Notebook Usage

1. Upload `svm_model.pkl` and `scaler.pkl` via "Add data" > "Upload file".
2. Use this sample code:

```python
import pickle
import numpy as np

# Load model
with open('/kaggle/input/your-upload-folder/svm_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load scaler
with open('/kaggle/input/your-upload-folder/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Input features (example)
features = np.array([[26, 45, 80, 40, 30, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]])

# Scale and predict
scaled = scaler.transform(features)
prediction = model.predict(scaled)
print("Recommended Fertilizer:", prediction[0])
```

---

## 🧠 Tech Stack

| Frontend        | Backend       | Machine Learning |
|----------------|---------------|------------------|
| HTML, CSS, JS  | Flask (Python) | SVM + Scikit-learn |


