# 🚨 RAKSHAK AI — Road Accident Prediction & Prevention

India road accident analytics dashboard with ML-based severity prediction.

## Project Structure

```
accident_app/
├── app.py                     # Main Streamlit app (5 tabs)
├── requirements.txt           # Python dependencies
├── accident_prediction.csv    # Dataset (3000 records, 22 features)
├── model.pkl                  # Trained Random Forest model
├── encoders.pkl               # Label encoders for categorical features
└── label_encoder.pkl          # Severity class encoder
```

## Features

| Tab | Content |
|-----|---------|
| 📊 Overview | Severity distribution, yearly trend, vehicle & location analysis |
| 🗺️ State Analysis | Top states, fatal accidents by state, road type breakdown |
| ⏰ Time & Weather | Hourly/daily/monthly patterns, weather & lighting impact |
| 🔬 Risk Factors | Driver age, alcohol, speed, road condition, feature importance |
| 🤖 Predict Severity | Interactive form → ML prediction + prevention tips |

## ML Model

- **Algorithm**: Random Forest (200 estimators)
- **Target**: Accident Severity (Fatal / Serious / Minor)
- **Features**: 17 features — speed, driver age, weather, road type, alcohol, etc.
- **Note**: Dataset is synthetically generated, so accuracy reflects random baseline (~33%). The dashboard analytics and UI are the main deliverables.

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Cloud (Free) — Step by Step

1. **Push to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit - RAKSHAK AI"
   git remote add origin https://github.com/YOUR_USERNAME/rakshak-ai.git
   git push -u origin main
   ```

2. **Go to** [share.streamlit.io](https://share.streamlit.io)

3. **Sign in** with GitHub

4. **Click "New app"** → Select your repo → Branch: `main` → Main file: `app.py`

5. **Click Deploy** — live URL in ~2 minutes ✅

> ⚠️ Make sure all 5 files (app.py, requirements.txt, accident_prediction.csv, model.pkl, encoders.pkl, label_encoder.pkl) are pushed to GitHub.

## Built With

- Python · Scikit-learn · Streamlit · Plotly
- India Road Accident Dataset (3000 records, 32 states)
