# 📈 FinBERT: Real-Time Financial Sentiment Analysis

Machine Learning Pipeline, which analyses news headlines about finance and forecasts sentiments (Bullish, Bearish, and Neutral).

The project will train a BERT-based model with PyTorch, implement API prediction requests using FastAPI, and display visualization results on a React-Bootstrap web interface.

### 🔗 Project Links
* **Live Application:** [Link](https://portfolio-frontend-livid.vercel.app/projects/sentiment-analysis-with-bert)
* **Project Blog Post:** [Read the full write-up on Medium/Dev.to]
* **Video Walkthrough:** [Watch the explanation on YouTube]
* **Model Weights (Optional):** [Link to your Hugging Face model hub if uploaded]

---

## 🏗️ System Architecture

* **Machine Learning:** PyTorch, Hugging Face `transformers`, Financial PhraseBank Dataset
* **Backend API:** Python, FastAPI, Uvicorn
* **Frontend UI:** React, React-Bootstrap
* **Database:** PostgreSQL (Neon/Supabase) via SQLAlchemy

---

## 🧠 The Machine Learning Pipeline

The core of this application is a fine-tuned NLP model.
1. **Base Model:** `ProsusAI/finbert`
2. **Fine-tuning:** Conducted in Google Colab using a T4 GPU. 
3. **Training Data:** The Kaggle Financial PhraseBank dataset.
*(Check the `/notebooks` directory to see the complete PyTorch training loop, tokenization process, and evaluation metrics).*

---

## 🚀 How to Run Locally

### 1. Clone the Repository

copy the command below and run it in your favourite terminal.
```bash
git clone https://github.com/mobadara/finbert-sentiment-analyzer-api.git &&
cd finbert-sentiment-analyzer-api.git
```

### 2. Create a virtual environment
```
python -m venv venv
```

### 3. Activate the virtual environment
#### a. On Linux/Mac
```bash
source venv/bin/activate
```

#### b. On Windows
```bash
venv\Scripts\activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Start the server
```bash
uvicorn app.main:app --reload
```

## 👨‍💻 Author:
**Muyiwa J. Obadara**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-%230077B5.svg?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/obadara-m)
[![X / Twitter](https://img.shields.io/badge/X-%23000000.svg?style=for-the-badge&logo=X&logoColor=white)](https://twitter.com/m_obadara)
[![Portfolio](https://img.shields.io/badge/Portfolio-Navy_Blue?style=for-the-badge&logo=google-chrome&logoColor=white&color=navy)](https://portfolio-frontend-livid.vercel.app)