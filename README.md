#  REEWEU — AI Fake Review Detector

An AI-powered Flask + MongoDB web app that detects fake / bot / paid product
reviews using NLP and Machine Learning (TF-IDF + Random Forest), with Groq
LLM-powered explanations of why a review is flagged.

Built for SZABIST Semester 6 AI Lab.

---

## Features
- **3-class detection:** GENUINE / FAKE / SUSPICIOUS
- **ML model:** TF-IDF (1-2 grams) + Random Forest (200 trees)
- **Rule-based indicators:** catches generic praise, urgency language, bot phrases
- **Groq LLM explanations** (llama-3.1-8b-instant) — plain-English "why"
- **Full CRUD** over review records (MongoDB)
- **Dashboard** with Chart.js bar chart, model accuracy, fake rate

---

## Tech Stack
Flask · MongoDB (PyMongo) · scikit-learn · NLTK · Groq · Jinja2 · Chart.js

---

## Project Structure
```
reviewguard/
├── app.py                 # Flask routes + ML inference + indicators
├── groq_helper.py         # Groq LLM explanation + tips
├── model/
│   ├── train_model.py     # Train & save model artifacts
│   ├── review_model.pkl   # (generated)
│   ├── tfidf_vectorizer.pkl
│   └── label_encoder.pkl
├── data/reviews.csv       # Kaggle dataset (you provide)
├── templates/             # Jinja2 HTML
├── static/style.css
├── .env                   # GROQ_API_KEY (you create)
├── requirements.txt
└── README.md
```

---

## Setup

1. Clone / unzip the project, then from the `reviewguard/` folder:

```bash
pip install -r requirements.txt
python -m nltk.downloader stopwords
```

2. **Dataset:** Search Kaggle for the *Amazon Fake Reviews Dataset*, download it,
   and save it as `data/reviews.csv` (needs a text column and a label column).

3. **Groq key:** Sign up free at https://console.groq.com, create an API key,
   then create a `.env` file (copy `.env.example`):

```
GROQ_API_KEY=your_key_here
```

4. **MongoDB:** make sure MongoDB is running locally at
   `mongodb://localhost:27017/` (DB name `reviewguard_db` is created automatically).

5. **Train the model** (run once):

```bash
python model/train_model.py
```

Expected accuracy: **85%+**.

6. **Run the app:**

```bash
python app.py
```

Open http://localhost:5000

---

## How It Works (viva summary)
1. Review text is cleaned (lowercase, strip HTML/special chars, remove
   stopwords, Porter stemming).
2. **TF-IDF** converts the text to numbers by word frequency — common words get
   lower weight, informative words higher.
3. **Random Forest** (an ensemble of decision trees) votes on the class.
4. Rule-based pattern matching flags common fake phrases.
5. **Groq LLM** turns all of this into a human-readable explanation + tips.
6. Everything is stored in **MongoDB** for history and future improvement.

---

## Notes
- If MongoDB or the Groq key is missing, the app degrades gracefully instead of
  crashing (analysis still runs; results just aren't saved / explained).
- Flask runs on port 5000 with debug mode on.
