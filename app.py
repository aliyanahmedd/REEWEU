"""
app.py
======
ReviewGuard — Flask web application for AI fake-review detection.

Brings together:
    - the trained Random Forest + TF-IDF model (model/*.pkl)
    - rule-based fake-indicator detection (FAKE_INDICATORS below)
    - Groq LLM explanations (groq_helper.py)
    - MongoDB storage of every analysed review (reviewguard_db)

Run with: python app.py  ->  http://localhost:5000
"""

import os
import re
import math
from datetime import datetime

import joblib
import nltk
from flask import (Flask, render_template, request, redirect,
                   url_for, flash)
from bson.objectid import ObjectId
from pymongo import MongoClient

import groq_helper
import scraper

# ----------------------------------------------------------------------------
# Setup
# ----------------------------------------------------------------------------
app = Flask(__name__)
app.secret_key = "reviewguard-secret-key"

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(THIS_DIR, "model")

# --- Load the ML artifacts. The SAME TF-IDF vectorizer used in training must
# be used in prediction — it is loaded from its .pkl file here. ---
try:
    MODEL = joblib.load(os.path.join(MODEL_DIR, "review_model.pkl"))
    VECTORIZER = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
    ENCODER = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
except Exception:
    # Model not trained yet — app still loads, predictions return SUSPICIOUS.
    MODEL = VECTORIZER = ENCODER = None

# --- Read the saved accuracy (written by train_model.py) for the dashboard ---
try:
    with open(os.path.join(MODEL_DIR, "accuracy.txt")) as f:
        MODEL_ACCURACY = f.read().strip()
except Exception:
    MODEL_ACCURACY = "N/A"

# --- NLTK stopwords (used by clean_text) with auto-download fallback ---
try:
    from nltk.corpus import stopwords
    STOPWORDS = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords")
    from nltk.corpus import stopwords
    STOPWORDS = set(stopwords.words("english"))

from nltk.stem import PorterStemmer
STEMMER = PorterStemmer()

# ----------------------------------------------------------------------------
# MongoDB — every prediction is stored for history tracking and future model
# improvement. DB name: reviewguard_db.
# ----------------------------------------------------------------------------
try:
    # Connection string comes from .env (MONGO_URI) — use your MongoDB Atlas
    # cloud URI there. Falls back to a local server if the variable is unset.
    from dotenv import load_dotenv
    load_dotenv()
    # `or` (not a default arg) so a blank MONGO_URI= falls back to local.
    MONGO_URI = os.environ.get("MONGO_URI") or "mongodb://localhost:27017/"
    mongo = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    mongo.server_info()  # force a connection check
    db = mongo["reviewguard_db"]
    records_col = db["review_records"]
    stats_col = db["stats"]
    DB_OK = True
except Exception:
    db = records_col = stats_col = None
    DB_OK = False


# ----------------------------------------------------------------------------
# Fake-indicator rules — rule-based pattern matching that catches common
# phrases used in bot-generated or paid reviews.
# ----------------------------------------------------------------------------
FAKE_INDICATORS = {
    "excessive_punctuation": ["!!!!", "!!!", "!!!!!", "????", "....."],
    "generic_praise": [
        "best product ever", "love it so much", "absolutely amazing",
        "highly recommend", "five stars", "perfect product",
        "works perfectly", "great quality",
    ],
    "urgency_language": [
        "buy now", "must buy", "everyone should buy", "dont hesitate",
        "do not hesitate", "worth every penny", "value for money",
    ],
    "suspicious_patterns": [
        "received this for free", "got this for review",
        "sent me this product", "discount in exchange", "free in exchange",
    ],
    "bot_language": [
        "i am very satisfied", "this product is good", "i like this product",
        "good product good", "product is very good",
    ],
}


def detect_fake_indicators(text):
    """
    Rule-based pattern matching — catches common phrases used in bot generated
    or paid reviews. Returns a list like ["five stars [generic_praise]", ...].
    """
    found = []
    text_lower = text.lower()
    for category, patterns in FAKE_INDICATORS.items():
        for pattern in patterns:
            if pattern in text_lower:
                found.append(f"{pattern} [{category}]")
    return found


def clean_text(text):
    """
    Cleans review text exactly the way train_model.py did, so the loaded
    TF-IDF vectorizer produces matching features at prediction time.
    """
    text = str(text).lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    words = text.split()
    words = [STEMMER.stem(w) for w in words if w not in STOPWORDS and len(w) > 1]
    return re.sub(r"\s+", " ", " ".join(words)).strip()


def predict_review(text):
    """
    Predicts the class (GENUINE / FAKE / SUSPICIOUS) and a confidence %
    for a single review using the loaded Random Forest model.

    Falls back to SUSPICIOUS @ 50% if the model has not been trained yet.
    """
    if MODEL is None:
        return "SUSPICIOUS", 50.0

    cleaned = clean_text(text)
    features = VECTORIZER.transform([cleaned])
    pred_num = MODEL.predict(features)[0]
    predicted = ENCODER.inverse_transform([pred_num])[0]
    # confidence = the probability the forest assigned to the winning class
    confidence = float(max(MODEL.predict_proba(features)[0]) * 100)
    return predicted, confidence


def get_stats():
    """
    Returns the running stats document (counts of each verdict). Always
    returns a dict even when MongoDB is unavailable so templates never break.
    """
    default = {"total_analyzed": 0, "fake_count": 0,
               "genuine_count": 0, "suspicious_count": 0}
    if not DB_OK:
        return default
    doc = stats_col.find_one({"_id": "global"})
    return doc if doc else default


def update_stats(predicted_class):
    """Increments the global stat counters after each new prediction."""
    if not DB_OK:
        return
    field = {"FAKE": "fake_count", "GENUINE": "genuine_count",
             "SUSPICIOUS": "suspicious_count"}.get(predicted_class, "suspicious_count")
    stats_col.update_one(
        {"_id": "global"},
        {"$inc": {"total_analyzed": 1, field: 1}},
        upsert=True,
    )


# ----------------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------------
@app.route("/")
def index():
    """Home page — shows headline stats and a call-to-action to analyze."""
    try:
        return render_template("index.html", stats=get_stats(), db_ok=DB_OK)
    except Exception as e:
        return f"Error loading home: {e}", 500


@app.route("/analyze", methods=["GET", "POST"])
def analyze():
    """
    GET  -> show the review input form.
    POST -> run the full pipeline (clean -> predict -> indicators -> Groq ->
            save to MongoDB) and redirect to the result page.
    """
    if request.method == "GET":
        # The scan forms now live on the home page.
        return redirect(url_for("index"))

    try:
        product_name = request.form.get("product_name", "").strip()
        rating = int(request.form.get("rating", 5))
        review_text = request.form.get("review_text", "").strip()

        if not review_text:
            flash("Please enter review text.")
            return redirect(url_for("analyze"))

        # Full pipeline: clean -> predict -> indicators -> Groq explanation/tips.
        record = build_record(product_name, rating, review_text, manually_added=False)

        if DB_OK:
            result = records_col.insert_one(record)        # 6: save
            update_stats(predicted_class)                  # 7: update stats
            return redirect(url_for("result", record_id=str(result.inserted_id)))

        # No DB: render result directly from the in-memory record.
        flash("MongoDB isn't connected, so this result wasn't saved.")
        return render_template("result.html", r=record, no_db=True)
    except Exception as e:
        flash(f"Analysis failed: {e}")
        return redirect(url_for("analyze"))


# Word signals used to build a quick, human-readable reason for each verdict.
# These mirror the kinds of patterns the model learned: fake reviews lean on
# generic superlatives, genuine ones mention specific details / criticism.
PROMO_WORDS = ["amazing", "perfect", "best ever", "best product", "love it",
               "highly recommend", "excellent", "awesome", "wonderful",
               "fantastic", "superb", "satisfied", "value for money",
               "worth every penny", "must buy", "five stars"]
SPECIFIC_WORDS = ["but", "however", "broke", "stopped", "return", "scratch",
                  "weak", "cheap", "late", "smaller", "bigger", "different",
                  "after", "expected", "problem", "battery", "quality",
                  "delivery", "price", "size", "color"]


def get_top_terms(review_text, n=5):
    """
    Returns the words from this review that carried the most weight in the
    model's TF-IDF input (highest TF-IDF score). These are literally the terms
    the Random Forest 'saw' — so they are the real basis for the prediction.
    Returns [] if the model isn't loaded or no words are in its vocabulary.
    """
    if MODEL is None:
        return []
    vec = VECTORIZER.transform([clean_text(review_text)])
    if vec.nnz == 0:
        return []
    names = VECTORIZER.get_feature_names_out()
    # Pair each present feature with its TF-IDF weight, take the top n.
    pairs = sorted(zip(vec.indices, vec.data), key=lambda x: -x[1])[:n]
    return [names[i] for i, _ in pairs]


def explain_local(review_text, predicted_class, indicators, confidence):
    """
    Builds a short, plain-English reason for WHY the model gave this verdict —
    without calling the LLM, so it's instant for bulk analysis.

    Reports concrete signals (promo words, exclamation spam, matched fake
    patterns, specific detail) AND always falls back to the real top TF-IDF
    terms the model weighed, so there is never a reason-less verdict.
    """
    low = review_text.lower()
    promos = [w for w in PROMO_WORDS if w in low]
    specifics = [w for w in SPECIFIC_WORDS if w in low]
    bang = low.count("!")
    top_terms = get_top_terms(review_text)

    bits = [f"We rated this {predicted_class.lower()} and we're about "
            f"{confidence:.0f}% sure."]

    if predicted_class == "FAKE":
        if promos:
            bits.append("It leans on generic praise like "
                        + ", ".join(f'"{w}"' for w in promos[:4]) + ".")
        if bang >= 3:
            bits.append(f"It piles on exclamation marks ({bang} of them).")
        if indicators:
            bits.append("It uses phrases we often see in fake reviews: "
                        + ", ".join(i.split(" [")[0] for i in indicators[:3]) + ".")
        if not specifics:
            bits.append("It never says anything specific about the product, "
                        "which paid or bot reviews tend to skip.")
    elif predicted_class == "GENUINE":
        if specifics:
            bits.append("It mentions real details like "
                        + ", ".join(f'"{w}"' for w in specifics[:4])
                        + ", which is what actual buyers usually do.")
        if not promos:
            bits.append("It skips the over the top praise that fake reviews rely on.")
        if not indicators:
            bits.append("None of our known fake phrases show up.")
    else:  # SUSPICIOUS / fallback
        bits.append("The signals are mixed, so this one is worth a closer look.")

    # Always show the real basis: the words the model actually weighed.
    if top_terms:
        bits.append("The words that weighed most here were "
                    + ", ".join(f'"{w}"' for w in top_terms) + ".")
    else:
        bits.append("The words in this review weren't in the training data "
                    "(for example Roman Urdu), so the model has very little to "
                    "go on. Read the low confidence with that in mind.")

    if confidence < 60:
        bits.append("Confidence is low, so treat this as a weak guess.")

    return " ".join(bits)


def build_record(product_name, rating, review_text, manually_added=False,
                 with_groq=True):
    """
    Runs the full analysis pipeline on one review and returns a record dict.
    Shared by /analyze, /add and /analyze_link so the logic lives in one place.

    with_groq=False skips the LLM calls — used for bulk link analysis where we
    analyze dozens of reviews and two Groq calls each would be far too slow.
    """
    predicted_class, confidence = predict_review(review_text)
    indicators = detect_fake_indicators(review_text)
    if with_groq:
        explanation = groq_helper.explain_prediction(
            review_text, predicted_class, indicators, confidence)
        tips = groq_helper.generate_shopping_tips(predicted_class)
    else:
        explanation = "Open this review to get a full explanation."
        tips = ""
    return {
        "product_name": product_name or "Unknown Product",
        "review_text": review_text,
        "rating": rating,
        "predicted_class": predicted_class,
        "confidence": round(confidence, 1),
        "fake_indicators": indicators,
        # Instant local "why" used for the (i) tooltip on each review.
        "reason": explain_local(review_text, predicted_class, indicators, confidence),
        "groq_explanation": explanation,
        "groq_tips": tips,
        "timestamp": datetime.utcnow(),
        "manually_added": manually_added,
    }


@app.route("/analyze_link", methods=["GET", "POST"])
def analyze_link():
    """
    Analyze a whole product PAGE by URL (any site, best-effort).

    GET  -> show the link input form.
    POST -> scrape reviews from the URL, analyze each, save them, and show a
            summary table. If scraping fails, flash an error and tell the user
            to paste text manually instead.
    """
    if request.method == "GET":
        # The link form now lives on the combined Analyze page.
        return redirect(url_for("analyze"))

    try:
        url = request.form.get("product_url", "").strip()
        product_name = request.form.get("product_name", "").strip()

        result = scraper.scrape_reviews(url)
        if not result["ok"]:
            # Graceful fallback — scraping is unreliable by nature.
            flash(result["error"] + " You can paste the review text manually instead.")
            return redirect(url_for("analyze"))

        analyzed = []
        for item in result["reviews"]:
            text = item["text"]
            # Use the scraped rating if present, else default to 3 (unknown).
            try:
                rating = int(float(item.get("rating"))) if item.get("rating") else 3
            except (TypeError, ValueError):
                rating = 3
            # Skip Groq per-review here — too many reviews to call the LLM each.
            record = build_record(product_name, rating, text,
                                  manually_added=False, with_groq=False)
            if DB_OK:
                record["_id"] = records_col.insert_one(record).inserted_id
                update_stats(record["predicted_class"])
            analyzed.append(record)

        # Quick summary counts for the results page.
        summary = {
            "total": len(analyzed),
            "fake": sum(1 for r in analyzed if r["predicted_class"] == "FAKE"),
            "genuine": sum(1 for r in analyzed if r["predicted_class"] == "GENUINE"),
            "suspicious": sum(1 for r in analyzed if r["predicted_class"] == "SUSPICIOUS"),
        }
        return render_template("link_results.html", reviews=analyzed,
                               summary=summary, url=url,
                               product_name=product_name or "this product",
                               db_ok=DB_OK)
    except Exception as e:
        flash(f"Link analysis failed: {e}")
        return redirect(url_for("analyze_link"))


@app.route("/result/<record_id>")
def result(record_id):
    """Result page — verdict banner, indicators, Groq explanation + tips."""
    try:
        if not DB_OK:
            flash("MongoDB not connected.")
            return redirect(url_for("analyze"))
        r = records_col.find_one({"_id": ObjectId(record_id)})
        if not r:
            flash("Record not found.")
            return redirect(url_for("records"))
        return render_template("result.html", r=r, no_db=False)
    except Exception as e:
        flash(f"Could not load result: {e}")
        return redirect(url_for("records"))


@app.route("/records")
def records():
    """Records table (READ) with product search, verdict filter, pagination."""
    try:
        stats = get_stats()
        if not DB_OK:
            return render_template("records.html", records=[], page=1,
                                   total_pages=1, search="", filter_val="All",
                                   db_ok=False)

        search = request.args.get("search", "").strip()
        filter_val = request.args.get("filter", "All")
        page = max(1, int(request.args.get("page", 1)))
        per_page = 10

        query = {}
        if search:
            query["product_name"] = {"$regex": re.escape(search), "$options": "i"}
        if filter_val in ("GENUINE", "FAKE", "SUSPICIOUS"):
            query["predicted_class"] = filter_val

        total = records_col.count_documents(query)
        total_pages = max(1, math.ceil(total / per_page))
        rows = list(records_col.find(query)
                    .sort("timestamp", -1)
                    .skip((page - 1) * per_page)
                    .limit(per_page))

        return render_template("records.html", records=rows, page=page,
                               total_pages=total_pages, search=search,
                               filter_val=filter_val, db_ok=True)
    except Exception as e:
        return f"Error loading records: {e}", 500


@app.route("/add", methods=["GET", "POST"])
def add():
    """Manually add a review record (CREATE). Still runs prediction + Groq."""
    if request.method == "GET":
        return render_template("edit.html", r=None, action="add")

    try:
        product_name = request.form.get("product_name", "").strip()
        rating = int(request.form.get("rating", 5))
        review_text = request.form.get("review_text", "").strip()

        record = build_record(product_name, rating, review_text, manually_added=True)

        if DB_OK:
            records_col.insert_one(record)
            update_stats(record["predicted_class"])
            flash("Review added.")
        else:
            flash("MongoDB not connected — not saved.")
        return redirect(url_for("records"))
    except Exception as e:
        flash(f"Could not add review: {e}")
        return redirect(url_for("add"))


@app.route("/edit/<record_id>", methods=["GET", "POST"])
def edit(record_id):
    """Edit an existing review record (UPDATE)."""
    try:
        if not DB_OK:
            flash("MongoDB not connected.")
            return redirect(url_for("records"))

        r = records_col.find_one({"_id": ObjectId(record_id)})
        if not r:
            flash("Record not found.")
            return redirect(url_for("records"))

        if request.method == "GET":
            return render_template("edit.html", r=r, action="edit")

        # POST: update the editable fields.
        product_name = request.form.get("product_name", "").strip()
        rating = int(request.form.get("rating", r.get("rating", 5)))
        review_text = request.form.get("review_text", "").strip()

        records_col.update_one(
            {"_id": ObjectId(record_id)},
            {"$set": {"product_name": product_name,
                      "rating": rating,
                      "review_text": review_text}},
        )
        flash("Review updated.")
        return redirect(url_for("records"))
    except Exception as e:
        flash(f"Could not edit review: {e}")
        return redirect(url_for("records"))


@app.route("/delete/<record_id>", methods=["POST"])
def delete(record_id):
    """Delete a review record (DELETE). Confirmed via JS confirm() in the UI."""
    try:
        if DB_OK:
            records_col.delete_one({"_id": ObjectId(record_id)})
            flash("Review deleted.")
        return redirect(url_for("records"))
    except Exception as e:
        flash(f"Could not delete: {e}")
        return redirect(url_for("records"))


@app.route("/dashboard")
def dashboard():
    """Dashboard — stat cards, verdict bar chart, accuracy, recent reviews."""
    try:
        stats = get_stats()
        total = stats.get("total_analyzed", 0) or 0
        fake_rate = round((stats.get("fake_count", 0) / total) * 100, 1) if total else 0.0
        recent = []
        if DB_OK:
            recent = list(records_col.find().sort("timestamp", -1).limit(5))
        return render_template("dashboard.html", stats=stats, fake_rate=fake_rate,
                               accuracy=MODEL_ACCURACY, recent=recent, db_ok=DB_OK)
    except Exception as e:
        return f"Error loading dashboard: {e}", 500


# Jinja helper: turn a 1-5 rating into filled/empty stars for display.
@app.template_filter("stars")
def stars_filter(rating):
    """Renders an int rating as ★/☆ characters for templates."""
    try:
        rating = int(rating)
    except (TypeError, ValueError):
        rating = 0
    return "★" * rating + "☆" * (5 - rating)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
