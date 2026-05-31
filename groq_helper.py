"""
groq_helper.py
==============
Wraps the Groq LLM (llama-3.1-8b-instant) to produce human-readable
explanations and shopping tips for ReviewGuard.

Groq LLM provides a plain-English explanation of why the ML model flagged a
review. llama-3.1-8b-instant is used because it is the fastest free model.

Every call is wrapped in try/except: if the Groq API key is missing or the
request fails, we return a safe fallback string instead of crashing the app.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# Initialise the Groq client lazily/safely. If the package or key is missing,
# `client` stays None and the functions below fall back gracefully.
try:
    from groq import Groq
    _api_key = os.environ.get("GROQ_API_KEY")
    client = Groq(api_key=_api_key) if _api_key else None
except Exception:
    client = None

MODEL = "llama-3.1-8b-instant"


def explain_prediction(review_text, predicted_class, fake_indicators, confidence):
    """
    Generates a 2-3 sentence, human-readable explanation of why a review was
    classified the way it was, using the Groq LLM.

    Returns a fallback message if Groq is unavailable so the app never crashes.
    """
    if client is None:
        return "We couldn't write an explanation because the Groq key isn't set up."

    prompt = f"""
    You are an expert in detecting fake product reviews.

    Review text: {review_text[:400]}

    ML Model Prediction: {predicted_class}
    Confidence: {confidence:.1f}%
    Suspicious patterns found:
    {', '.join(fake_indicators) if fake_indicators else 'None'}

    In 2-3 sentences explain to a regular user:
    1. Why this review is classified as {predicted_class}
    2. What specific patterns make it look fake (if fake)
    3. What the user should do with this information

    Be direct and simple. Plain paragraph only. No bullet points.
    """

    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=MODEL,
            max_tokens=200,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except Exception:
        # Never let an LLM error break the prediction flow.
        return "We couldn't write an explanation right now."


def generate_shopping_tips(predicted_class):
    """
    Generates exactly 3 short shopping-safety tips tailored to the verdict
    (GENUINE / FAKE / SUSPICIOUS), using the Groq LLM.

    Returns sensible static tips if Groq is unavailable.
    """
    if client is None:
        return ("- Read several reviews before trusting one\n"
                "- Check for verified purchase badges\n"
                "- Be cautious of overly generic praise")

    prompt = f"""
    Give exactly 3 short tips for a shopper who just saw
    a {predicted_class} product review.
    Format: 3 tips, one per line, starting with dash (-).
    No extra text. Max 15 words per tip.
    """

    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=MODEL,
            max_tokens=100,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return ("- Read several reviews before trusting one\n"
                "- Check for verified purchase badges\n"
                "- Be cautious of overly generic praise")
