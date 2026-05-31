"""
make_starter_dataset.py
=======================
Generates a STARTER labelled training set (data/reviews.csv) so ReviewGuard
has a working model before you download the real Kaggle dataset.

It builds realistic GENUINE and FAKE reviews from templates:
  - GENUINE: specific, mention concrete details, mixed sentiment, real gripes.
  - FAKE:    generic gushing praise, exclamation spam, urgency/bot phrases,
             "received for free" disclosures — the patterns paid/bot reviews use.

This is a stand-in for the Kaggle "Amazon Fake Reviews" dataset. Replace
data/reviews.csv with the real one and re-run train_model.py for best results.
"""

import os
import random

import pandas as pd

random.seed(42)
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(THIS_DIR, "..", "data", "reviews.csv")

# ---- building blocks ----
products = ["headphones", "phone case", "blender", "charger", "backpack",
            "shoes", "watch", "lamp", "keyboard", "water bottle", "mirror sticker",
            "earbuds", "power bank", "t-shirt", "sunglasses", "kettle"]

genuine_templates = [
    "The {p} works well but the battery drains faster than advertised.",
    "Decent {p} for the price, though the build feels a bit cheap.",
    "Used the {p} for two weeks now, holds up fine during daily use.",
    "The {p} arrived on time but the color was slightly different from the photo.",
    "Good {p}, however the instructions were confusing to follow.",
    "I returned the {p} because the {p} stopped working after a few days.",
    "Solid {p}. Not premium quality but does the job for everyday tasks.",
    "The {p} is comfortable but gets warm after about an hour of use.",
    "Shipping took longer than expected, the {p} itself is okay.",
    "The {p} sticks well on glass but not on painted walls, just so you know.",
    "Average {p}. The zipper broke after a month, a little disappointing.",
    "Bought this {p} for my brother, he says the sound is a bit weak.",
    "The {p} does what it says, setup took around ten minutes.",
    "Quality is fine for the price but the {p} scratches easily.",
    "It works but the {p} is smaller than I imagined from the listing.",
]

fake_templates = [
    "Amazing!!!! best product ever, highly recommend, five stars!!!",
    "Absolutely amazing perfect product, works perfectly, love it so much!",
    "Best {p} ever!!! must buy, do not hesitate, worth every penny!!!",
    "This {p} is very good, i am very satisfied, i like this product.",
    "Buy now everyone should buy this {p}, value for money, five stars!",
    "I received this for free but honestly this {p} is very good, amazing!",
    "Perfect product!!! great quality, highly recommend, works perfectly!!!",
    "Good product good {p} good quality good price, i like this product.",
    "Wow amazing {p}!!! love it so much, best product ever, must buy now!",
    "Got this for review, absolutely amazing, perfect {p}, five stars!!!",
    "Highly recommend!!! this {p} is the best ever, worth every penny!!!",
    "Super duper amazing {p}, love it, love it, love it, best ever!!!!",
    "Excellent excellent excellent {p}, perfect product, must buy, amazing!",
    "I am very satisfied, this product is good, great quality, five stars!!",
    "Best purchase ever!!! everyone should buy this {p}, do not hesitate!!!",
]

rows = []
for _ in range(450):
    p = random.choice(products)
    rows.append({"text": random.choice(genuine_templates).format(p=p), "label": "OR"})
    rows.append({"text": random.choice(fake_templates).format(p=p), "label": "CG"})

os.makedirs(os.path.dirname(OUT), exist_ok=True)
pd.DataFrame(rows).sample(frac=1, random_state=42).to_csv(OUT, index=False)
print(f"Wrote {len(rows)} reviews to {OUT}")
print("Now run: python model/train_model.py")
