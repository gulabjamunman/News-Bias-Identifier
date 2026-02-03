import os
import requests
import json
from openai import OpenAI
import re
import csv
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

AIRTABLE_TOKEN = os.getenv("AIRTABLE_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

BASE_ID = "appNakTUaXtBXu8Vs"
TABLE_NAME = "Data1"

AIRTABLE_URL = f"https://api.airtable.com/v0/{BASE_ID}/{TABLE_NAME}"
HEADERS = {
    "Authorization": f"Bearer {AIRTABLE_TOKEN}",
    "Content-Type": "application/json"
}

client = OpenAI(api_key=OPENAI_API_KEY)

# ---------------- LEXICON LOADS ----------------

SHAPIRO_LEXICON = {}
with open("shapiro_lexicon.csv", newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        SHAPIRO_LEXICON[row["word"].lower()] = float(row["score"])

EMOLEX = {}
with open("NRC-Emotion-Lexicon-Wordlevel-v0.92.txt", encoding="utf-8") as f:
    for line in f:
        word, emotion, flag = line.strip().split("\t")
        if int(flag) == 1:
            EMOLEX.setdefault(word, set()).add(emotion)

BWS_LEXICON = {}
with open("bws_emotion_lexicon.csv", newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        BWS_LEXICON[row["word"].lower()] = float(row["intensity"])

# ---------------- HELPER FUNCTIONS ----------------

def vader_emotional_score(text):
    return sia.polarity_scores(text)["compound"]

def derive_sentiment_label(text):
    score = vader_emotional_score(text)
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    return "Neutral"

def shapiro_economic_score(text):
    words = re.findall(r"\b[a-zA-Z]+\b", text.lower())
    total = len(words) or 1
    score = sum(SHAPIRO_LEXICON.get(w, 0) for w in words)
    return score / total

def emotion_profile(text):
    words = re.findall(r"\b[a-zA-Z]+\b", text.lower())
    emotions = {"anger":0,"fear":0,"trust":0,"joy":0,"disgust":0}
    for w in words:
        if w in EMOLEX:
            for emo in EMOLEX[w]:
                if emo in emotions:
                    emotions[emo] += 1
    total = sum(emotions.values()) or 1
    return {k: v/total for k, v in emotions.items()}

def bws_intensity_score(text):
    words = re.findall(r"\b[a-zA-Z]+\b", text.lower())
    total = len(words) or 1
    return sum(BWS_LEXICON.get(w, 0) for w in words) / total

def emotional_multiplier(vader_score):
    return 1 + abs(vader_score) if vader_score < 0 else 1

def economic_multiplier(shapiro_score):
    return 1 + abs(shapiro_score)*1.5 if shapiro_score < 0 else 1

def threat_multiplier(emotions):
    return 1 + (emotions["anger"] + emotions["fear"] + emotions["disgust"])*2 - emotions["trust"]

def compute_composite_ideology(framing, intensity, sensationalism, text):
    vader_score = vader_emotional_score(text)
    shapiro_score = shapiro_economic_score(text)
    emotions = emotion_profile(text)
    bws_score = bws_intensity_score(text)

    base = framing * (0.6 + 0.4 * intensity)

    return base * emotional_multiplier(vader_score) * economic_multiplier(shapiro_score) * threat_multiplier(emotions) * (1 + bws_score)

def derive_political_leaning(framing, shapiro_score):
    adjusted = framing + (shapiro_score * 0.5)
    if adjusted > 0.2:
        return "Right"
    elif adjusted < -0.2:
        return "Left"
    return "Neutral"

# ---------------- FETCH ARTICLES ----------------

def get_unprocessed_articles():
    print("Fetching records from Airtable...")
    res = requests.get(AIRTABLE_URL, headers=HEADERS)
    print("Status:", res.status_code)

    data = res.json()
    records = data.get("records", [])

    print(f"Total records in table: {len(records)}")

    unprocessed = [r for r in records if not r["fields"].get("Processed")]
    print(f"Unprocessed records found: {len(unprocessed)}")
    return unprocessed

# ---------------- OPENAI ANALYSIS ----------------

def analyze_article(text):
    prompt = f"""
You are analyzing a news article for research purposes.

Return JSON with:
{{
  "framing_direction": number from -1 (left) to +1 (right),
  "language_intensity": number from 0 (neutral tone) to 1 (highly value-laden),
  "sensationalism_score": number from 0 (not sensational) to 1 (very dramatic),
  "topic": "1-3 word topic label",
  "bias_explanation": "One concise sentence explaining framing or bias"
}}

Article:
{text[:4000]}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        response_format={"type": "json_object"}
    )

    return json.loads(response.choices[0].message.content)

# ---------------- UPDATE AIRTABLE ----------------

def update_record(record_id, composite_score, political_leaning, sentiment_label, topic, bias_explanation):
    data = {
        "fields": {
            "Composite Ideology Score": composite_score,
            "Political Leaning": political_leaning,
            "Sentiment": sentiment_label,
            "Topic": topic,
            "Bias Explanation": bias_explanation,
            "Processed": True
        }
    }

    response = requests.patch(f"{AIRTABLE_URL}/{record_id}", headers=HEADERS, json=data)
    if response.status_code != 200:
        print("Airtable update error:", response.text)

# ---------------- MAIN LOOP ----------------

articles = get_unprocessed_articles()

if not articles:
    print("No articles to analyze after filtering.")
else:
    print("Starting analysis loop...")

for article in articles:
    headline = article["fields"].get("Headline")
    content = article["fields"].get("Content", "")

    print("\nProcessing article:", headline)
    print("Content length:", len(content))

    if len(content) < 250:
        print("Skipping — content too short")
        continue

    print("Sending to OpenAI...")
    analysis = analyze_article(content)
    print("Model response received")

    framing = analysis["framing_direction"]
    intensity = analysis["language_intensity"]
    sensationalism = analysis["sensationalism_score"]

    composite_score = compute_composite_ideology(framing, intensity, sensationalism, content)
    sentiment_label = derive_sentiment_label(content)
    political_leaning = derive_political_leaning(framing, shapiro_economic_score(content))

    update_record(
        article["id"],
        composite_score,
        political_leaning,
        sentiment_label,
        analysis.get("topic"),
        analysis.get("bias_explanation")
    )

    print("Airtable updated\n")
