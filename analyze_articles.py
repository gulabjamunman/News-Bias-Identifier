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

# Loughran–McDonald Financial Sentiment
LM_NEGATIVE = set()
LM_UNCERTAINTY = set()

with open("LoughranMcDonald_2016.csv", newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)

    # Normalize column names to lowercase
    reader.fieldnames = [name.lower() for name in reader.fieldnames]

    for row in reader:
        row = {k.lower(): v for k, v in row.items()}

        word = row.get("word")
        if not word:
            continue

        word = word.lower()

        if row.get("negative") and int(row["negative"]) > 0:
            LM_NEGATIVE.add(word)

        if row.get("uncertainty") and int(row["uncertainty"]) > 0:
            LM_UNCERTAINTY.add(word)

# NRC Emotion Lexicons (English + Hindi)
EMOLEX = {}

def load_emolex(path):
    with open(path, encoding="utf-8") as f:
        header = f.readline().strip().split("\t")

        # Case 1: Word–Emotion–Flag format (English style)
        if len(header) == 3:
            f.seek(0)
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) != 3:
                    continue
                word, emotion, flag = parts
                if flag == "1":
                    EMOLEX.setdefault(word.lower(), set()).add(emotion)

        # Case 2: Matrix format (Hindi style)
        else:
            emotions = header[1:]  # skip the word column
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) != len(header):
                    continue
                word = parts[0].lower()
                flags = parts[1:]

                for emotion, flag in zip(emotions, flags):
                    if flag == "1":
                        EMOLEX.setdefault(word, set()).add(emotion)

# NRC Emotion Intensity (Best-Worst Scaling)
BWS_LEXICON = {}
with open("NRC-Emotion-Intensity-Lexicon-v1.txt", encoding="utf-8") as f:
    for line in f:
        word, emotion, score = line.strip().split("\t")
        BWS_LEXICON[word.lower()] = float(score)

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

def economic_risk_score(text):
    words = re.findall(r"\b[\w']+\b", text.lower())
    total = len(words) or 1
    neg = sum(1 for w in words if w in LM_NEGATIVE)
    unc = sum(1 for w in words if w in LM_UNCERTAINTY)
    return (neg + 1.5 * unc) / total

def emotion_profile(text):
    words = re.findall(r"\b[\w']+\b", text.lower())
    emotions = {"anger":0,"fear":0,"trust":0,"joy":0,"disgust":0}
    for w in words:
        if w in EMOLEX:
            for emo in EMOLEX[w]:
                if emo in emotions:
                    emotions[emo] += 1
    total = sum(emotions.values()) or 1
    return {k: v/total for k, v in emotions.items()}

def bws_intensity_score(text):
    words = re.findall(r"\b[\w']+\b", text.lower())
    total = len(words) or 1
    return sum(BWS_LEXICON.get(w, 0) for w in words) / total

def emotional_multiplier(vader_score):
    return 1 + abs(vader_score) if vader_score < 0 else 1

def economic_multiplier(econ_score):
    return 1 + econ_score * 5

def threat_multiplier(emotions):
    return 1 + (emotions["anger"] + emotions["fear"] + emotions["disgust"])*2 - emotions["trust"]

def compute_composite_ideology(framing, intensity, sensationalism, text):
    vader_score = vader_emotional_score(text)
    econ_score = economic_risk_score(text)
    emotions = emotion_profile(text)
    bws_score = bws_intensity_score(text)

    base = framing * (0.6 + 0.4 * intensity)

    return base * emotional_multiplier(vader_score) * economic_multiplier(econ_score) * threat_multiplier(emotions) * (1 + bws_score)

def derive_political_leaning(framing, econ_score):
    adjusted = framing + (econ_score * 0.5)
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
    unprocessed = [r for r in records if not r["fields"].get("Processed")]
    print(f"Unprocessed records found: {len(unprocessed)}")
    return unprocessed

# ---------------- OPENAI ANALYSIS ----------------

def analyze_article(text):
    prompt = f"""
Return JSON with:
{{
  "framing_direction": number from -1 (left) to +1 (right),
  "language_intensity": number from 0 to 1,
  "sensationalism_score": number from 0 to 1,
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
    print("No articles to analyze.")
else:
    print("Starting analysis loop...")

for article in articles:
    headline = article["fields"].get("Headline")
    content = article["fields"].get("Content", "")

    print("\nProcessing article:", headline)

    if len(content) < 250:
        print("Skipping — content too short")
        continue

    analysis = analyze_article(content)

    framing = analysis["framing_direction"]
    intensity = analysis["language_intensity"]
    sensationalism = analysis["sensationalism_score"]

    composite_score = compute_composite_ideology(framing, intensity, sensationalism, content)
    sentiment_label = derive_sentiment_label(content)
    econ_score = economic_risk_score(content)
    political_leaning = derive_political_leaning(framing, econ_score)

    update_record(
        article["id"],
        composite_score,
        political_leaning,
        sentiment_label,
        analysis.get("topic"),
        analysis.get("bias_explanation")
    )

    print("Airtable updated")
