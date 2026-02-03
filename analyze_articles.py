import os
import requests
import json
import re
import csv
import pickle
import nltk
from openai import OpenAI
from nltk.sentiment import SentimentIntensityAnalyzer

# ---------------- SETUP ----------------

nltk.download("vader_lexicon", quiet=True)
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

CACHE_FILE = "lexicon_cache.pkl"

# ---------------- TEXT CLEANING ----------------

def clean_text(text):
    text = text.encode("latin1", errors="ignore").decode("utf-8", errors="ignore")
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def strip_boilerplate(text):
    cut_markers = [
        "First Published:",
        "Last Updated:",
        "Newsletter",
        "Disclaimer:",
        "Loading comments"
    ]
    for marker in cut_markers:
        if marker in text:
            text = text.split(marker)[0]
    return text

# ---------------- LEXICON CACHE SYSTEM ----------------

def build_lexicons():
    print("Building lexicons from source files...")

    LM_NEGATIVE = set()
    LM_UNCERTAINTY = set()

    with open("LoughranMcDonald_2016.csv", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        reader.fieldnames = [name.lower() for name in reader.fieldnames]

        for row in reader:
            word = row.get("word", "").lower()
            if not word:
                continue
            if row.get("negative") and int(row["negative"]) > 0:
                LM_NEGATIVE.add(word)
            if row.get("uncertainty") and int(row["uncertainty"]) > 0:
                LM_UNCERTAINTY.add(word)

    EMOLEX = {}

    def load_emolex(path):
        with open(path, encoding="utf-8") as f:
            header = f.readline().strip().split("\t")
            if len(header) == 3:
                f.seek(0)
                for line in f:
                    word, emotion, flag = line.strip().split("\t")
                    if flag == "1":
                        EMOLEX.setdefault(word.lower(), set()).add(emotion)
            else:
                emotions = header[1:]
                for line in f:
                    parts = line.strip().split("\t")
                    word = parts[0].lower()
                    for emotion, flag in zip(emotions, parts[1:]):
                        if flag == "1":
                            EMOLEX.setdefault(word, set()).add(emotion)

    load_emolex("NRC-Emotion-Lexicon-Wordlevel-v0.92.txt")
    load_emolex("Hindi-NRC-EmoLex.txt")

    BWS_LEXICON = {}
    with open("NRC-Emotion-Intensity-Lexicon-v1.txt", encoding="utf-8") as f:
        for line in f:
            word, emotion, score = line.strip().split("\t")
            BWS_LEXICON[word.lower()] = float(score)

    return LM_NEGATIVE, LM_UNCERTAINTY, EMOLEX, BWS_LEXICON


def load_or_build_lexicons():
    if os.path.exists(CACHE_FILE):
        print("Loading lexicons from cache...")
        with open(CACHE_FILE, "rb") as f:
            return pickle.load(f)

    lexicons = build_lexicons()
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(lexicons, f)

    print("Lexicons cached.")
    return lexicons


LM_NEGATIVE, LM_UNCERTAINTY, EMOLEX, BWS_LEXICON = load_or_build_lexicons()

# ---------------- ANALYSIS FUNCTIONS ----------------

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
    emotions = {"anger":0, "fear":0, "trust":0, "joy":0, "disgust":0}
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

def compute_composite_ideology(framing, intensity, text):
    vader_score = vader_emotional_score(text)
    econ_score = economic_risk_score(text)
    emotions = emotion_profile(text)
    bws_score = bws_intensity_score(text)

    base = framing * (0.6 + 0.4 * intensity)
    emotional_mult = 1 + abs(vader_score) if vader_score < 0 else 1
    economic_mult = 1 + econ_score * 5
    threat_mult = 1 + (emotions["anger"] + emotions["fear"] + emotions["disgust"]) * 2 - emotions["trust"]

    return base * emotional_mult * economic_mult * threat_mult * (1 + bws_score)

def derive_political_leaning(framing, econ_score):
    adjusted = framing + (econ_score * 0.5)
    if adjusted > 0.2:
        return "Right"
    elif adjusted < -0.2:
        return "Left"
    return "Neutral"

# ---------------- OPENAI ----------------

def analyze_article(text):
    prompt = f"""
Return JSON with:
{{
  "framing_direction": number from -1 to +1,
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

# ---------------- AIRTABLE ----------------

def get_unprocessed_articles():
    records = []
    offset = None

    while True:
        params = {"offset": offset} if offset else {}
        res = requests.get(AIRTABLE_URL, headers=HEADERS, params=params)
        data = res.json()

        records.extend(data.get("records", []))
        offset = data.get("offset")
        if not offset:
            break

    return [r for r in records if not r["fields"].get("Processed")]

def update_record(record_id, fields):
    response = requests.patch(
        f"{AIRTABLE_URL}/{record_id}",
        headers=HEADERS,
        json={"fields": fields}
    )
    if response.status_code != 200:
        print("Airtable update error:", response.text)

# ---------------- MAIN ----------------

def main():
    articles = get_unprocessed_articles()

    if not articles:
        print("No articles to analyze.")
        return

    for article in articles:
        headline = article["fields"].get("Headline", "Untitled")
        content = article["fields"].get("Content", "")

        content = clean_text(content)
        content = strip_boilerplate(content)

        print(f"\nProcessing: {headline}")

        words = re.findall(r"\b[\w']+\b", content)
        if len(words) < 40:
            print("Skipped, too short after cleaning.")
            continue

        try:
            analysis = analyze_article(content)

            framing = analysis["framing_direction"]
            intensity = analysis["language_intensity"]

            composite_score = compute_composite_ideology(framing, intensity, content)
            sentiment_label = derive_sentiment_label(content)
            econ_score = economic_risk_score(content)
            political_leaning = derive_political_leaning(framing, econ_score)

            update_record(article["id"], {
                "Composite Ideology Score": composite_score,
                "Political Leaning": political_leaning,
                "Sentiment": sentiment_label,
                "Topic": analysis["topic"],
                "Bias Explanation": analysis["bias_explanation"],
                "Processed": True
            })

            print("Success")

        except Exception as e:
            print(f"Failed: {headline}")
            print("Reason:", e)
            print("Preview:", content[:300])


if __name__ == "__main__":
    main()
