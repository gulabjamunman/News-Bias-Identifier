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
WORD_PATTERN = re.compile(r"\w+", re.UNICODE)

# ---------------- TEXT CLEANING ----------------

def clean_text(text):
    text = text.encode("latin1", errors="ignore").decode("utf-8", errors="ignore")
    text = re.sub(r"\r", "\n", text)
    return text.strip()

def strip_boilerplate(text):
    cut_markers = [
        "First Published:", "Last Updated:", "Newsletter",
        "Disclaimer:", "Loading comments",
        "News18 Newsletter", "ABP Live", "Follow Us On",
        "ALSO READ", "Read More", "Advertisement"
    ]
    for marker in cut_markers:
        if marker in text:
            text = text.split(marker)[0]
    return text

def contains_devanagari(text):
    return any('\u0900' <= c <= '\u097F' for c in text)

def normalize_news_article(text):
    lines = text.split("\n")
    cleaned_lines = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if any(x in line for x in [
            "Curated By", "Updated:", "Published:", "Last Updated",
            "Follow us", "Subscribe", "Watch:", "Advertisement"
        ]):
            continue

        if line.lower().startswith(("photo", "image")):
            continue

        if "pic.twitter.com" in line:
            continue

        words = line.split()

        # keep short Hindi sentences
        if len(words) < 4:
            if contains_devanagari(line):
                cleaned_lines.append(line)
            continue

        # soften uppercase removal
        if line.isupper() and len(words) < 8:
            continue

        cleaned_lines.append(line)

    article_body = " ".join(cleaned_lines)
    return re.sub(r"\s+", " ", article_body).strip()

def is_probably_hindi(text):
    return contains_devanagari(text)

# ---------------- LEXICON CACHING ----------------

def build_lexicons():
    LM_NEGATIVE, LM_UNCERTAINTY = set(), set()

    with open("LoughranMcDonald_2016.csv", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        reader.fieldnames = [name.lower() for name in reader.fieldnames]
        for row in reader:
            word = row.get("word", "").lower()
            if word:
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
        with open(CACHE_FILE, "rb") as f:
            return pickle.load(f)
    lexicons = build_lexicons()
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(lexicons, f)
    return lexicons

LM_NEGATIVE, LM_UNCERTAINTY, EMOLEX, BWS_LEXICON = load_or_build_lexicons()

# ---------------- SCORING FUNCTIONS ----------------

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
    words = WORD_PATTERN.findall(text.lower())
    total = len(words) or 1
    neg = sum(1 for w in words if w in LM_NEGATIVE)
    unc = sum(1 for w in words if w in LM_UNCERTAINTY)
    return (neg + 1.5 * unc) / total

def emotion_profile(text):
    words = WORD_PATTERN.findall(text.lower())
    emotions = {"anger": 0, "fear": 0, "trust": 0, "joy": 0, "disgust": 0}
    for w in words:
        if w in EMOLEX:
            for emo in EMOLEX[w]:
                if emo in emotions:
                    emotions[emo] += 1
    total = sum(emotions.values()) or 1
    return {k: v / total for k, v in emotions.items()}

def bws_intensity_score(text):
    words = WORD_PATTERN.findall(text.lower())
    total = len(words) or 1
    return sum(BWS_LEXICON.get(w, 0) for w in words) / total

def threat_signal_score(text):
    emotions = emotion_profile(text)
    return emotions["anger"] + emotions["fear"] + emotions["disgust"]

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

# ---------------- FORMATTING HELPERS ----------------

def format_bias_explanation(bias):
    return (
        "FRAMING\n"
        f"{bias.get('framing_reason', '').strip()}\n\n"
        "LANGUAGE INTENSITY\n"
        f"{bias.get('intensity_reason', '').strip()}\n\n"
        "SENSATIONALISM\n"
        f"{bias.get('sensationalism_reason', '').strip()}\n\n"
        "OVERALL INTERPRETATION\n"
        f"{bias.get('overall_interpretation', '').strip()}"
    )

def format_behavioural_analysis(behaviour):
    return (
        "ATTENTION AND SALIENCE\n"
        f"{behaviour.get('attention_and_salience', '').strip()}\n\n"
        "EMOTIONAL TRIGGERS\n"
        f"{behaviour.get('emotional_triggers', '').strip()}\n\n"
        "SOCIAL AND IDENTITY CUES\n"
        f"{behaviour.get('social_and_identity_cues', '').strip()}\n\n"
        "MOTIVATION AND ACTION SIGNALS\n"
        f"{behaviour.get('motivation_and_action_signals', '').strip()}\n\n"
        "OVERALL BEHAVIOURAL INTERPRETATION\n"
        f"{behaviour.get('overall_behavioural_interpretation', '').strip()}"
    )

# ---------------- LLM ANALYSIS (UNCHANGED PROMPT) ----------------

def analyze_article(text):
    prompt = f"""
You are analyzing a news article from TWO independent perspectives:

--------------------------------------------------
PERSPECTIVE 1: POLITICAL FRAMING AND MEDIA BIAS
--------------------------------------------------
Your task is to evaluate how the article frames political reality, not whether claims are true.

Right-leaning framing includes language or narratives that:
- Emphasize nationalism, national security, border control, patriotism
- Support conservative, market-oriented, or law-and-order positions
- Defend right-leaning parties or criticize progressive/left parties
- Frame issues around sovereignty, cultural identity, internal or external threats

Left-leaning framing includes language or narratives that:
- Emphasize social justice, welfare, minority rights, equality
- Criticize conservative or nationalist politics from a progressive lens
- Focus on redistribution, civil liberties, structural inequality
- Frame issues around marginalization, discrimination, or corporate power

If an article mainly reports statements from a political actor, use the direction of that actor’s framing.

Score:
- framing_direction from -1 (strongly left-framed) to +1 (strongly right-framed)
- language_intensity from 0 (calm, factual) to 1 (highly emotional or charged)
- sensationalism_score from 0 (restrained reporting) to 1 (dramatic or exaggerated)

--------------------------------------------------
PERSPECTIVE 2: BEHAVIOURAL INFLUENCE ON READERS
--------------------------------------------------
Now analyze psychological influence patterns in the article.

Explain:
- What the article makes salient or attention-grabbing
- What emotions it may evoke (fear, anger, pride, hope, outrage, empathy)
- Whether it invokes group identity or "us vs them" cues
- Whether it creates urgency, blame, reassurance, pride, or concern

You are not judging correctness. You are identifying influence patterns in language and framing.

--------------------------------------------------
OUTPUT FORMAT
--------------------------------------------------

Return ONLY valid JSON with this structure:

{
  "framing_direction": number,
  "language_intensity": number,
  "sensationalism_score": number,
  "topic": "1-3 word topic label",

  "bias_explanation": {
      "framing_reason": "",
      "intensity_reason": "",
      "sensationalism_reason": "",
      "overall_interpretation": ""
  },

  "behavioural_analysis": {
      "attention_and_salience": "",
      "emotional_triggers": "",
      "social_and_identity_cues": "",
      "motivation_and_action_signals": "",
      "overall_behavioural_interpretation": ""
  }
}

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
    records, offset = [], None
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
    requests.patch(f"{AIRTABLE_URL}/{record_id}", headers=HEADERS, json={"fields": fields})

# ---------------- MAIN ----------------

def main():
    articles = get_unprocessed_articles()

    for article in articles:
        headline = article["fields"].get("Headline", "Untitled")
        raw_content = article["fields"].get("Content", "")
        content = normalize_news_article(strip_boilerplate(clean_text(raw_content)))

        word_count = len(WORD_PATTERN.findall(content))
        char_count = len(content)

        if word_count < 40 and char_count < 250:
            continue

        try:
            analysis = analyze_article(content)

            framing = analysis["framing_direction"]
            intensity = analysis["language_intensity"]
            sensational = analysis["sensationalism_score"]

            ai_threat = threat_signal_score(content)
            ai_lex_intensity = bws_intensity_score(content)

            sentiment_label = "Neutral" if is_probably_hindi(content) else derive_sentiment_label(content)
            econ_score = 0 if is_probably_hindi(content) else economic_risk_score(content)

            composite_score = compute_composite_ideology(framing, intensity, content)
            political_leaning = derive_political_leaning(framing, econ_score)

            update_record(article["id"], {
                "Composite Ideology Score": composite_score,
                "Political Leaning": political_leaning,
                "Sentiment": sentiment_label,
                "Topic": analysis["topic"],
                "Bias Explanation": format_bias_explanation(analysis["bias_explanation"]),
                "Behavioural Analysis": format_behavioural_analysis(analysis["behavioural_analysis"]),
                "Processed": True,
                "AI Framing Direction": framing,
                "AI Language Intensity": intensity,
                "AI Sensationalism": sensational,
                "AI Threat Signal": ai_threat,
                "AI Lexical Emotional Intensity": ai_lex_intensity
            })

            print(f"Processed: {headline}")

        except Exception as e:
            print(f"Failed: {headline}", e)

if __name__ == "__main__":
    main()
