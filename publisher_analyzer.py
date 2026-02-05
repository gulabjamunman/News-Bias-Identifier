import os
import requests
import re
from collections import defaultdict
publisher_stats = defaultdict(lambda: {"seen": 0, "processed": 0, "skipped": 0})
from your_existing_analyzer_filename import (
    analyze_article,
    threat_signal_score,
    bws_intensity_score,
    derive_sentiment_label,
    economic_risk_score,
    compute_composite_ideology,
    derive_political_leaning,
    format_bias_explanation,
    format_behavioural_analysis,
    is_probably_hindi,
    WORD_PATTERN
)

AIRTABLE_TOKEN = os.getenv("AIRTABLE_TOKEN")
BASE_ID = "appNakTUaXtBXu8Vs"
TABLE_NAME = "Data1"

AIRTABLE_URL = f"https://api.airtable.com/v0/{BASE_ID}/{TABLE_NAME}"
HEADERS = {"Authorization": f"Bearer {AIRTABLE_TOKEN}", "Content-Type": "application/json"}

# ---------------- PUBLISHER CLEANERS ----------------

def clean_generic(text):
    return re.sub(r"\s+", " ", text).strip()

def clean_live_style(text):
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if "Updated:" in line or "LIVE" in line:
            continue
        if len(line.split()) < 3:
            continue
        cleaned.append(line)
    return " ".join(cleaned)

def clean_hindi_shortform(text):
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if "विज्ञापन" in line:
            continue
        cleaned.append(line)
    return " ".join(cleaned)

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
        publisher = article["fields"].get("Publisher Name", "")
        publisher_stats[publisher]["seen"] += 1
        raw_content = article["fields"].get("Content", "")

        if publisher in ["News18", "ABP India"]:
            content = clean_live_style(raw_content)
        elif any('\u0900' <= c <= '\u097F' for c in raw_content):
            content = clean_hindi_shortform(raw_content)
        else:
            content = clean_generic(raw_content)

        word_count = len(WORD_PATTERN.findall(content))
        char_count = len(content)

        if word_count < 40 and char_count < 250:
            publisher_stats[publisher]["skipped"] += 1
            continue

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
        publisher_stats[publisher]["processed"] += 1
        print("Processed:", headline)

if __name__ == "__main__":
    main()

    print("\nProcessing Summary")
    print("------------------")
    for pub, stats in publisher_stats.items():
        print(f"{pub:15} Seen: {stats['seen']:3}  Processed: {stats['processed']:3}  Skipped: {stats['skipped']:3}")
