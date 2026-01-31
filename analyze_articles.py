import os
import requests
import json
from openai import OpenAI

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

def get_unprocessed_articles():
    formula = "{Processed} = FALSE()"
    encoded_formula = requests.utils.quote(formula)
    url = f"{AIRTABLE_URL}?filterByFormula={encoded_formula}"
    
    res = requests.get(url, headers=HEADERS)
    data = res.json()
    articles = get_unprocessed_articles()
    if not articles:
        print("No unprocessed articles found. Check field name or table.")
    else:
        print(f"Analyzing {len(articles)} articles...")
        
    print("Airtable response status:", res.status_code)
    print("Records returned:", len(data.get("records", [])))
    
    return data.get("records", [])

def analyze_article(text):
    prompt = f"""
You are analyzing a news article for research purposes.

Return your answer ONLY as valid JSON with this structure:

{{
  "ideology_score": number from -1 (left) to 1 (right),
  "political_leaning": "Left" | "Right" | "Neutral",
  "sentiment": "Positive" | "Negative" | "Neutral",
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
        response_format={"type": "json_object"}  # forces JSON output
    )
    
def update_record(record_id, analysis):
    data = {
        "fields": {
            "Ideology Score": analysis.get("ideology_score"),
            "Political Leaning": analysis.get("political_leaning"),
            "Sentiment": analysis.get("sentiment"),
            "Topic": analysis.get("topic"),
            "Bias Explanation": analysis.get("bias_explanation"),
            "Processed": True
        }
    }

    response = requests.patch(f"{AIRTABLE_URL}/{record_id}", headers=HEADERS, json=data)
    if response.status_code != 200:
        print("Airtable update error:", response.text)

    analysis= analyze_articles(content)
    update_record(article["id"], analysis)
