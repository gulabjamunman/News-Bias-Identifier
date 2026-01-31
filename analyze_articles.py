import os
import requests
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
    formula = "NOT({Processed})"
    url = f"{AIRTABLE_URL}?filterByFormula={formula}"
    res = requests.get(url, headers=HEADERS)
    return res.json().get("records", [])

def analyze_article(text):
    prompt = f"""
    Analyze the following news article and return:
    1. Political leaning (Left, Right, Neutral)
    2. Overall sentiment (Positive, Negative, Neutral)
    3. Main topic in 1-3 words
    4. One-sentence explanation of framing or bias

    Article:
    {text[:4000]}
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    return response.choices[0].message.content

def update_record(record_id, analysis_text):
    data = {
        "fields": {
            "Bias Explanation": analysis_text,
            "Processed": True
        }
    }
    requests.patch(f"{AIRTABLE_URL}/{record_id}", headers=HEADERS, json=data)

articles = get_unprocessed_articles()

for article in articles:
    content = article["fields"].get("Content", "")
    if len(content) < 500:
        continue

    print("Analyzing:", article["fields"].get("Headline"))
    result = analyze_article(content)
    update_record(article["id"], result)
