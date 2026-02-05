import feedparser
from newspaper import Article
from datetime import datetime, timedelta, timezone
from dateutil import parser as dateparser
import requests
import time
import os
import urllib.parse

# ---------------- RSS FEEDS ----------------
RSS_FEEDS = {
    "News18": "https://www.news18.com/commonfeeds/v1/eng/rss/india.xml",
    "ABP India": "https://www.abplive.com/news/india/feed",
    "Indian Express": "https://indianexpress.com/section/india/feed",
    "The Hindu": "https://frontline.thehindu.com/the-nation/feeder/default.rss"
}

# ---------------- AIRTABLE CONFIG ----------------
AIRTABLE_TOKEN = os.getenv("AIRTABLE_TOKEN")
BASE_ID = "appNakTUaXtBXu8Vs"
TABLE_NAME = "Data1"

AIRTABLE_URL = f"https://api.airtable.com/v0/{BASE_ID}/{TABLE_NAME}"
HEADERS = {
    "Authorization": f"Bearer {AIRTABLE_TOKEN}",
    "Content-Type": "application/json"
}

# ---------------- HELPERS ----------------

def extract_raw_text(url):
    try:
        article = Article(url, request_timeout=10)
        article.download()
        article.parse()
        return article.text.strip(), article.authors
    except:
        return None, []

def url_exists(article_url):
    formula = f"{{URL}} = '{article_url.replace('\'', '\\\'')}'"
    lookup_url = f"{AIRTABLE_URL}?filterByFormula={urllib.parse.quote(formula)}"
    response = requests.get(lookup_url, headers=HEADERS)
    if response.status_code != 200:
        return False
    return len(response.json().get("records", [])) > 0

def push_to_airtable(data):
    requests.post(AIRTABLE_URL, headers=HEADERS, json={"fields": data})

# ---------------- MAIN ----------------

NOW = datetime.now(timezone.utc)
WINDOW = NOW - timedelta(hours=6)

for publisher, feed_url in RSS_FEEDS.items():
    feed = feedparser.parse(feed_url)

    for entry in feed.entries:
        published_raw = getattr(entry, "published", None) or getattr(entry, "updated", None)
        if not published_raw:
            continue

        try:
            pub_time = dateparser.parse(published_raw).astimezone(timezone.utc)
        except:
            continue

        if pub_time < WINDOW:
            continue

        url = entry.link
        if url_exists(url):
            continue

        content, authors = extract_raw_text(url)
        if not content:
            continue

        record = {
            "Author": ", ".join(authors),
            "Publisher Name": publisher,
            "Publication Date & Time": pub_time.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
            "Headline": entry.title,
            "Content": content[:100000],
            "URL": url,
            "Processed": False
        }

        push_to_airtable(record)
        time.sleep(1)
