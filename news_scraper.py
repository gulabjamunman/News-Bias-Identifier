import feedparser
from newspaper import Article
from datetime import datetime, timedelta, timezone
from dateutil import parser as dateparser
import requests
import time
import os
import urllib.parse
# ==============================
# RSS FEEDS
# ==============================
RSS_FEEDS = {
    "Times of India": "https://timesofindia.indiatimes.com/rssfeeds/-2128936835.cms",
    "Hindustan Times": "https://www.hindustantimes.com/feeds/rss/india-news/rssfeed.xml",
    "News18": "https://www.news18.com/commonfeeds/v1/eng/rss/india.xml",
    "NDTV": "https://feeds.feedburner.com/ndtvnews-india-news",
    "Indian Express": "https://indianexpress.com/section/india/feed",
    "The Hindu": "https://frontline.thehindu.com/the-nation/feeder/default.rss",
    "FirstPost India": "https://www.firstpost.com/commonfeeds/v1/mfp/rss/india.xml",
    "ABP India": "https://www.abplive.com/news/india/feed",
    "Rising Kashmir": "https://risingkashmir.com/feed/",
    "Storify News": "https://www.storifynews.com/feed/",
    "DNA India": "https://www.dnaindia.com/feeds/india.xml",
    "India Today": "https://www.indiatoday.in/rss/1206578",
}

# ==============================
# AIRTABLE CONFIG
# ==============================
import os
AIRTABLE_TOKEN = os.getenv("AIRTABLE_TOKEN")
print("Token loaded:", AIRTABLE_TOKEN is not None)

AIRTABLE_BASE_ID = "appNakTUaXtBXu8Vs"
AIRTABLE_TABLE_NAME = "Data1"

AIRTABLE_URL = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{AIRTABLE_TABLE_NAME}"
HEADERS = {
    "Authorization": f"Bearer {AIRTABLE_TOKEN}",
    "Content-Type": "application/json"
}

print("Headers ready")

# ==============================
# TIME FILTER
# ==============================
NOW = datetime.now(timezone.utc)
THREE_HOURS_AGO = NOW - timedelta(hours=1)

# ==============================
# FUNCTIONS
# ==============================
from readability import Document
from bs4 import BeautifulSoup
import requests

def extract_article_text(url):
    try:
        # First attempt: newspaper3k
        article = Article(url)
        article.download()
        article.parse()
        text = article.text.strip()

        # If newspaper3k failed (very short text), use fallback
        if len(text) < 500:
            print("Fallback extractor used for:", url)
            response = requests.get(url, timeout=10)
            doc = Document(response.text)
            html = doc.summary()
            soup = BeautifulSoup(html, "html.parser")

            # Remove scripts/styles
            for tag in soup(["script", "style", "aside", "header", "footer", "nav"]):
                tag.decompose()

            text = "\n".join(p.get_text() for p in soup.find_all("p"))

        return text.strip(), article.authors

    except Exception as e:
        print(f"Failed to parse article: {url} | Error: {e}")
        return None, []


def sanitize_record(data):
    cleaned = {}
    for k, v in data.items():
        if isinstance(v, str):
            cleaned[k] = v.replace('"', '').replace("'", "").strip()
        else:
            cleaned[k] = v
    return cleaned

def url_exists(article_url):
    safe_url = article_url.replace("'", "\\'")
    formula = f"{{URL}} = '{safe_url}'"
    encoded_formula = urllib.parse.quote(formula)
    lookup_url = f"{AIRTABLE_URL}?filterByFormula={encoded_formula}"

    response = requests.get(lookup_url, headers=HEADERS)
    if response.status_code != 200:
        print("Airtable lookup error:", response.text)
        return False

    records = response.json().get("records", [])
    return len(records) > 0

def push_to_airtable(data):
    data = sanitize_record(data)  # FINAL CLEANING STEP
    response = requests.post(AIRTABLE_URL, headers=HEADERS, json={"fields": data})
    if response.status_code != 200:
        print("Airtable error:", response.text)
    else:
        print("Uploaded:", data["Headline"])

# ==============================
# MAIN LOOP
# ==============================
for publisher, feed_url in RSS_FEEDS.items():
    print(f"\nChecking {publisher}")
    feed = feedparser.parse(feed_url)

    recent_articles = []

    for entry in feed.entries:
        if not hasattr(entry, "published"):
            continue

        try:
            published_time = dateparser.parse(entry.published).astimezone(timezone.utc)
        except:
            continue

        if published_time >= THREE_HOURS_AGO:
            recent_articles.append((published_time, entry))

    # Sort newest first and take top 3
    recent_articles.sort(reverse=True, key=lambda x: x[0])
    top_three = recent_articles[:3]

    for pub_time, entry in top_three:
        url = entry.link
        headline = entry.title

        print(f"Scraping: {headline}")

        content, authors = extract_article_text(url)
        if not content:
            continue

        record = {
            "Author": ", ".join(authors) if authors else "",
            "Publisher Name": publisher,
            "Publication Date & Time": pub_time.isoformat(),
            "Headline": headline,
            "Content": content[:100000],
            "URL": url
        }

        if url_exists(record["URL"]):
           print("Duplicate skipped (URL exists):", record["URL"])
        else:
           push_to_airtable(record)
           
        time.sleep(1)
