import os
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from supabase import create_client

load_dotenv(dotenv_path=".env", override=True)

# ---------------- Supabase Setup ----------------

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

print("Supabase URL:", SUPABASE_URL[:30], "...")
print("Supabase key loaded:", bool(SUPABASE_KEY))

# ---------------- App Setup ----------------

app = Flask(__name__, static_folder=".")
sessions = {}

# ---------------- Serve Frontend ----------------

@app.route("/")
def home():
    return app.send_static_file("index.html")

# ---------------- Supabase Helpers ----------------

def validate_reviewer(rid):
    res = supabase.table("reviewers") \
        .select("id, active") \
        .eq("id", rid) \
        .eq("active", True) \
        .execute()

    return bool(res.data)


def get_next_article():
    """
    Fetch one active article from review_articles
    that still exists in articles.
    """
    res = supabase.table("review_articles") \
        .select("article_id, articles(id, headline, content)") \
        .eq("active", True) \
        .limit(1) \
        .execute()

    if not res.data:
        return None

    return res.data[0]["articles"]


def save_review(data):
    supabase.table("human_reviews").insert(data).execute()

# ---------------- Chat Logic ----------------

@app.route("/chat", methods=["POST"])
def chat():
    user = request.json["user_id"]
    msg = request.json["message"]

    if user not in sessions:
        sessions[user] = {"stage": "ask_id"}

    s = sessions[user]

    # ---------- ASK REVIEWER ID ----------
    if s["stage"] == "ask_id":
        if validate_reviewer(msg):
            s["reviewer_id"] = msg
            article = get_next_article()

            if not article:
                return jsonify({"reply": "No articles left to review. Thank you!"})

            s["article"] = article
            s["responses"] = {}
            s["stage"] = "ask_political"

            return jsonify({
                "reply": (
                    f"Headline: {article['headline']}\n\n"
                    f"{article['content']}\n\n"
                    "On a scale of 1–5, how politically left/right did this feel?"
                )
            })

        return jsonify({"reply": "Invalid ID. Try again."})

    # ---------- RATINGS FLOW ----------
    elif s["stage"] == "ask_political":
        s["responses"]["political"] = int(msg)
        s["stage"] = "ask_intensity"
        return jsonify({"reply": "How emotionally intense was the language? (1–5)"})

    elif s["stage"] == "ask_intensity":
        s["responses"]["intensity"] = int(msg)
        s["stage"] = "ask_sensational"
        return jsonify({"reply": "How dramatic or sensational was it? (1–5)"})

    elif s["stage"] == "ask_sensational":
        s["responses"]["sensational"] = int(msg)
        s["stage"] = "ask_threat"
        return jsonify({"reply": "How alarming or threatening did it feel? (1–5)"})

    elif s["stage"] == "ask_threat":
        s["responses"]["threat"] = int(msg)
        s["stage"] = "ask_group"
        return jsonify({"reply": "Did it feel like an 'us vs them' conflict? (1–5)"})

    elif s["stage"] == "ask_group":
        s["responses"]["group_conflict"] = int(msg)
        s["stage"] = "ask_emotions"
        return jsonify({"reply": "What emotions did you feel? (comma separated)"})

    elif s["stage"] == "ask_emotions":
        s["responses"]["emotions"] = msg
        s["stage"] = "ask_highlight"
        return jsonify({"reply": "Paste a sentence that shaped your impression (optional)"})

    # ---------- SAVE ----------
    elif s["stage"] == "ask_highlight":
        s["responses"]["highlight"] = msg

        save_review({
            "reviewer_id": s["reviewer_id"],
            "article_id": s["article"]["id"],
            **s["responses"]
        })

        s["stage"] = "ask_id"
        return jsonify({
            "reply": "Thanks! Your brain just trained a model 🧠✨\nSend your ID again to review another article."
        })

    return jsonify({"reply": "Something went wrong."})


if __name__ == "__main__":
    app.run(debug=True)
