from flask import Flask, render_template, request
import pickle
import sqlite3
from datetime import datetime

# -------------------------
# Initialize Flask app
# -------------------------
app = Flask(__name__)

# -------------------------
# Load trained ML model
# -------------------------
with open("model/model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# -------------------------
# Initialize SQLite database
# -------------------------
def init_db():
    conn = sqlite3.connect("history.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT,
            prediction TEXT,
            confidence REAL,
            timestamp TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

# -------------------------
# Main route: Home page
# -------------------------
@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    confidence = None
    news_text = ""

    if request.method == "POST":
        news_text = request.form.get("news_text", "").strip()

        if news_text != "":
            # Vectorize input and predict
            vectorized_text = vectorizer.transform([news_text])
            result = model.predict(vectorized_text)[0]
            prob = model.predict_proba(vectorized_text)[0]

            confidence = round(max(prob) * 100, 2)
            prediction = "REAL NEWS" if result == 1 else "FAKE NEWS"

            # Save prediction to database (first 500 chars)
            conn = sqlite3.connect("history.db")
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO history (text, prediction, confidence, timestamp)
                VALUES (?, ?, ?, ?)
            """, (news_text[:500], prediction, confidence, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
            conn.commit()
            conn.close()

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        news_text=news_text
    )

# -------------------------
# History page route
# -------------------------
@app.route("/history")
def history():
    conn = sqlite3.connect("history.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM history ORDER BY id DESC")
    records = cursor.fetchall()
    conn.close()

    return render_template("history.html", records=records)

# -------------------------
# About page route
# -------------------------
@app.route("/about")
def about():
    return render_template("about.html")

# -------------------------
# Run Flask server
# -------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)