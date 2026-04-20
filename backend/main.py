from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
from PIL import Image
import numpy as np
from datetime import datetime

app = FastAPI()

# ------------------ CORS ------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------ LOAD MODEL ------------------
model = joblib.load("model/text_model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

# ------------------ KEYWORDS ------------------
keywords_list = [
    "ignore", "bypass", "admin", "hack",
    "confidential", "password", "override",
    "leak", "exploit", "access", "inject"
]

critical_keywords = [
    "password", "admin", "bypass", "hack",
    "override", "confidential", "leak", "exploit"
]

# ------------------ HISTORY ------------------
history = []

# ------------------ SCHEMA ------------------
class TextInput(BaseModel):
    text: str

# ------------------ SAVE HISTORY ------------------
def save_history(entry):
    history.insert(0, entry)
    if len(history) > 5:
        history.pop()

# ------------------ ROOT ------------------
@app.get("/")
def home():
    return {"message": "🚀 NeuroShield Backend Running"}

# ------------------ TEXT DETECTION ------------------
@app.post("/detect-text")
def detect_text(input: TextInput):
    text = input.text.lower()

    # ML prediction
    X = vectorizer.transform([text])
    prob = model.predict_proba(X)[0][1]

    # Keyword detection
    keywords = [k for k in keywords_list if k in text]
    critical_hit = any(k in text for k in critical_keywords)

    # ------------------ HYBRID SCORING ------------------
    base_score = prob * 100
    keyword_score = len(keywords) * 8
    critical_boost = 25 if critical_hit else 0

    final_score = min(100, base_score + keyword_score + critical_boost)

    # ------------------ DECISION ------------------
    if final_score >= 75:
        status = "ATTACK"
        reason = "High confidence malicious input"
    elif final_score >= 40:
        status = "SUSPICIOUS"
        reason = "Potentially harmful input"
    else:
        status = "SAFE"
        reason = "No significant threat detected"

    # ------------------ THREAT LEVEL ------------------
    if final_score >= 75:
        threat = "HIGH"
    elif final_score >= 40:
        threat = "MEDIUM"
    else:
        threat = "LOW"

    result = {
        "type": "text",
        "status": status,
        "risk_score": round(final_score, 2),
        "threat_level": threat,
        "reason": reason,
        "matched_keywords": keywords,
        "timestamp": datetime.now().strftime("%H:%M:%S")
    }

    save_history(result)
    return result

# ------------------ IMAGE DETECTION ------------------
@app.post("/detect-image")
def detect_image(file: UploadFile = File(...)):
    filename = file.filename.lower()

    suspicious_words = ["hack", "exploit", "attack"]
    matched = [w for w in suspicious_words if w in filename]

    # Rule 1: filename check
    if matched:
        result = {
            "type": "image",
            "status": "ATTACK",
            "risk_score": 85,
            "threat_level": "HIGH",
            "reason": "Suspicious filename detected",
            "matched_keywords": matched,
            "timestamp": datetime.now().strftime("%H:%M:%S")
        }
        save_history(result)
        return result

    try:
        # Rule 2: noise detection
        img = Image.open(file.file)
        arr = np.array(img)

        variance = np.var(arr)

        if variance > 5000:
            result = {
                "type": "image",
                "status": "SUSPICIOUS",
                "risk_score": 65,
                "threat_level": "MEDIUM",
                "reason": "High noise detected (possible adversarial)",
                "matched_keywords": [],
                "timestamp": datetime.now().strftime("%H:%M:%S")
            }
        else:
            result = {
                "type": "image",
                "status": "SAFE",
                "risk_score": 20,
                "threat_level": "LOW",
                "reason": "Image appears normal",
                "matched_keywords": [],
                "timestamp": datetime.now().strftime("%H:%M:%S")
            }

    except:
        result = {
            "type": "image",
            "status": "ATTACK",
            "risk_score": 90,
            "threat_level": "HIGH",
            "reason": "Invalid or corrupted image",
            "matched_keywords": [],
            "timestamp": datetime.now().strftime("%H:%M:%S")
        }

    save_history(result)
    return result

# ------------------ HISTORY API ------------------
@app.get("/history")
def get_history():
    return history