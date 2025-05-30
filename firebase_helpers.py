import requests
from datetime import datetime
import streamlit as st

API_KEY = st.secrets["firebase"]["apiKey"]
PROJECT_ID = st.secrets["firebase"]["projectId"]
FIRESTORE_EVALS_URL = f"https://firestore.googleapis.com/v1/projects/{PROJECT_ID}/databases/(default)/documents/evaluations"

def save_evaluation_firestore(id_token, user_id, evaluator_name, prompt, query, response, sources, rating, feedback):
    headers = {"Authorization": f"Bearer {id_token}"}
    doc = {
        "fields": {
            "prompt": {"stringValue": prompt},
            "query": {"stringValue": query},
            "response": {"stringValue": response},
            "sources": {"stringValue": str(sources)},
            "rating": {"integerValue": str(rating)},
            "feedback": {"stringValue": feedback},
            "timestamp": {"timestampValue": datetime.utcnow().isoformat() + "Z"},
            "user_id": {"stringValue": user_id},
            "evaluator_name": {"stringValue": evaluator_name}
        }
    }
    resp = requests.post(FIRESTORE_EVALS_URL, headers=headers, json=doc)
    return resp.status_code == 200

def fetch_evaluations_from_firestore(id_token, user_id=None):
    headers = {"Authorization": f"Bearer {id_token}"}
    url = FIRESTORE_EVALS_URL
    resp = requests.get(url, headers=headers)
    if resp.status_code != 200:
        return []
    docs = resp.json().get("documents", [])
    evaluations = []
    for doc in docs:
        fields = doc.get("fields", {})
        # Optionally filter by user_id
        if user_id and fields.get("user_id", {}).get("stringValue") != user_id:
            continue
        evaluations.append({
            "prompt": fields.get("prompt", {}).get("stringValue", ""),
            "query": fields.get("query", {}).get("stringValue", ""),
            "response": fields.get("response", {}).get("stringValue", ""),
            "sources": fields.get("sources", {}).get("stringValue", ""),
            "rating": int(fields.get("rating", {}).get("integerValue", 0)),
            "feedback": fields.get("feedback", {}).get("stringValue", ""),
            "timestamp": fields.get("timestamp", {}).get("timestampValue", ""),
            "evaluator_name": fields.get("evaluator_name", {}).get("stringValue", ""),
        })
    return evaluations 