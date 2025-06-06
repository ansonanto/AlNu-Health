import requests
from datetime import datetime
import streamlit as st
import uuid

API_KEY = st.secrets["firebase"]["apiKey"]
PROJECT_ID = st.secrets["firebase"]["projectId"]
FIRESTORE_EVALS_URL = f"https://firestore.googleapis.com/v1/projects/{PROJECT_ID}/databases/(default)/documents/evaluations"
PROMPTS_FIRESTORE_URL = f"https://firestore.googleapis.com/v1/projects/{PROJECT_ID}/databases/(default)/documents/prompts"

def save_prompt_firestore(id_token, user_id, user_name, name, prompt_text):
    headers = {"Authorization": f"Bearer {id_token}"}
    prompt_id = str(uuid.uuid4())
    doc = {
        "fields": {
            "id": {"stringValue": prompt_id},
            "name": {"stringValue": name},
            "text": {"stringValue": prompt_text},
            "created_at": {"timestampValue": datetime.utcnow().isoformat() + "Z"},
            "created_by": {"stringValue": user_id},
            "created_by_name": {"stringValue": user_name}
        }
    }
    prompt_doc_url = f"{PROMPTS_FIRESTORE_URL}/{prompt_id}"
    resp = requests.patch(prompt_doc_url, headers=headers, json=doc)
    return resp.status_code == 200, prompt_id

def load_prompts_firestore(id_token):
    headers = {"Authorization": f"Bearer {id_token}"}
    resp = requests.get(PROMPTS_FIRESTORE_URL, headers=headers)
    if resp.status_code != 200:
        return []
    docs = resp.json().get("documents", [])
    prompts = []
    for doc in docs:
        fields = doc.get("fields", {})
        prompts.append({
            "id": fields.get("id", {}).get("stringValue", ""),
            "name": fields.get("name", {}).get("stringValue", ""),
            "text": fields.get("text", {}).get("stringValue", ""),
            "created_at": fields.get("created_at", {}).get("timestampValue", ""),
            "created_by": fields.get("created_by", {}).get("stringValue", ""),
            "created_by_name": fields.get("created_by_name", {}).get("stringValue", "")
        })
    prompts.sort(key=lambda x: x.get('created_at', ''), reverse=True)
    return prompts

def save_evaluation_firestore(id_token, user_id, evaluator_name, prompt_id, prompt, query, response, sources, rating, feedback):
    headers = {"Authorization": f"Bearer {id_token}"}
    doc = {
        "fields": {
            "prompt_id": {"stringValue": prompt_id or ""},
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
    print(resp.status_code, resp.text)  # Debug print for troubleshooting
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
            "document_id": doc.get("name", "").split("/")[-1],
            "prompt_id": fields.get("prompt_id", {}).get("stringValue", ""),
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

def delete_evaluation_firestore(id_token, evaluation_doc_id):
    headers = {"Authorization": f"Bearer {id_token}"}
    url = f"{FIRESTORE_EVALS_URL}/{evaluation_doc_id}"
    resp = requests.delete(url, headers=headers)
    return resp.status_code == 200 