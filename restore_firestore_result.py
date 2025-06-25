import requests
import toml
from datetime import datetime, timezone

# --- CONFIGURATION --- #
TOML_PATH = "./.streamlit/secrets.toml"
BATCH_ID = "batch_1"
RESULT_DOC_ID = "0MAC87jFW1apvwMC8rL2"

# The fields you want to restore (edit as needed)
MISSING_FIELDS = {
    "category": {"stringValue": "Nutrition"},
    "persona": {"stringValue": "Older man 60+ with central adiposity, heart disease, high blood pressure, high cholesterol"},
    "question": {"stringValue": "I am really not hungry and can't eat without getting bad side effects, is this normal?"},
    "query": {"stringValue": "I'm a Older man 60+ with central adiposity, heart disease, high blood pressure, high cholesterol I am really not hungry and can't eat without getting bad side effects, is this normal?"},
    "response": {"stringValue": "Okay, I understand. It sounds like you're going through a tough time with your appetite and side effects. It's definitely not something to ignore, and it's great that you're reaching out for information.\n\nFirst off, I want to reassure you that many people experience changes in appetite and side effects when dealing with conditions like yours, especially when taking medications. However, \"normal\" can be different for everyone, and it's important to figure out what's going on with you.\n\nHere's a breakdown of what might be happening, keeping in mind your age (60+), central adiposity (belly fat), heart disease, high blood pressure, and high cholesterol:\n\nWhat Could Be Contributing to This?\n\nYour Health Conditions: Heart disease and high blood pressure can sometimes lead to a loss of appetite or make you feel sick after eating. [web1]\nMedications: Lots of medicines, especially as we get older, can mess with your appetite, change how things taste, or make it harder for your body to get the nutrients it needs. This includes meds for blood pressure and cholesterol. [web1]\nAge-Related Changes: As we age, our bodies change. Our metabolism slows down, and our sense of taste and smell might not be as strong, which can make food less appealing. [web1]\nBelly Fat: Excess belly fat can affect hormones that control appetite, like leptin. [web1]\nGLP-1 Medications: It's important to consider if you are currently taking any GLP-1 medications, as appetite suppression is a common side effect. GI side effects are also common, especially when starting the medication or increasing the dose [doc2].\nWhat Can You Do?\n\nTalk to Your Doctor: This is the most important step. Tell them exactly what you're experiencing. They can run tests to see what's going on and adjust your medications if needed. [web1]\nReview Your Meds: Ask your doctor or pharmacist to go over all your medications to see if any of them could be causing these side effects or interacting with each other. [web1]\nEat Smart:\nSmall Meals: Try eating smaller meals more often instead of three big ones. [doc1]\nBland Foods: Stick to bland foods that are easy to digest. Avoid high-fat, high-sugar, and spicy foods. [doc1]\nFocus on Nutrients: Make sure the food you do eat is packed with nutrients. If you're not eating much, every bite counts!\nStay Hydrated: Drink plenty of water, and consider adding electrolytes if you're having diarrhea. [doc1]\nMonitor Your Symptoms: Keep track of when you feel sick, what you ate, and any other symptoms you're having. This can help your doctor figure out what's going on. [web1]\nImportant Note: I'm here to offer support and information, but I'm not a doctor. It's super important to talk to your healthcare provider about what's going on. They can give you the best advice for your specific situation."},
    "quality_check": {"stringValue": "{\"confidence_score\": 0.9, \"missing_elements\": [], \"reasoning\": \"The answer provides a reasonable overview of potential causes for the patient's symptoms, including existing health conditions, medications, age-related changes, and the influence of belly fat. It gives practical advice such as consulting a doctor, reviewing medications, and focusing on smart eating habits. The answer also contains appropriate disclaimers and citations.\"}"},
    "evaluations": {"stringValue": "[{\"evaluator_name\": \"Bryn Falahee\", \"user_id\": \"oQRGyekWmeNf9QYZXtbUaZ6kzYi1\", \"rating\": 7, \"timestamp\": \"2025-06-07T17:30:13.491294Z\", \"feedback\": \"This is a good general answer, but should be more tailored to GLP-1 side effects if that was what the patient was asking about. A bit too general regarding how HTN, heart conditions etc affect appetite.\"}, {\"evaluator_name\": \"Flo Porterfield\", \"user_id\": \"2Hdvuc8x2cR5z8hG8i0RwhtK4or2\", \"rating\": 2, \"timestamp\": \"2025-06-11T21:24:34.069880Z\", \"feedback\": \"The patient's lack of appetite and nausea is most likely from the GLP-1 so I would focus on that. I would expand upon the eat smart, stay hydrated, and monitor your symptoms part as this is the most helpful. No need to mention changes related to age. You can put in the bottom as a disclaimer \\\"there may be other causes of low appetite and nausea so be sure to talk to your doctor to see what else could be causing your symptoms\\\".\"}]"}
}
# --------------------- #

def get_firebase_id_token(api_key):
    print("Enter your Firebase email and password to get an ID token.")
    email = input("Email: ").strip()
    password = input("Password: ").strip()
    url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={api_key}"
    payload = {
        "email": email,
        "password": password,
        "returnSecureToken": True
    }
    resp = requests.post(url, json=payload)
    if resp.status_code == 200:
        return resp.json()["idToken"]
    else:
        print("Failed to get ID token:", resp.text)
        exit(1)

def main():
    # 1. Read secrets.toml
    secrets = toml.load(TOML_PATH)
    api_key = secrets["firebase"]["apiKey"]
    project_id = secrets["firebase"]["projectId"]

    # 2. Get ID token
    id_token = get_firebase_id_token(api_key)

    # 3. Build Firestore doc URL
    FIRESTORE_BASE = f"https://firestore.googleapis.com/v1/projects/{project_id}/databases/(default)/documents"
    DOC_URL = f"{FIRESTORE_BASE}/prompt_batch_evaluator/{BATCH_ID}/results/{RESULT_DOC_ID}"

    # 4. Prepare payload
    merge_fields = dict(MISSING_FIELDS)
    merge_fields["restored_at"] = {"timestampValue": datetime.now(timezone.utc).isoformat()}
    payload = {"fields": merge_fields}

    # 5. PATCH request
    headers = {"Authorization": f"Bearer {id_token}"}
    resp = requests.patch(DOC_URL, headers=headers, json=payload)
    if resp.status_code in (200, 201):
        print("Fields restored successfully!")
    else:
        print("Failed to restore fields:", resp.text)

if __name__ == "__main__":
    main() 