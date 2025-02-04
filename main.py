import pandas as pd
import torch
import sqlite3
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.preprocessing import LabelEncoder
import uvicorn

# Load Dataset
df = pd.read_csv("Simplified_Complaint_Classification_Dataset.csv")
label_enc_category = LabelEncoder()
df["Category_Label"] = label_enc_category.fit_transform(df["Category"])

# Set Up SQLite Database
conn = sqlite3.connect("complaints.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute("""
    CREATE TABLE IF NOT EXISTS complaints (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        text TEXT,
        category TEXT,
        department TEXT,
        severity TEXT
    )
""")
conn.commit()

# Load Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(df["Category"].unique()))
model.to(device)

app = FastAPI()

class ComplaintRequest(BaseModel):
    text: str

# Classify Complaint Function
def classify_complaint_bert(complaint_text):
    encoding = tokenizer(complaint_text, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
    input_ids, attention_mask = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        prediction = torch.argmax(outputs.logits, dim=1).cpu().numpy()[0]

    category = label_enc_category.inverse_transform([prediction])[0]
    dept_severity = df[df["Category"] == category][["Department", "Severity"]].iloc[0]

    return {"Category": category, "Department": dept_severity["Department"], "Severity": dept_severity["Severity"]}

# FastAPI Endpoints
@app.post("/classify_complaint/")
async def classify_complaint(request: ComplaintRequest):
    result = classify_complaint_bert(request.text)
    cursor.execute("INSERT INTO complaints (text, category, department, severity) VALUES (?, ?, ?, ?)", 
                   (request.text, result["Category"], result["Department"], result["Severity"]))
    conn.commit()
    return result

@app.get("/get_complaints/")
async def get_complaints():
    cursor.execute("SELECT * FROM complaints")
    return [{"id": row[0], "text": row[1], "category": row[2], "department": row[3], "severity": row[4]} for row in cursor.fetchall()]

@app.get("/get_complaints/{category}")
async def get_complaints_by_category(category: str):
    cursor.execute("SELECT * FROM complaints WHERE category=?", (category,))
    results = [{"id": row[0], "text": row[1], "category": row[2], "department": row[3], "severity": row[4]} for row in cursor.fetchall()]
    return results if results else {"message": "No complaints found"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5001)
