from fastapi import FastAPI, Request, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
import joblib
import os
import io
import shap

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Load trained model
MODEL_PATH = 'trained_models/prevention_model_unsw.pkl'
model = joblib.load(MODEL_PATH)
model_features = model.feature_names_in_

# Descriptions of each attack type
ATTACK_EXPLANATIONS = {
    'Normal': 'No signs of attack detected. Network traffic is safe.',
    'DoS': 'Denial of Service attack caused by overwhelming traffic or protocol abuse.',
    'Exploit': 'Leverages software vulnerabilities to gain unauthorized access.',
    'Worm': 'Self-replicating attack spreading through the network.',
    'Fuzzer': 'Random/malformed data injection to crash or exploit systems.',
    'Generic': 'Pattern-matching attack commonly flagged by IDS tools.',
    'Reconnaissance': 'Probing or scanning activities to discover system weaknesses.'
}

@app.get("/", response_class=HTMLResponse)
async def read_form(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.post("/analyze", response_class=HTMLResponse)
async def analyze(request: Request, file: UploadFile = File(...)):
    contents = await file.read()
    try:
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
    except UnicodeDecodeError:
        df = pd.read_csv(io.StringIO(contents.decode('ISO-8859-1')))

    # Ensure model features are present
    if not all(f in df.columns for f in model_features):
        raise HTTPException(status_code=400, detail=f"Missing features: {model_features}")

    # Use only the first row for dynamic analysis
    row = df.iloc[0:1][model_features]
    pred = model.predict(row)[0]
    pred_proba = model.predict_proba(row)[0][1]

    # Generate SHAP values for the uploaded row
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(row)
    contributions = shap_vals[1][0] if isinstance(shap_vals, list) else shap_vals[0]
    top_contributors = sorted(
        zip(model_features, contributions),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:3]

    # Generate descriptive dynamic alerts
    def generate_alert(feature, value, impact):
        direction = "↑" if impact > 0 else "↓"
        return f"⚠️ {feature} {direction} unusual ({value:.2f}) – contributing to attack risk."

    alerts = [
        generate_alert(f, float(row[f]), impact)
        for f, impact in top_contributors
    ]

    # Generate dynamic AI agent report
    if pred == 0:
        agent_report = "🟢 No intrusion detected. All monitored metrics fall within safe ranges."
        prediction_label = "Normal"
    else:
        prediction_label = "Attack"
        agent_report = "🔴 **Intrusion Likely Detected**\n\n"
        agent_report += "### Contributing Feature Values:\n"
        for f, impact in top_contributors:
            agent_report += f"- **{f}**: {row[f].values[0]:.2f}\n"
        agent_report += "\n### 🛡 Recommended Preventive Actions:\n"
        agent_report += "- Block unusual IPs and monitor port access\n"
        agent_report += "- Inspect packet headers for spoofed addresses\n"
        agent_report += "- Limit access to exposed services and APIs\n"
        agent_report += "- Schedule full traffic analysis using IDS/IPS tools\n"

    attack_explanation = ATTACK_EXPLANATIONS.get(prediction_label, "No description available.")

    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "prediction": prediction_label,
        "alerts": alerts,
        "agent_report": agent_report,
        "attack_explanation": attack_explanation
    })
