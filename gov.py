from flask import Flask, request, render_template, redirect, url_for, send_from_directory, session, jsonify
import pandas as pd
import os
import plotly.express as px
import plotly.io as pio

app = Flask(__name__)
app.secret_key = "your_secret_key"  # Needed for session
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    summary_html = None
    plot_div = ""
    cleaned_file = None
    login_message = None

    # Handle login message
    if "login_message" in session:
        login_message = session.pop("login_message")

    # Only allow upload if logged in
    if request.method == "POST" and session.get("logged_in"):
        import json
        import numpy as np
        from sklearn.impute import KNNImputer
        file = request.files["survey_file"]
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        # Read CSV or Excel file
        if filepath.endswith(".csv"):
            df = pd.read_csv(filepath)
        else:
            df = pd.read_excel(filepath)

        # Schema mapping
        schema_file = request.files.get("schema_file")
        if schema_file and schema_file.filename:
            schema = json.load(schema_file)
            df.rename(columns=schema, inplace=True)

        # Outlier detection (apply only to numeric columns)
        outlier_method = request.form.get("outlier_method", "none")
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        if outlier_method == "iqr":
            Q1 = df[numeric_cols].quantile(0.25)
            Q3 = df[numeric_cols].quantile(0.75)
            IQR = Q3 - Q1
            mask = ~((df[numeric_cols] < (Q1 - 1.5 * IQR)) | (df[numeric_cols] > (Q3 + 1.5 * IQR))).any(axis=1)
            df = df[mask]
        elif outlier_method == "zscore":
            z_scores = np.abs((df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std())
            mask = (z_scores < 3).all(axis=1)
            df = df[mask]
        elif outlier_method == "winsor":
            for col in numeric_cols:
                df[col] = df[col].clip(lower=df[col].quantile(0.05), upper=df[col].quantile(0.95))

        # Drop columns with >40% missing values (default, can be expanded for row/col selection)
        threshold = 0.4
        missing_fraction = df.isnull().mean()
        cols_to_drop = missing_fraction[missing_fraction > threshold].index
        df.drop(columns=cols_to_drop, inplace=True)

        # Imputation
        impute_method = request.form.get("impute_method", "median")
        for col in df.columns:
            if df[col].dtype in ['float64', 'int64']:
                if impute_method == "mean":
                    df[col] = df[col].fillna(df[col].mean())
                elif impute_method == "median":
                    df[col] = df[col].fillna(df[col].median())
                elif impute_method == "knn":
                    imputer = KNNImputer(n_neighbors=3)
                    df[[col]] = imputer.fit_transform(df[[col]])
            else:
                if not df[col].mode().empty:
                    df[col] = df[col].fillna(df[col].mode()[0])

        # Rule-based validation
        rules_file = request.files.get("rules_file")
        rule_warnings = []
        if rules_file and rules_file.filename:
            rules = json.load(rules_file)
            for col, rule in rules.items():
                if col not in df.columns:
                    rule_warnings.append(f"Warning: Column '{col}' not found in data, rule skipped.")
                    continue
                if "min" in rule:
                    invalid = df[df[col] < rule["min"]]
                    if not invalid.empty:
                        rule_warnings.append(f"{col}: {len(invalid)} values below {rule['min']}")
                if "max" in rule:
                    invalid = df[df[col] > rule["max"]]
                    if not invalid.empty:
                        rule_warnings.append(f"{col}: {len(invalid)} values above {rule['max']}")

        # Weight application
        weight_column = request.form.get("weight_column")
        weighted_mean = None
        if weight_column and weight_column in df.columns:
            # Ensure weights are numeric and drop rows with invalid weights
            df[weight_column] = pd.to_numeric(df[weight_column], errors='coerce')
            valid_weights = df[weight_column].notnull()
            weights = df.loc[valid_weights, weight_column]
            # Use only numeric columns (excluding the weight column)
            numeric_cols = df.select_dtypes(include='number').columns.drop(weight_column, errors='ignore')
            data_for_weight = df.loc[valid_weights, numeric_cols]
            denom = weights.sum()
            if pd.notnull(denom) and denom != 0:
                weighted_mean = (data_for_weight.multiply(weights, axis=0).sum() / denom)
            else:
                weighted_mean = None
            # You can add more weighted stats as needed

        # Save cleaned file
        cleaned_filename = "cleaned_" + file.filename
        cleaned_filepath = os.path.join(UPLOAD_FOLDER, cleaned_filename)
        df.to_csv(cleaned_filepath, index=False)
        cleaned_file = cleaned_filename

        # Generate summary
        summary_html = df.describe().to_html(classes="table table-bordered")
        if weighted_mean is not None:
            summary_html += "<br><b>Weighted Means:</b><br>" + weighted_mean.to_frame("Weighted Mean").to_html(classes="table table-bordered")
        if rule_warnings:
            summary_html += "<br><b>Rule Warnings:</b><ul>" + "".join(f"<li>{w}</li>" for w in rule_warnings) + "</ul>"

        # Generate bar chart
        if not df.empty:
            fig = px.bar(df.describe().transpose(), title="Summary Bar Chart")
            plot_div = pio.to_html(fig, full_html=False)

    return render_template(
        "index.html",
        summary=summary_html,
        plot_div=plot_div,
        cleaned_file=cleaned_file,
        login_message=login_message,
        logged_in=session.get("logged_in", False)
    )

@app.route("/login", methods=["POST"])
def login():
    username = request.form["username"]
    password = request.form["password"]
    # Simple check (replace with real authentication)
    if username == "admin" and password == "admin":
        session["logged_in"] = True
        session["login_message"] = "Login successful!"
    else:
        session["logged_in"] = False
        session["login_message"] = "Invalid credentials."
    return redirect(url_for("index"))

@app.route("/logout")
def logout():
    session.pop("logged_in", None)
    session["login_message"] = "Logged out."
    return redirect(url_for("index"))

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)

# --- Report Generation Route ---
from datetime import datetime
import pdfkit

@app.route("/report/<filename>")
def generate_report(filename):
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    df = pd.read_csv(filepath)

    # Diagnostics (example: imputation log)
    imputation_log = "Missing values filled using forward fill."

    # Summary statistics
    summary_html = df.describe().to_html(classes="table table-bordered")

    # Visualization (bar chart)
    fig = px.bar(df.describe().transpose(), title="Summary Bar Chart")
    plot_div = pio.to_html(fig, full_html=False)

    # Workflow log
    workflow_log = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "file_lineage": filename,
        "changes_applied": "Forward fill imputation"
    }

    # Render HTML report
    report_html = render_template(
        "report_template.html",
        summary=summary_html,
        imputation_log=imputation_log,
        plot_div=plot_div,
        workflow_log=workflow_log
    )

    # Export to PDF if requested
    export_format = request.args.get("format", "html")
    if export_format == "pdf":
        pdf_path = os.path.join(UPLOAD_FOLDER, filename.replace(".csv", "_report.pdf"))
        pdfkit.from_string(report_html, pdf_path)
        return send_from_directory(app.config['UPLOAD_FOLDER'], os.path.basename(pdf_path), as_attachment=True)
    else:
        return report_html

# --- Simple Chat API ---
@app.route("/chat", methods=["POST"])
def chat():
    """AI chat endpoint with Ollama (local) and OpenAI support.
    Priority:
      1) If AI_PROVIDER=ollama, call local Ollama
      2) Else if OPENAI_API_KEY set, call OpenAI
      3) Else try Ollama as best-effort
      4) Else return stub message
    """
    data = request.get_json(silent=True) or {}
    user_msg = (data.get("message") or "").strip()
    if not user_msg:
        return jsonify({"error": "message is required"}), 400

    provider = (os.environ.get("AI_PROVIDER") or "").lower()
    api_key = os.environ.get("OPENAI_API_KEY")
    openai_model = os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo")
    openai_base = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
    ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    ollama_model = os.environ.get("OLLAMA_MODEL", "llama3")

    def call_ollama(msg: str):
        try:
            import requests
            payload = {
                "model": ollama_model,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant for the Survey Data Dashboard."},
                    {"role": "user", "content": msg},
                ],
                "stream": False,
            }
            r = requests.post(f"{ollama_host}/api/chat", json=payload, timeout=30)
            if r.status_code == 200:
                j = r.json()
                reply = (j.get("message") or {}).get("content") or j.get("response") or ""
                return {"ok": True, "text": reply or "(no content)"}
            return {"ok": False, "text": f"Ollama error: {r.status_code} {r.text[:200]}"}
        except Exception as e:
            return {"ok": False, "text": f"Ollama backend error: {e}"}

    def call_openai(msg: str):
        try:
            import requests
            payload = {
                "model": openai_model,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant for the Survey Data Dashboard."},
                    {"role": "user", "content": msg},
                ],
                "temperature": 0.3,
                "max_tokens": 300,
            }
            resp = requests.post(
                f"{openai_base}/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=20,
            )
            if resp.status_code == 200:
                j = resp.json()
                reply = j.get("choices", [{}])[0].get("message", {}).get("content", "")
                return {"ok": True, "text": reply or "(no content)"}
            if resp.status_code == 429:
                return {"ok": True, "text": "AI quota or rate limit exceeded. Please check plan/billing or try again later."}
            return {"ok": False, "text": f"AI error: {resp.status_code} {resp.text[:200]}"}
        except Exception as e:
            return {"ok": False, "text": f"AI backend error: {e}"}

    # Strategy
    if provider == "ollama":
        res = call_ollama(user_msg)
        status = 200 if res["ok"] else 502
        return jsonify({"reply": res["text"]}), status
    if api_key:
        res = call_openai(user_msg)
        if res["ok"]:
            return jsonify({"reply": res["text"]})
        # Fallback to Ollama if available
        res2 = call_ollama(user_msg)
        status = 200 if res2["ok"] else 502
        return jsonify({"reply": res["text"] if res2["ok"] is False else res2["text"]}), status
    # Try Ollama as best-effort when no OpenAI key
    res = call_ollama(user_msg)
    if res["ok"]:
        return jsonify({"reply": res["text"]})
    return jsonify({"reply": "AI is not configured. Set OPENAI_API_KEY or run Ollama with a model (e.g., OLLAMA_MODEL=llama3)."})

if __name__ == "__main__":
    app.run(debug=True)