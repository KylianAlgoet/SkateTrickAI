from flask import Flask, request, render_template_string
from pathlib import Path
import pandas as pd
from joblib import load
from src.extract_features import extract_video_features
from config import MODEL_PATH

app = Flask(__name__)

# Model laden
bundle = load(MODEL_PATH)
model = bundle["model"]
feature_columns = bundle["feature_columns"]

HTML = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8" />
    <title>SkateTrick AI</title>
    <style>
        :root {
            --bg: #050816;
            --bg-card: #0b1120;
            --accent: #6366f1;
            --accent-soft: #4f46e5;
            --text: #e5e7eb;
            --muted: #9ca3af;
            --border: #1f2937;
            --danger: #ef4444;
            --warning: #f97316;
            --success: #22c55e;
        }

        * {
            box-sizing: border-box;
        }

        body {
            margin: 0;
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            background: radial-gradient(circle at top, #1e293b 0, #020617 55%, #000 100%);
            color: var(--text);
        }

        .container {
            width: 100%;
            max-width: 720px;
            padding: 24px;
        }

        .card {
            background: linear-gradient(145deg, rgba(15,23,42,0.98), rgba(15,23,42,0.9));
            border-radius: 20px;
            border: 1px solid var(--border);
            padding: 24px 28px 28px;
            box-shadow:
                0 24px 60px rgba(15,23,42,0.9),
                0 0 0 1px rgba(148,163,184,0.05);
        }

        .header {
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 12px;
        }

        .logo-circle {
            width: 40px;
            height: 40px;
            border-radius: 999px;
            background: radial-gradient(circle at 30% 30%, #f97316, #4f46e5);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 22px;
            box-shadow: 0 0 20px rgba(129,140,248,0.7);
        }

        h1 {
            margin: 0;
            font-size: 26px;
            letter-spacing: 0.03em;
        }

        .subtitle {
            margin: 0;
            font-size: 14px;
            color: var(--muted);
        }

        .upload-card {
            margin-top: 20px;
            padding: 16px 18px;
            border-radius: 16px;
            border: 1px dashed rgba(148,163,184,0.6);
            background: radial-gradient(circle at 0 0, rgba(129,140,248,0.15), transparent 60%);
        }

        .upload-row {
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
            align-items: center;
            margin-top: 10px;
        }

        input[type="file"] {
            color: var(--muted);
            font-size: 14px;
            max-width: 100%;
        }

        button {
            border: none;
            outline: none;
            cursor: pointer;
            padding: 10px 18px;
            border-radius: 999px;
            font-size: 14px;
            font-weight: 600;
            letter-spacing: 0.03em;
            text-transform: uppercase;
            background: linear-gradient(135deg, var(--accent), var(--accent-soft));
            color: white;
            box-shadow:
                0 10px 25px rgba(79,70,229,0.55),
                0 0 0 1px rgba(191,219,254,0.25);
            transition: transform 0.12s ease, box-shadow 0.12s ease, filter 0.12s ease;
        }

        button:hover {
            transform: translateY(-1px);
            filter: brightness(1.05);
            box-shadow:
                0 14px 35px rgba(79,70,229,0.7),
                0 0 0 1px rgba(191,219,254,0.3);
        }

        button:active {
            transform: translateY(0);
            box-shadow:
                0 6px 18px rgba(79,70,229,0.65),
                0 0 0 1px rgba(191,219,254,0.25);
        }

        .result-card {
            margin-top: 22px;
            padding: 16px 18px 18px;
            border-radius: 16px;
            background: radial-gradient(circle at top right, rgba(52,211,153,0.16), rgba(15,23,42,0.98));
            border: 1px solid rgba(148,163,184,0.35);
        }

        .result-title {
            font-size: 15px;
            text-transform: uppercase;
            letter-spacing: 0.14em;
            color: var(--muted);
            margin-bottom: 6px;
        }

        .result-main {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            align-items: center;
            justify-content: space-between;
        }

        .trick-name {
            font-size: 18px;
            font-weight: 700;
        }

        .pill {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 4px 10px;
            border-radius: 999px;
            font-size: 12px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            border: 1px solid rgba(148,163,184,0.5);
            background: rgba(15,23,42,0.85);
        }

        .pill-dot {
            width: 8px;
            height: 8px;
            border-radius: 999px;
        }

        .pill-label {
            color: var(--muted);
        }

        .pill-value {
            color: var(--text);
        }

        .meter-row {
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
            margin-top: 12px;
        }

        .meter {
            flex: 1 1 180px;
        }

        .meter-label {
            font-size: 12px;
            color: var(--muted);
            margin-bottom: 4px;
        }

        .meter-bar {
            position: relative;
            width: 100%;
            height: 8px;
            border-radius: 999px;
            background: rgba(15,23,42,0.9);
            overflow: hidden;
            border: 1px solid rgba(31,41,55,0.9);
        }

        .meter-fill {
            position: absolute;
            inset: 0;
            width: 0;
            border-radius: 999px;
        }

        .meter-text {
            margin-top: 3px;
            font-size: 12px;
            color: var(--muted);
        }

        .footer-note {
            margin-top: 12px;
            font-size: 11px;
            color: var(--muted);
        }

        @media (max-width: 640px) {
            .card { padding: 20px; }
        }
    </style>
</head>
<body>
<div class="container">
    <div class="card">
        <div class="header">
            <div class="logo-circle">🛹</div>
            <div>
                <h1>SkateTrick AI</h1>
                <p class="subtitle">Upload een clip &amp; laat de AI raden of je een kickflip of pop shuvit landt.</p>
            </div>
        </div>

        <form method="post" enctype="multipart/form-data">
            <div class="upload-card">
                <div class="subtitle" style="font-size: 13px;">
                    ① Film één gelande trick &nbsp;·&nbsp; ② Upload de video &nbsp;·&nbsp; ③ Check prediction &amp; cleanliness.
                </div>
                <div class="upload-row">
                    <input type="file" name="video" accept="video/*" required>
                    <button type="submit">Analyze</button>
                </div>
            </div>
        </form>

        {% if result %}
            <div class="result-card">
                <div class="result-title">Resultaat</div>
                <div class="result-main">
                    <div class="trick-name">
                        {{ result.trick|replace("_", " ")|title }}
                    </div>

                    <div style="display: flex; gap: 8px; flex-wrap: wrap;">
                        <div class="pill">
                            <div class="pill-dot" style="background-color: {{ result.conf_color }};"></div>
                            <span class="pill-label">Confidence</span>
                            <span class="pill-value">{{ result.confidence }}</span>
                        </div>
                        <div class="pill">
                            <div class="pill-dot" style="background-color: {{ result.clean_color }};"></div>
                            <span class="pill-label">Cleanliness</span>
                            <span class="pill-value">{{ result.cleanliness }} / 100</span>
                        </div>
                    </div>
                </div>

                <div class="meter-row">
                    <div class="meter">
                        <div class="meter-label">Confidence</div>
                        <div class="meter-bar">
                            <div class="meter-fill" style="width: {{ result.conf_pct }}%; background: {{ result.conf_color }};"></div>
                        </div>
                        <div class="meter-text">{{ result.conf_level }}</div>
                    </div>
                    <div class="meter">
                        <div class="meter-label">Cleanliness score</div>
                        <div class="meter-bar">
                            <div class="meter-fill" style="width: {{ result.cleanliness }}%; background: {{ result.clean_color }};"></div>
                        </div>
                        <div class="meter-text">{{ result.clean_level }}</div>
                    </div>
                </div>

                <div class="footer-note">
                    Dit prototype gebruikt MediaPipe Pose + een RandomForest-classifier getraind op je eigen skatefootage.
                </div>
            </div>
        {% endif %}
    </div>
</div>
</body>
</html>
"""

def confidence_color_and_label(conf: float):
    if conf >= 0.8:
        return "#22c55e", "High confidence"
    elif conf >= 0.5:
        return "#f97316", "Medium confidence"
    else:
        return "#ef4444", "Low confidence"

def cleanliness_color_and_label(score: int):
    if score >= 80:
        return "#22c55e", "Super clean landing"
    elif score >= 50:
        return "#f97316", "Ok, maar kan cleaner"
    else:
        return "#ef4444", "Sketchy landing"


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        file = request.files.get("video")
        if file:
            save_path = Path("upload_temp.mp4")
            file.save(save_path)

            feats = extract_video_features(save_path)
            save_path.unlink(missing_ok=True)

            if feats is None:
                result = {
                    "trick": "unknown",
                    "confidence": "0.00",
                    "cleanliness": 0,
                    "conf_color": "#ef4444",
                    "conf_level": "No pose detected",
                    "conf_pct": 0,
                    "clean_color": "#ef4444",
                    "clean_level": "No score",
                }
            else:
                df = pd.DataFrame([feats])
                for col in feature_columns:
                    if col not in df.columns:
                        df[col] = 0.0
                df = df[feature_columns]

                pred = model.predict(df)[0]
                proba = model.predict_proba(df)[0]
                labels = model.classes_
                conf = float(proba[list(labels).index(pred)])

                clean_raw = float(feats.get("cleanliness_score_raw", 0.0))
                clean_display = int(max(0, min(100, clean_raw * 100)))

                conf_color, conf_level = confidence_color_and_label(conf)
                clean_color, clean_level = cleanliness_color_and_label(clean_display)

                result = {
                    "trick": pred,
                    "confidence": f"{conf:.2f}",
                    "cleanliness": clean_display,
                    "conf_color": conf_color,
                    "conf_level": conf_level,
                    "conf_pct": int(conf * 100),
                    "clean_color": clean_color,
                    "clean_level": clean_level,
                }

    return render_template_string(HTML, result=result)


if __name__ == "__main__":
    app.run(debug=True)
