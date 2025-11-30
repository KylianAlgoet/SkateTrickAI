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
    <title>SkateTrick AI</title>
</head>
<body style="font-family: sans-serif; max-width: 600px; margin: 40px auto;">
    <h1>🛹 SkateTrick AI</h1>
    <p>Upload een korte video (1 trick) en ik probeer te raden: kickflip of pop shuvit.</p>
    <form method="post" enctype="multipart/form-data">
        <input type="file" name="video" accept="video/*" required>
        <button type="submit">Analyze</button>
    </form>

    {% if result %}
        <h2>Resultaat</h2>
        <p><strong>Predicted trick:</strong> {{ result.trick }} (confidence: {{ result.confidence }})</p>
    {% endif %}
</body>
</html>
"""

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
                result = {"trick": "unknown", "confidence": "0.00"}
            else:
                df = pd.DataFrame([feats])
                for col in feature_columns:
                    if col not in df.columns:
                        df[col] = 0.0
                df = df[feature_columns]

                pred = model.predict(df)[0]
                proba = model.predict_proba(df)[0]
                labels = model.classes_
                conf = proba[list(labels).index(pred)]

                result = {
                    "trick": pred,
                    "confidence": f"{conf:.2f}",
                }

    return render_template_string(HTML, result=result)


if __name__ == "__main__":
    # Start de Flask dev server
    app.run(debug=True)
