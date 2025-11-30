from pathlib import Path
import sys
import pandas as pd
from joblib import load
from config import MODEL_PATH
from src.extract_features import extract_video_features


def predict_video(video_path_str: str):
    video_path = Path(video_path_str)
    if not video_path.exists():
        print(f"❌ Video does not exist: {video_path}")
        return

    bundle = load(MODEL_PATH)
    model = bundle["model"]
    feature_columns = bundle["feature_columns"]

    feats = extract_video_features(video_path)
    if feats is None:
        print("❌ Could not extract features from video.")
        return

    df = pd.DataFrame([feats])

    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0.0
    df = df[feature_columns]

    pred = model.predict(df)[0]
    proba = model.predict_proba(df)[0]
    labels = model.classes_
    confidence = proba[list(labels).index(pred)]

    print(f"🎯 Predicted trick: {pred} (confidence: {confidence:.2f})")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.predict_video path/to/video.mp4")
        sys.exit(1)
    predict_video(sys.argv[1])
