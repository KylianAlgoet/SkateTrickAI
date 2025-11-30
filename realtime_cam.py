import cv2
from collections import deque
from pathlib import Path
import pandas as pd
from joblib import load

from src.extract_features import extract_video_features
from config import MODEL_PATH

print("✅ realtime_cam module geladen")

# Model laden
bundle = load(MODEL_PATH)
model = bundle["model"]
feature_columns = bundle["feature_columns"]
print("✅ Model geladen uit", MODEL_PATH)


def analyze_buffer(frame_buffer, fps=30):
    """
    Slaat de frames in frame_buffer tijdelijk op als video
    en stuurt die door dezelfde pipeline als de gewone predictie.
    """
    if not frame_buffer:
        print("⚠️ Geen frames in buffer.")
        return None

    temp_path = Path("realtime_temp.mp4")

    # Frame properties
    h, w, _ = frame_buffer[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(temp_path), fourcc, fps, (w, h))

    for frame in frame_buffer:
        out.write(frame)

    out.release()

    # Extract features
    feats = extract_video_features(temp_path)
    temp_path.unlink(missing_ok=True)

    if feats is None:
        print("⚠️ Geen pose gedetecteerd in realtime opname.")
        return None

    # Dataframe
    df = pd.DataFrame([feats])
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0.0
    df = df[feature_columns]

    # Predictie
    pred = model.predict(df)[0]
    proba = model.predict_proba(df)[0]
    labels = model.classes_
    conf = proba[list(labels).index(pred)]

    clean_raw = feats.get("cleanliness_score_raw", 0.0)
    clean_display = int(max(0, min(100, clean_raw * 100)))

    result = {
        "trick": pred,
        "confidence": float(conf),
        "cleanliness": clean_display,
    }

    return result


def try_open_cam():
    """Probeer camera 0 en 1, return (cap, index) of (None, None)."""
    for idx in [0, 1]:
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            print(f"✅ Webcam gevonden op index {idx}")
            return cap, idx
        cap.release()
    print("❌ Geen webcam gevonden op index 0 of 1.")
    return None, None


def main():
    print("🚀 realtime_cam main() gestart")

    cap, cam_index = try_open_cam()
    if cap is None:
        print("Stop: geen webcam.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps > 120:
        fps = 30  # fallback

    print(f"🎥 FPS geschat op: {fps}")

    # Buffer met laatste ~2 seconden
    buffer_seconds = 2.0
    max_len = int(fps * buffer_seconds)
    frame_buffer = deque(maxlen=max_len)

    last_result_text = "Press T to analyze trick, Q to quit."

    print("🎥 Realtime mode gestart.")
    print(" - Druk op T om de laatste ~2 seconden te analyseren.")
    print(" - Druk op Q om te stoppen.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Geen frame van webcam (index", cam_index, ").")
            break

        frame_buffer.append(frame.copy())

        # Overlay text
        overlay = frame.copy()
        cv2.putText(
            overlay,
            last_result_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("SkateTrick AI - Realtime", overlay)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            print("👋 Q gedrukt, stoppen...")
            break
        elif key == ord("t"):
            print("🔍 Analyseren van laatste frames...")
            result = analyze_buffer(list(frame_buffer), fps=fps)
            if result is None:
                last_result_text = "No trick detected. Try again."
            else:
                last_result_text = (
                    f"{result['trick']} "
                    f"(conf: {result['confidence']:.2f}, "
                    f"clean: {result['cleanliness']}/100)"
                )
                print(
                    f"🎯 Trick: {result['trick']}, "
                    f"confidence: {result['confidence']:.2f}, "
                    f"cleanliness: {result['cleanliness']}/100"
                )

    cap.release()
    cv2.destroyAllWindows()
    print("✅ Webcam vrijgegeven, vensters gesloten.")


if __name__ == "__main__":
    main()
