import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from pathlib import Path
from config import DATA_RAW_DIR, DATASET_CSV, TRICKS

mp_pose = mp.solutions.pose

# Landmarks to use (indexes from MediaPipe Pose)
LANDMARKS_USED = {
    "left_ankle":  mp_pose.PoseLandmark.LEFT_ANKLE.value,
    "right_ankle": mp_pose.PoseLandmark.RIGHT_ANKLE.value,
    "left_knee":   mp_pose.PoseLandmark.LEFT_KNEE.value,
    "right_knee":  mp_pose.PoseLandmark.RIGHT_KNEE.value,
    "left_hip":    mp_pose.PoseLandmark.LEFT_HIP.value,
    "right_hip":   mp_pose.PoseLandmark.RIGHT_HIP.value,
}


def extract_video_features(video_path: Path) -> dict | None:
    """
    Take a single video, extract pose keypoints on multiple frames,
    aggregate simple statistics and return a feature dict.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"⚠️ Could not open video: {video_path}")
        return None

    frame_landmarks = []

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        frame_idx = 0

        while True:
            success, frame = cap.read()
            if not success:
                break

            # Skip every second frame for speed
            if frame_idx % 2 != 0:
                frame_idx += 1
                continue

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                pts = []
                for name, idx in LANDMARKS_USED.items():
                    lm = landmarks[idx]
                    pts.append([lm.x, lm.y])

                pts = np.array(pts)

                left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
                right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
                hip_center = np.array([(left_hip.x + right_hip.x) / 2,
                                       (left_hip.y + right_hip.y) / 2])

                pts_centered = pts - hip_center

                hip_dist = np.linalg.norm(
                    np.array([left_hip.x, left_hip.y]) -
                    np.array([right_hip.x, right_hip.y])
                )
                if hip_dist > 0:
                    pts_norm = pts_centered / hip_dist
                else:
                    pts_norm = pts_centered

                frame_landmarks.append(pts_norm)

            frame_idx += 1

    cap.release()

    if len(frame_landmarks) == 0:
        print(f"⚠️ No pose detected in: {video_path}")
        return None

    arr = np.stack(frame_landmarks, axis=0)  # (T, L, 2)

    features = {}
    n_frames, n_landmarks, n_coords = arr.shape

    for li, name in enumerate(LANDMARKS_USED.keys()):
        for ci, coord_name in enumerate(["x", "y"]):
            series = arr[:, li, ci]
            base = f"{name}_{coord_name}"
            features[f"{base}_mean"] = float(series.mean())
            features[f"{base}_std"] = float(series.std())
            features[f"{base}_min"] = float(series.min())
            features[f"{base}_max"] = float(series.max())

    left_ankle_y = arr[:, list(LANDMARKS_USED.keys()).index("left_ankle"), 1]
    right_ankle_y = arr[:, list(LANDMARKS_USED.keys()).index("right_ankle"), 1]
    features["min_left_ankle_y"] = float(left_ankle_y.min())
    features["min_right_ankle_y"] = float(right_ankle_y.min())

    return features


def build_dataset():
    rows = []

    for trick in TRICKS:
        trick_dir = DATA_RAW_DIR / trick
        if not trick_dir.exists():
            print(f"⚠️ Folder does not exist: {trick_dir}")
            continue

        video_files = sorted(
            [p for p in trick_dir.iterdir() if p.suffix.lower() in [".mp4", ".mov", ".avi"]]
        )

        print(f"▶️ Processing {len(video_files)} videos for {trick}...")

        for video_path in video_files:
            feats = extract_video_features(video_path)
            if feats is None:
                continue

            feats["label"] = trick
            feats["video_path"] = str(video_path.relative_to(DATA_RAW_DIR.parent))
            rows.append(feats)

    if not rows:
        print("❌ No data collected, check your videos & paths.")
        return

    df = pd.DataFrame(rows)
    DATASET_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATASET_CSV, index=False)
    print(f"✅ Dataset saved to: {DATASET_CSV}")
    print(df.head())


if __name__ == "__main__":
    build_dataset()
