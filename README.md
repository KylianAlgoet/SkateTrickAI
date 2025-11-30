# Skate AI – Trick Classifier (Ollie / Pop Shuvit / Kickflip)

Simple experiment to classify flatground skateboard tricks from short videos
using pose estimation (MediaPipe) + a RandomForest classifier.

## Project structure

```text
skate_ai/
  config.py
  requirements.txt
  README.md
  src/
    __init__.py
    extract_features.py
    train_model.py
    predict_video.py
  data/
    raw/
      ollie/
      pop_shuvit/
      kickflip/
  models/
```

## Setup

```bash
# create & activate venv (Windows)
python -m venv venv
venv\Scripts\activate

# or (macOS/Linux)
python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

## Add your videos

Put your flatground videos here (one trick per video, same camera angle):

```text
data/raw/ollie/ollie_01.mp4
data/raw/ollie/ollie_02.mp4
...
data/raw/pop_shuvit/pop_01.mp4
...
data/raw/kickflip/kickflip_01.mp4
...
```

## Build dataset

```bash
python -m src.extract_features
```

This creates `data/dataset.csv` with one feature row per video.

## Train model

```bash
python -m src.train_model
```

This trains a RandomForest classifier and saves it to `models/trick_classifier.joblib`.

## Predict on a new video

```bash
python -m src.predict_video data/raw/kickflip/kickflip_01.mp4
```

You should see a prediction like:

```text
🎯 Predicted trick: kickflip (confidence: 0.93)
```
