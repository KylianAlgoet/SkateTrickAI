# 🛹 SkateTrick AI

SkateTrick AI is een AI-prototype dat skateboardtricks herkent op basis van video.
De tool is getraind op eigen skatefootage (kickflip & pop shuvit) en voorspelt niet alleen
welke trick werd uitgevoerd, maar geeft ook kwalitatieve feedback zoals een echte skatecoach.

Naast trick-herkenning berekent het systeem een cleanliness score (hoe clean is de landing)
én genereert het korte, menselijk leesbare coach tips op basis van de analyse.

---

## ✨ Features

- Upload een video met één gelande trick (kickflip of pop shuvit)
- AI voorspelt:
  - welke trick werd uitgevoerd
  - confidence score (hoe zeker het model is)
- Cleanliness score (0–100)
  - gebaseerd op spronghoogte
  - en stabiliteit van de landing
- AI Coach feedback
  - korte tips zoals een skatecoach
  - dynamisch & gevarieerd (niet altijd dezelfde tekst)
  - afhankelijk van trick, confidence en cleanliness
- Realtime webcam mode
  - druk op `T` → laatste ±2 seconden worden geanalyseerd
- Volledig lokaal (geen cloud, geen externe API’s)
- Model getraind met MediaPipe Pose + RandomForest

---

## 🧠 Technische pipeline

### 1. Data
- Eigen skatevideo’s gefilmd (flatground, vaste camerahoek)
- Mappenstructuur:
  - data/raw/kickflip/
  - data/raw/pop_shuvit/

---

### 2. Pose-extractie (MediaPipe Pose)
- Per frame worden 6 belangrijke landmarks gebruikt:
  - left ankle
  - right ankle
  - left knee
  - right knee
  - left hip
  - right hip
- Coördinaten worden:
  - gecentreerd t.o.v. het heup-middenpunt
  - genormaliseerd op basis van de afstand tussen de heupen
- Niet elk frame wordt gebruikt om performance te verbeteren

---

### 3. Feature engineering
Per video wordt één feature-vector opgebouwd met:

- Per landmark (x & y):
  - mean
  - std
  - min
  - max
- Extra features:
  - minimale enkelhoogte (spronghoogte)
  - height_score_raw
  - landing_stability_raw
  - cleanliness_score_raw  
    (combinatie van hoogte + stabiliteit)

---

### 4. Model
- RandomForestClassifier (scikit-learn)
- Eén rij per video (geen frame-per-frame classificatie)
- Model + feature-kolommen opgeslagen in:
  - models/trick_classifier.joblib

---

### 5. Inference & feedback
- Nieuwe video → zelfde feature pipeline
- Output:
  - voorspelde trick
  - confidence score
  - cleanliness score (0–100)
- AI Coach module:
  - zet technische scores om naar begrijpbare feedback
  - meerdere mogelijke tips per situatie (randomized)
  - maakt de AI menselijker en minder “robotic”

---

## 🧑‍🏫 AI Coach feedback (voorbeelden)

- “Goed geland, maar buig dieper door je knieën bij de landing.”
- “Strakke kickflip 👌 Volgende stap: hoger poppen.”
- “Sketchy landing — focus op controle en voeten boven de bolts.”

Dit maakt het prototype niet enkel analytisch, maar ook coachend.

---

## 🛠️ Installatie

```bash
# Python 3.11 gebruiken
py -3.11 -m venv venv
venv\Scripts\activate

pip install -r requirements.txt
