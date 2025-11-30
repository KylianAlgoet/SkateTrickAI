# 🛹 SkateTrick AI

SkateTrick AI is een prototype AI-tool die skateboardtricks herkent op basis van video.
De tool is getraind op eigen skatefootage (kickflip & pop shuvit) en geeft naast de trick
ook een **cleanliness score** (hoe “clean” de trick werd geland).

## ✨ Features

- Upload een video met één trick (kickflip of pop shuvit)
- AI voorspelt welke trick het is + confidence score
- Extra **cleanliness score (0–100)** op basis van spronghoogte en landingsstabiliteit
- Realtime mode met webcam: druk op `T` en de laatste ~2s worden geanalyseerd
- Model getraind op eigen dataset met MediaPipe Pose + RandomForest classifier

## 🧠 Technische pipeline

1. **Data**  
   - Eigen videodata gefilmd (zelfde camerahoek, flatground)  
   - Mappenstructuur:
     - `data/raw/pop_shuvit/`
     - `data/raw/kickflip/`

2. **Pose-extractie (MediaPipe Pose)**  
   - Voor elk frame worden de 6 belangrijkste landmarks gebruikt:
     - left/right ankle, knee, hip  
   - Coördinaten worden:
     - genormaliseerd t.o.v. het heup-middenpunt  
     - geschaald op basis van de afstand tussen de heupen  

3. **Feature engineering**  
   - Per landmark: mean, std, min, max voor x en y  
   - Extra features:
     - minimale enkelhoogte (spronghoogte)
     - **height_score_raw**
     - **landing_stability_raw**
     - **cleanliness_score_raw** = combinatie van height + stabiliteit

4. **Model**  
   - `RandomForestClassifier` (scikit-learn)
   - Getraind op het feature-vector per video (één rij per clip)
   - Model + feature-columns opgeslagen in `models/trick_classifier.joblib`

5. **Inference**  
   - Nieuwe video → zelfde feature pipeline → model.predict + predict_proba  
   - Cleanliness wordt getoond als 0–100 score

## 🛠️ Installatie

```bash
# Python 3.11 gebruiken!
py -3.11 -m venv venv
venv\Scripts\activate

pip install -r requirements.txt
