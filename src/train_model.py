import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from joblib import dump
from config import DATASET_CSV, MODEL_PATH

def train():
    df = pd.read_csv(DATASET_CSV)

    X = df.drop(columns=["label", "video_path"])
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print("=== Classification report ===")
    print(classification_report(y_test, y_pred))

    print("=== Confusion matrix ===")
    print(confusion_matrix(y_test, y_pred, labels=sorted(y.unique())))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    dump({"model": clf, "feature_columns": X.columns.tolist()}, MODEL_PATH)
    print(f"✅ Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    train()
