from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from app.preprocessing import clean_text

ARTIFACT_DIR = Path(__file__).resolve().parent.parent / "artifacts"


@dataclass
class ClassificationResult:
    coarse_category: str
    issue_type: str
    confidence: float
    should_escalate: bool


class HierarchicalClassifier:
    CONFIDENCE_THRESHOLD = 0.35  # below this, escalate to a human agent

    def __init__(self):
        self.coarse_vectorizer: TfidfVectorizer = None
        self.coarse_model: CalibratedClassifierCV = None
        # one fine-grained vectorizer+model per coarse category
        self.fine_models: Dict[str, Tuple[TfidfVectorizer, CalibratedClassifierCV]] = {}

    def train(self, df: pd.DataFrame) -> dict:
        df = df.copy()
        df["clean_text"] = df["text"].apply(clean_text)
        X_train, X_test, y_train, y_test = train_test_split(
            df["clean_text"], df["coarse_category"], test_size=0.2,
            random_state=42, stratify=df["coarse_category"]
        )
        self.coarse_vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        Xt = self.coarse_vectorizer.fit_transform(X_train)
        Xv = self.coarse_vectorizer.transform(X_test)

        base_svc = LinearSVC()
        self.coarse_model = CalibratedClassifierCV(base_svc, cv=3)
        self.coarse_model.fit(Xt, y_train)
        coarse_acc = accuracy_score(y_test, self.coarse_model.predict(Xv))
        fine_accs = []
        for category, group in df.groupby("coarse_category"):
            if group["issue_type"].nunique() < 2:
                # only one issue type in this category -> trivial, no model needed
                self.fine_models[category] = (None, group["issue_type"].iloc[0])
                continue

            gtrain, gtest = train_test_split(
                group, test_size=0.2, random_state=42, stratify=group["issue_type"]
            )
            vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
            Xt_f = vec.fit_transform(gtrain["clean_text"])
            Xv_f = vec.transform(gtest["clean_text"])

            model = CalibratedClassifierCV(LinearSVC(), cv=3)
            model.fit(Xt_f, gtrain["issue_type"])
            acc = accuracy_score(gtest["issue_type"], model.predict(Xv_f))
            fine_accs.append(acc)

            self.fine_models[category] = (vec, model)

        overall_fine_acc = float(np.mean(fine_accs)) if fine_accs else 1.0

        return {
            "coarse_accuracy": float(coarse_acc),
            "fine_accuracy_avg": overall_fine_acc,
            "n_coarse_categories": df["coarse_category"].nunique(),
            "n_issue_types": df["issue_type"].nunique(),
            "n_training_rows": len(df),
        }

    def predict(self, raw_text: str) -> ClassificationResult:
        text = clean_text(raw_text)
        vec_input = self.coarse_vectorizer.transform([text])

        coarse_probs = self.coarse_model.predict_proba(vec_input)[0]
        coarse_idx = int(np.argmax(coarse_probs))
        coarse_category = self.coarse_model.classes_[coarse_idx]
        coarse_conf = float(coarse_probs[coarse_idx])

        fine_vec, fine_model = self.fine_models.get(coarse_category, (None, None))

        if fine_vec is None:
            # trivial single-issue-type category
            issue_type = fine_model if isinstance(fine_model, str) else "unknown"
            fine_conf = coarse_conf
        else:
            Xf = fine_vec.transform([text])
            fine_probs = fine_model.predict_proba(Xf)[0]
            fine_idx = int(np.argmax(fine_probs))
            issue_type = fine_model.classes_[fine_idx]
            fine_conf = float(fine_probs[fine_idx])

        overall_confidence = coarse_conf * fine_conf
        should_escalate = overall_confidence < self.CONFIDENCE_THRESHOLD

        return ClassificationResult(
            coarse_category=coarse_category,
            issue_type=issue_type,
            confidence=round(overall_confidence, 4),
            should_escalate=should_escalate,
        )

    def save(self, path: Path = ARTIFACT_DIR):
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path / "classifier.joblib")

    @staticmethod
    def load(path: Path = ARTIFACT_DIR) -> "HierarchicalClassifier":
        return joblib.load(path / "classifier.joblib")
