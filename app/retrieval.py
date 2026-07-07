from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.preprocessing import clean_text

ARTIFACT_DIR = Path(__file__).resolve().parent.parent / "artifacts"


class SolutionRetriever:
    def __init__(self):
        self.vectorizer: TfidfVectorizer = None
        self.matrix = None
        self.df: pd.DataFrame = None

    def build(self, df: pd.DataFrame):
        self.df = df.copy()
        self.df["clean_text"] = self.df["text"].apply(clean_text)
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        self.matrix = self.vectorizer.fit_transform(self.df["clean_text"])

    def top_resolution_for_issue_type(self, issue_type: str) -> Optional[str]:
        """Fast path: most tickets of a known issue_type share one canonical resolution."""
        subset = self.df[self.df["issue_type"] == issue_type]
        if subset.empty:
            return None
        return subset["resolution"].mode().iloc[0]

    def nearest_similar_tickets(self, raw_text: str, issue_type: str, k: int = 3):
        text = clean_text(raw_text)
        query_vec = self.vectorizer.transform([text])

        subset_mask = self.df["issue_type"] == issue_type
        subset_idx = self.df[subset_mask].index

        if len(subset_idx) == 0:
            return []

        sims = cosine_similarity(query_vec, self.matrix[subset_idx]).flatten()
        top_k_local = sims.argsort()[::-1][:k]
        results = []
        for i in top_k_local:
            row = self.df.iloc[subset_idx[i]]
            results.append({
                "ticket_id": int(row["ticket_id"]),
                "similarity": round(float(sims[i]), 4),
                "resolution": row["resolution"],
            })
        return results

    def save(self, path: Path = ARTIFACT_DIR):
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path / "retriever.joblib")

    @staticmethod
    def load(path: Path = ARTIFACT_DIR) -> "SolutionRetriever":
        return joblib.load(path / "retriever.joblib")
