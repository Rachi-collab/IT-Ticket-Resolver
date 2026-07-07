import json
import time
from pathlib import Path

import pandas as pd

from app.classifier import HierarchicalClassifier
from app.retrieval import SolutionRetriever

DATA_PATH = Path(__file__).resolve().parent / "data" / "tickets.csv"
ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"


def main():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    print(f"  {len(df)} tickets, {df['coarse_category'].nunique()} coarse categories, "
          f"{df['issue_type'].nunique()} issue types")

    print("\nTraining hierarchical classifier...")
    clf = HierarchicalClassifier()
    metrics = clf.train(df)
    clf.save(ARTIFACT_DIR)
    print(f"  coarse accuracy: {metrics['coarse_accuracy']:.3f}")
    print(f"  fine accuracy (avg across categories): {metrics['fine_accuracy_avg']:.3f}")

    print("\nBuilding retrieval index...")
    retriever = SolutionRetriever()
    retriever.build(df)
    retriever.save(ARTIFACT_DIR)
    print(f"  indexed {len(df)} tickets for nearest-neighbor solution retrieval")

    # quick end-to-end latency check
    print("\nRunning latency sanity check (100 predictions)...")
    sample_texts = df["text"].sample(min(100, len(df)), random_state=1).tolist()
    start = time.time()
    for t in sample_texts:
        result = clf.predict(t)
        _ = retriever.top_resolution_for_issue_type(result.issue_type)
    elapsed = time.time() - start
    per_ticket_ms = (elapsed / len(sample_texts)) * 1000
    print(f"  avg latency per ticket: {per_ticket_ms:.2f} ms (budget: 2000 ms)")

    metrics["avg_latency_ms"] = round(per_ticket_ms, 2)
    with open(ARTIFACT_DIR / "training_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nArtifacts saved to artifacts/. Training complete.")


if __name__ == "__main__":
    main()
