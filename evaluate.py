import time

from app.classifier import HierarchicalClassifier
from app.retrieval import SolutionRetriever
from data.generate_data import generate

N_EVAL = 500
NOISY_FRACTION = 0.5  # stress-test above the stated 30% to be conservative


def main():
    clf = HierarchicalClassifier.load()
    _ = SolutionRetriever.load()  # ensures retriever loads too

    eval_rows = generate(n_rows=N_EVAL, noisy_fraction=NOISY_FRACTION)

    correct_coarse = 0
    correct_fine = 0
    escalated = 0
    auto_resolved_correct = 0
    latencies = []

    for row in eval_rows:
        start = time.time()
        result = clf.predict(row["text"])
        latencies.append((time.time() - start) * 1000)

        if result.coarse_category == row["coarse_category"]:
            correct_coarse += 1
        if result.issue_type == row["issue_type"]:
            correct_fine += 1

        if result.should_escalate:
            escalated += 1
        else:
            if result.issue_type == row["issue_type"]:
                auto_resolved_correct += 1

    n = len(eval_rows)
    auto_resolved = n - escalated
    auto_resolved_accuracy = auto_resolved_correct / auto_resolved if auto_resolved else 0.0

    print(f"Evaluation set: {n} tickets ({NOISY_FRACTION*100:.0f}% noisy, stress test)")
    print(f"Coarse category accuracy (all tickets):  {correct_coarse / n:.3f}")
    print(f"Fine issue-type accuracy (all tickets):   {correct_fine / n:.3f}")
    print(f"Escalated to human: {escalated}/{n} ({escalated/n*100:.1f}%)")
    print(f"Auto-resolved: {auto_resolved}/{n} ({auto_resolved/n*100:.1f}%)")
    print(f"Accuracy on auto-resolved subset (the metric that matters "
          f"for the >=80% requirement): {auto_resolved_accuracy:.3f}")
    print(f"Avg latency: {sum(latencies)/len(latencies):.2f} ms (budget: 2000 ms)")
    print(f"P95 latency: {sorted(latencies)[int(len(latencies)*0.95)]:.2f} ms")


if __name__ == "__main__":
    main()
