"""Evaluate integrity engine."""

from pathlib import Path

from app import enums
from app.db.session import SessionLocal
from app.ml.integrity import evaluate_integrity, train_anomaly_model


def main():
    print("Training anomaly model...")
    session = SessionLocal()
    train_anomaly_model(session)

    print("Evaluating against synthetic population...")
    import sqlite3

    conn = sqlite3.connect("sme_credit.db")
    eval_rows = conn.execute(
        "SELECT business_id, application_id, is_synthetic_fraud FROM loan_evaluations"
    ).fetchall()

    y_true = []
    y_pred = []
    y_high = []

    for row in eval_rows:
        b_id, app_id, is_fraud = row
        y_true.append(is_fraud)

        try:
            res = evaluate_integrity(app_id, session=session)
            flags = res.get("flags", [])
            has_flag = len(flags) > 0
            has_high_flag = any(f["severity"] == enums.Severity.HIGH for f in flags)

            y_pred.append(1 if has_flag else 0)
            y_high.append(1 if has_high_flag else 0)
        except Exception as e:
            print(f"Error on app {app_id}: {e}")
            y_pred.append(0)
            y_high.append(0)

    if not y_true:
        print("No synthetic data found.")
        return

    true_pos = sum(1 for t, p in zip(y_true, y_pred, strict=False) if t == 1 and p == 1)
    false_pos = sum(1 for t, p in zip(y_true, y_pred, strict=False) if t == 0 and p == 1)
    false_neg = sum(1 for t, p in zip(y_true, y_pred, strict=False) if t == 1 and p == 0)
    true_neg = sum(1 for t, p in zip(y_true, y_pred, strict=False) if t == 0 and p == 0)

    precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
    recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
    far = false_pos / (false_pos + true_neg) if (false_pos + true_neg) > 0 else 0

    # High severity only metrics
    true_pos_h = sum(1 for t, p in zip(y_true, y_high, strict=False) if t == 1 and p == 1)
    false_pos_h = sum(1 for t, p in zip(y_true, y_high, strict=False) if t == 0 and p == 1)
    false_neg_h = sum(1 for t, p in zip(y_true, y_high, strict=False) if t == 1 and p == 0)
    true_neg_h = sum(1 for t, p in zip(y_true, y_high, strict=False) if t == 0 and p == 0)

    precision_h = true_pos_h / (true_pos_h + false_pos_h) if (true_pos_h + false_pos_h) > 0 else 0
    recall_h = true_pos_h / (true_pos_h + false_neg_h) if (true_pos_h + false_neg_h) > 0 else 0
    far_h = false_pos_h / (false_pos_h + true_neg_h) if (false_pos_h + true_neg_h) > 0 else 0

    report = f"""
### Validation — Integrity Engine
- **Any Flag Performance**: Precision {precision:.3f}, Recall {recall:.3f}, FAR {far:.3f}
- **High Severity Performance**: Precision {precision_h:.3f}, Recall {recall_h:.3f}, FAR {far_h:.3f}
- True Positives (High): {true_pos_h}
- False Positives (High): {false_pos_h}
- True Negatives: {true_neg_h}
- False Negatives (High): {false_neg_h}
"""
    print(report)

    # Append to VALIDATION.md
    val_path = Path("docs/VALIDATION.md")
    if val_path.exists():
        with open(val_path, "a") as f:
            f.write(report)


if __name__ == "__main__":
    main()
