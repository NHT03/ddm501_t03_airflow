"""
DDM501 Tutorial 03 — a data pipeline that runs

Five tasks: ingest -> validate -> split -> scale -> report.
"""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from airflow.decorators import dag, task
from airflow.exceptions import AirflowFailException

log = logging.getLogger(__name__)

PROJECT = Path(__file__).resolve().parents[1]
RAW = PROJECT / "data" / "raw" / "wdbc.csv"
STAGING = PROJECT / "data" / "staging"

FEATURES_MIN = 0.0                 # every WDBC measurement is a non-negative size
LABELS = {"M", "B"}
MAX_BAD_FRACTION = 0.05            # above this the extract is not worth using
TEST_FRACTION = 0.20


def run_dir(ds: str) -> Path:
    """One folder per logical date. Re-running a date overwrites its own folder
    and touches nothing else, which is what makes a re-run safe."""
    d = STAGING / ds
    d.mkdir(parents=True, exist_ok=True)
    return d


@dag(
    dag_id="wdbc_pipeline",
    description="Breast cancer extract: ingest, validate, split, scale",
    schedule="@daily",
    start_date=datetime(2026, 8, 20),
    catchup=False,
    max_active_runs=1,
    default_args={
        "retries": 3,
        "retry_delay": timedelta(seconds=10),
        "retry_exponential_backoff": True,
    },
    tags=["ddm501", "tutorial-03"],
)
def wdbc_pipeline():

    @task
    def ingest(ds: str = None) -> dict:
        """Copy the extract into this run's folder and freeze it there.

        Reading the source again in a later task would mean two tasks seeing
        two different files if the source changes mid-run. Snapshot once.
        """
        if not RAW.exists():
            raise AirflowFailException(f"source extract missing: {RAW}")
        frame = pd.read_csv(RAW)
        out = run_dir(ds) / "raw.parquet"
        frame.to_parquet(out, index=False)
        log.info("ingested %d rows, %d columns", len(frame), frame.shape[1])
        # Returned dicts travel as XCom, which is stored in the metadata
        # database. Keep them to counts and paths -- never a DataFrame.
        return {"rows": len(frame), "cols": frame.shape[1], "path": str(out)}

    @task
    def validate(meta: dict, ds: str = None) -> dict:
        """Quarantine bad rows; fail only if too many of them."""
        frame = pd.read_parquet(meta["path"])
        numeric = [c for c in frame.columns if c not in ("sample_id", "diagnosis")]

        problems = pd.DataFrame(index=frame.index)
        problems["null"] = frame[numeric].isna().any(axis=1)
        problems["negative"] = (frame[numeric] < FEATURES_MIN).any(axis=1)
        problems["bad_label"] = ~frame["diagnosis"].isin(LABELS)
        problems["duplicate"] = frame.duplicated(subset="sample_id", keep="first")
        # An area 20x the 99th percentile is a data-entry error, not a tumour.
        cutoff = frame["mean_area"].quantile(0.99) * 20
        problems["outlier"] = frame["mean_area"] > cutoff

        bad = problems.any(axis=1)
        counts = {k: int(v) for k, v in problems.sum().items()}
        fraction = float(bad.mean())
        log.info("validation: %s  (%.2f%% of rows rejected)", counts, fraction * 100)

        clean = frame[~bad]
        rejected = frame[bad]
        rejected.to_parquet(run_dir(ds) / "rejected.parquet", index=False)
        clean_path = run_dir(ds) / "clean.parquet"
        clean.to_parquet(clean_path, index=False)
        (run_dir(ds) / "validation_report.json").write_text(
            json.dumps({"counts": counts, "bad_fraction": fraction,
                        "clean_rows": len(clean)}, indent=2))

        if fraction > MAX_BAD_FRACTION:
            # AirflowFailException stops the run without burning the retries:
            # a malformed file will still be malformed on the third attempt.
            raise AirflowFailException(
                f"{fraction:.1%} of rows rejected, limit is {MAX_BAD_FRACTION:.0%}")
        return {"path": str(clean_path), "clean_rows": len(clean), **counts}

    @task
    def split(meta: dict, ds: str = None) -> dict:
        """Deterministic split by hashing the id -- no random seed involved.

        A seeded shuffle gives the same split only if the rows arrive in the
        same order. Hashing the id gives the same split for a given row
        forever, on any machine, even if tomorrow's extract adds rows.
        """
        frame = pd.read_parquet(meta["path"])

        def bucket(sample_id: str) -> int:
            digest = hashlib.sha256(sample_id.encode()).hexdigest()
            return int(digest[:8], 16) % 100

        is_test = frame["sample_id"].map(bucket) < TEST_FRACTION * 100
        for name, part in (("train", frame[~is_test]), ("test", frame[is_test])):
            part.to_parquet(run_dir(ds) / f"{name}_unscaled.parquet", index=False)
        log.info("split: %d train / %d test", (~is_test).sum(), is_test.sum())
        return {"train": int((~is_test).sum()), "test": int(is_test.sum())}

    @task
    def scale(meta: dict, ds: str = None) -> dict:
        """Fit the scaler on train only, then apply it to both."""
        train = pd.read_parquet(run_dir(ds) / "train_unscaled.parquet")
        test = pd.read_parquet(run_dir(ds) / "test_unscaled.parquet")
        numeric = [c for c in train.columns if c not in ("sample_id", "diagnosis")]

        mean, std = train[numeric].mean(), train[numeric].std().replace(0, 1)
        for name, part in (("train", train), ("test", test)):
            scaled = part.copy()
            scaled[numeric] = (part[numeric] - mean) / std
            scaled.to_parquet(run_dir(ds) / f"{name}.parquet", index=False)

        (run_dir(ds) / "scaler.json").write_text(json.dumps(
            {"mean": mean.round(6).to_dict(), "std": std.round(6).to_dict()}, indent=2))
        log.info("scaled with statistics from %d training rows", len(train))
        return {"scaled_columns": len(numeric), "fitted_on": len(train)}

    @task
    def train_and_log_mlflow(scaling: dict, ds: str = None) -> dict:
        """Train a RandomForest model on scaled train data, evaluate on test,
        and log parameters, metrics, artifacts, and model to MLflow."""
        import os
        import mlflow
        import mlflow.sklearn
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("wdbc_breast_cancer")

        train = pd.read_parquet(run_dir(ds) / "train.parquet")
        test = pd.read_parquet(run_dir(ds) / "test.parquet")

        feature_cols = [c for c in train.columns if c not in ("sample_id", "diagnosis")]
        X_train, y_train = train[feature_cols], train["diagnosis"].map({"B": 0, "M": 1})
        X_test, y_test = test[feature_cols], test["diagnosis"].map({"B": 0, "M": 1})

        n_estimators = 100
        max_depth = 5
        random_state = 42

        clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1]

        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        auc = float(roc_auc_score(y_test, y_prob))

        # Confusion matrix plot
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Benign", "Malignant"], yticklabels=["Benign", "Malignant"])
        plt.title(f"Confusion Matrix ({ds})")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        cm_path = run_dir(ds) / "confusion_matrix.png"
        plt.savefig(cm_path, bbox_inches="tight")
        plt.close()

        with mlflow.start_run(run_name=f"run_{ds}") as run:
            mlflow.log_param("ds", ds)
            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_param("max_depth", max_depth)
            mlflow.log_param("random_state", random_state)
            mlflow.log_param("train_samples", len(X_train))
            mlflow.log_param("test_samples", len(X_test))

            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision", prec)
            mlflow.log_metric("recall", rec)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("roc_auc", auc)

            mlflow.log_artifact(str(cm_path))
            mlflow.log_artifact(str(run_dir(ds) / "scaler.json"))

            # Log model and register it
            mlflow.sklearn.log_model(
                sk_model=clf,
                artifact_path="model",
                registered_model_name="wdbc_breast_cancer_model"
            )

            run_id = run.info.run_id

        log.info("MLflow run logged. Run ID: %s, Accuracy: %.4f, AUC: %.4f", run_id, acc, auc)
        return {"run_id": run_id, "accuracy": acc, "roc_auc": auc}

    @task
    def report(validation: dict, split_info: dict, scaling: dict, ds: str = None) -> str:
        """One line per run, appended to a log the whole pipeline shares."""
        summary = {"ds": ds, **validation, **split_info, **scaling}
        summary.pop("path", None)
        (run_dir(ds) / "summary.json").write_text(json.dumps(summary, indent=2))

        line = json.dumps(summary, sort_keys=True)
        history = STAGING / "history.jsonl"
        kept = [l for l in (history.read_text().splitlines() if history.exists() else [])
                if json.loads(l).get("ds") != ds]
        history.write_text("\n".join(kept + [line]) + "\n")
        log.info("summary: %s", line)
        return line

    ingested = ingest()
    validated = validate(ingested)
    split_info = split(validated)
    scaling = scale(validated)
    split_info >> scaling
    ml_results = train_and_log_mlflow(scaling)
    report(validated, split_info, scaling)


wdbc_pipeline()

