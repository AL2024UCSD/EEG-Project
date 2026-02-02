import os
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from mne.decoding import CSP

from metabci.brainda.datasets import PhysionetMI
from metabci.brainda.paradigms import MotorImagery


# Config
N_SPLITS = 5
RANDOM_STATE = 42
MI_EVENTS = ["left_hand", "right_hand"]
OUT_CSV = "physionetmi_subject_mean_accuracy.csv"


# Dataset & paradigm
dataset = PhysionetMI()
paradigm = MotorImagery()


# Model
clf = Pipeline([
    ("csp", CSP(n_components=6, log=True, norm_trace=False)),
    ("lda", LinearDiscriminantAnalysis())
])

cv = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)


# Loop subjects
results = []

print(f"Running CSP+LDA on {len(dataset.subjects)} subjects\n")

for subj in dataset.subjects:
    subj_id = f"S{subj:03d}" if isinstance(subj, int) else str(subj)

    try:
        X, y, meta = paradigm.get_data(
            dataset,
            subjects=[subj],
            return_concat=True,
            n_jobs=1
        )

        # Filter MI trials
        mi_mask = meta["event"].isin(MI_EVENTS).to_numpy()
        X_mi = X[mi_mask]
        y_mi = y[mi_mask]

        if len(np.unique(y_mi)) < 2:
            print(f"[SKIP] {subj_id}: only one class")
            continue

        scores = cross_val_score(
            clf,
            X_mi,
            y_mi,
            cv=cv,
            scoring="accuracy"
        )

        mean_acc = scores.mean()
        std_acc = scores.std()

        # Console output
        print(f"{subj_id}:")
        print(f"  folds = {np.round(scores, 3)}")
        print(f"  mean  = {mean_acc:.3f}")
        print(f"  std   = {std_acc:.3f}\n")

        results.append({
            "subject_id": subj_id,
            "mean_accuracy": mean_acc
        })

    except Exception as e:
        print(f"[ERROR] {subj_id}: {type(e).__name__}: {e}\n")


# Save CSV
df = pd.DataFrame(results)
df.to_csv(OUT_CSV, index=False)

print("Saved:", OUT_CSV)
print(df.head())
