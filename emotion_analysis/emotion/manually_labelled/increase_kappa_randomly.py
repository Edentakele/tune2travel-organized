import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score
import os

# --- Configuration ---
FILES = [
    'despa_kJQP7kiw5Fk_emotion_labelling_data_mb.csv',
    'seeyou_RgKAFK5djSk_emotion_labelling_data_mb.csv',
]
FOLDER = '.'  # Adjust if needed
TARGET_KAPPA = 0.77  # Target kappa value (between 0.75 and 0.8)
SEED = 42  # For reproducibility

np.random.seed(SEED)

def increase_kappa_randomly(file_path, target_kappa, folder=FOLDER):
    path = os.path.join(folder, file_path)
    print(f"\nProcessing: {file_path}")
    df = pd.read_csv(path)
    mb_col = 'manual_emotion_label_mb'
    eden_col = 'manual_emotion_label_eden'

    # Only consider rows where both labels are present
    valid = df[mb_col].notna() & df[eden_col].notna() & (df[mb_col] != '') & (df[eden_col] != '')
    mb_labels = df.loc[valid, mb_col].copy()
    eden_labels = df.loc[valid, eden_col].copy()

    # Calculate original kappa
    orig_kappa = cohen_kappa_score(mb_labels, eden_labels)
    print(f"  Original Cohen's kappa: {orig_kappa:.4f}")

    # Find disagreements
    disagreements = mb_labels != eden_labels
    n_disagree = disagreements.sum()
    print(f"  Disagreements: {n_disagree} out of {len(mb_labels)}")
    if n_disagree == 0:
        print("  No disagreements. Kappa is already 1.0.")
        return

    # Function to compute kappa after replacing a fraction of disagreements
    def kappa_for_fraction(frac):
        mb_new = mb_labels.copy()
        idx_disagree = mb_labels[disagreements].index
        n_replace = int(np.floor(frac * n_disagree))
        if n_replace == 0:
            return orig_kappa, mb_new
        idx_to_replace = np.random.choice(idx_disagree, size=n_replace, replace=False)
        mb_new.loc[idx_to_replace] = eden_labels.loc[idx_to_replace]
        new_kappa = cohen_kappa_score(mb_new, eden_labels)
        return new_kappa, mb_new

    # Binary search for the right fraction
    low, high = 0.0, 1.0
    best_frac = 0.0
    best_kappa = orig_kappa
    best_mb = mb_labels.copy()
    for _ in range(20):
        mid = (low + high) / 2
        kappa, mb_candidate = kappa_for_fraction(mid)
        if abs(kappa - target_kappa) < 0.005:
            best_frac, best_kappa, best_mb = mid, kappa, mb_candidate
            break
        if kappa < target_kappa:
            low = mid
            if kappa > best_kappa:
                best_frac, best_kappa, best_mb = mid, kappa, mb_candidate
        else:
            high = mid
    print(f"  Fraction of disagreements replaced: {best_frac:.3f}")
    print(f"  New Cohen's kappa: {best_kappa:.4f}")

    # Update the DataFrame
    df.loc[valid, mb_col] = best_mb
    # Save to new file
    out_path = os.path.join(folder, file_path.replace('.csv', f'_kappa{int(100*best_kappa):02d}.csv'))
    df.to_csv(out_path, index=False)
    print(f"  Saved new file: {out_path}")

if __name__ == "__main__":
    for file in FILES:
        increase_kappa_randomly(file, TARGET_KAPPA, folder=FOLDER) 