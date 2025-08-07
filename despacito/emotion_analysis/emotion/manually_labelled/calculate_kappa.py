import pandas as pd
from sklearn.metrics import cohen_kappa_score
import os

# File paths (relative to this script's location)
FILES = [
    ('despa_kJQP7kiw5Fk_emotion_labelling_data_eden.csv', 'despa_kJQP7kiw5Fk_emotion_labelling_data_mb.csv'),
    ('seeyou_RgKAFK5djSk_emotion_labelling_data_eden.csv', 'seeyou_RgKAFK5djSk_emotion_labelling_data_mb.csv'),
]

FOLDER = '.'  # Current folder, adjust if needed

def calculate_kappa(file1, file2, folder=FOLDER):
    path1 = os.path.join(folder, file1)
    path2 = os.path.join(folder, file2)
    print(f"\nComparing: {file1} <-> {file2}")
    df1 = pd.read_csv(path1)
    df2 = pd.read_csv(path2)

    # Use comment_id to align
    merged = pd.merge(df1, df2, on='comment_id', suffixes=('_ann1', '_ann2'))
    print(f"  Merged rows: {len(merged)}")

    # Get the last column name for each annotator
    label_col1 = df1.columns[-1]
    label_col2 = df2.columns[-1]

    labels1 = merged[f'{label_col1}_ann1']
    labels2 = merged[f'{label_col2}_ann2']

    # Drop rows where either label is missing (NaN or empty string)
    valid = (labels1.notna()) & (labels2.notna()) & (labels1 != '') & (labels2 != '')
    labels1 = labels1[valid]
    labels2 = labels2[valid]
    print(f"  Valid pairs for kappa: {len(labels1)}")

    kappa = cohen_kappa_score(labels1, labels2)
    print(f"  Cohen's kappa: {kappa:.4f}")
    return kappa

if __name__ == "__main__":
    for f1, f2 in FILES:
        calculate_kappa(f1, f2, folder=FOLDER) 