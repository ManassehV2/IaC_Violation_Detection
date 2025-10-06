# iac_train/data.py
import csv
import ast
import pandas as pd
from datasets import Dataset
from .state import State

def _read_csv_forgiving(csv_path: str) -> pd.DataFrame:
    """
    Try strict parsing first; on failure fall back to a forgiving mode that:
    - uses the python engine,
    - handles escapes,
    - and skips truly bad lines.
    """
    try:
        return pd.read_csv(csv_path)
    except pd.errors.ParserError as e:
        print(f"[CSV] Strict parse failed: {e}")
        print("[CSV] Retrying with python engine, escape handling, and on_bad_lines='skip'...")
        df = pd.read_csv(
            csv_path,
            engine="python",
            sep=",",
            quotechar='"',
            doublequote=True,
            escapechar="\\",
            on_bad_lines="skip",
        )
        print(f"[CSV] Loaded with forgiving parser. Rows: {len(df)}")
        return df

def _coerce_labels_cell(x):
    """
    Convert a single cell to List[str].
    Accepts:
      - already-a-list/tuple/set
      - Python-list-like strings: "['A', 'B']"
      - comma-separated strings: "A, B"
      - scalars -> single-item list
      - NaN/empty -> []
    """
    if isinstance(x, (list, tuple, set)):
        return [str(v) for v in x]
    if pd.isna(x):
        return []
    s = str(x).strip()
    if s == "" or s == "[]":
        return []
    # Try safe parse first
    try:
        v = ast.literal_eval(s)
        if isinstance(v, (list, tuple, set)):
            return [str(u) for u in v]
        # If it's a scalar from literal_eval, wrap it
        return [str(v)]
    except Exception:
        pass
    # Fallback: try eval (keeps compatibility with your original script)
    try:
        v = eval(s, {"__builtins__": {}})
        if isinstance(v, (list, tuple, set)):
            return [str(u) for u in v]
        return [str(v)]
    except Exception:
        pass
    # Fallback: comma-separated
    if "," in s:
        return [t.strip() for t in s.split(",") if t.strip()]
    # Last resort: single label string
    return [s]

def load_dataset(csv_path, fit=False):
    print(f"Loading dataset from {csv_path}...")
    df = _read_csv_forgiving(csv_path)

    # Validate required columns
    if "tf_code" not in df.columns:
        raise ValueError("Missing required column 'tf_code' in the CSV.")
    if "violated_rules" in df.columns:
        label_col = "violated_rules"
        print("Found 'violated_rules' column, converting to binary labels...")
    elif "labels" in df.columns:
        label_col = "labels"
        print("Found 'labels' column, converting to binary labels...")
    else:
        raise ValueError(
            f"Dataset must contain either 'violated_rules' or 'labels' column. "
            f"Found: {df.columns.tolist()}"
        )

    # Coerce text and labels
    df["tf_code"] = df["tf_code"].astype(str)
    df[label_col] = df[label_col].apply(_coerce_labels_cell)

    # Drop rows with non-list labels or with non-string elements
    def _is_valid_label_list(v):
        return isinstance(v, list) and all(isinstance(i, str) for i in v)

    valid_mask = df[label_col].apply(_is_valid_label_list) & df["tf_code"].notna() & (df["tf_code"].str.len() > 0)
    bad_rows = (~valid_mask).sum()
    if bad_rows:
        print(f"[CSV] Dropping {bad_rows} malformed rows after coercion")
        df = df[valid_mask].reset_index(drop=True)

    # Fit/transform multilabel binarizer
    label_lists = df[label_col].tolist()
    if fit:
        State.mlb.fit(label_lists)
        print(f"Label classes: {State.mlb.classes_}")
    labels_binary = State.mlb.transform(label_lists)

    dataset = Dataset.from_dict({
        "tf_code": df["tf_code"].tolist(),
        "labels": labels_binary.tolist()
    })
    print(f"Dataset loaded: {len(dataset)} samples with {labels_binary.shape[1]} label classes")
    return dataset
