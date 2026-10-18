import pandas as pd


def missing_report(df):
    """Summarize missing values and column kinds of a DataFrame."""
    missing = df.isna()
    numeric = sorted(c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]))
    return {
        'missing_fraction': {c: float(missing[c].mean()) for c in df.columns},
        'rows_with_missing': float(missing.any(axis=1).mean()),
        'numeric': numeric,
        'categorical': sorted(c for c in df.columns if c not in numeric),
    }


df = pd.read_csv('houses.csv')
print(df)
print(missing_report(df))
