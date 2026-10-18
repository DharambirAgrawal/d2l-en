import pandas as pd


def missing_report(df):
    """Summarize missing values and column kinds of a DataFrame."""
    # Your implementation here
    pass


df = pd.read_csv('houses.csv')
print(df)
print(missing_report(df))
