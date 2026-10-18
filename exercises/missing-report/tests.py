import pandas as pd

from submission import missing_report
from grader import expect_close, expect_equal, expect_true


def report():
    return missing_report(pd.read_csv('houses.csv'))


def test_returns_all_keys():
    """Returns the four required keys"""
    r = report()
    expect_true(isinstance(r, dict), 'missing_report must return a dict.')
    expect_equal(sorted(r), ['categorical', 'missing_fraction', 'numeric', 'rows_with_missing'])


def test_missing_fraction():
    """Per-column missing fractions"""
    frac = report()['missing_fraction']
    expect_close(frac['NumRooms'], 2 / 8)
    expect_close(frac['RoofType'], 3 / 8)
    expect_close(frac['Lot'], 1 / 8)
    expect_close(frac['Neighborhood'], 1 / 8)
    expect_close(frac['Price'], 1 / 8)


def test_rows_with_missing():
    """Fraction of rows with any missing value"""
    expect_close(report()["rows_with_missing"], 5 / 8)


def test_column_kinds():
    """Numeric vs. categorical columns"""
    r = report()
    expect_equal(r['numeric'], ['Lot', 'NumRooms', 'Price'])
    expect_equal(r['categorical'], ['Neighborhood', 'RoofType'])


def test_no_missing_values():
    """Works on a complete DataFrame"""
    r = missing_report(pd.DataFrame({'a': [1, 2], 'b': ['x', 'y']}))
    expect_equal(r['missing_fraction'], {'a': 0.0, 'b': 0.0})
    expect_close(r['rows_with_missing'], 0.0)
