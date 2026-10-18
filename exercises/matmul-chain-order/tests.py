from submission import chain_cost
from grader import expect_equal, expect_raises


def test_small_example():
    """Shapes (2, 3), (3, 4), (4, 5)"""
    r = chain_cost((2, 3), (3, 4), (4, 5))
    expect_equal(r['(AB)C'], (2 * 3 * 4 + 2 * 4 * 5, 2 * 4))
    expect_equal(r['A(BC)'], (3 * 4 * 5 + 2 * 3 * 5, 3 * 5))
    expect_equal(r['best'], '(AB)C')


def test_exercise_shapes():
    """Exercise 10: (AB)C is far cheaper"""
    r = chain_cost((2**10, 2**16), (2**16, 2**5), (2**5, 2**14))
    expect_equal(r['(AB)C'], (2**31 + 2**29, 2**15))
    expect_equal(r['A(BC)'], (2**35 + 2**40, 2**30))
    expect_equal(r['best'], '(AB)C')


def test_other_order_wins():
    """Sometimes A(BC) is cheaper"""
    expect_equal(chain_cost((100, 2), (2, 100), (100, 1))['best'], 'A(BC)')


def test_tie_prefers_ab_c():
    """On a tie, pick (AB)C"""
    expect_equal(chain_cost((2, 2), (2, 2), (2, 2))['best'], '(AB)C')


def test_mismatched_shapes():
    """Mismatched inner dimensions raise ValueError"""
    expect_raises(ValueError, chain_cost, (2, 3), (4, 5), (5, 6))
