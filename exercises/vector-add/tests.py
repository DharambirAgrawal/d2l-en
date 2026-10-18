from submission import vector_add
from grader import expect_close, expect_equal, expect_raises, expect_type


def test_example():
    """[1, 2] + [3, 4] = [4, 6]"""
    expect_equal(vector_add([1, 2], [3, 4]), [4, 6])


def test_returns_a_list():
    """The result is a list"""
    expect_type(vector_add([1, 2], [3, 4]), list)


def test_negative_numbers():
    """Works with negative numbers"""
    expect_equal(vector_add([-1, 5, -3], [2, -5, -4]), [1, 0, -7])


def test_floats():
    """Works with floats (compared with a tolerance)"""
    expect_close(vector_add([0.1, 2.5, -1.25], [0.2, 0.5, 1.0]), [0.3, 3.0, -0.25])


def test_empty_vectors():
    """[] + [] = []"""
    expect_equal(vector_add([], []), [])


def test_longer_vector():
    """Works for a 1000-element vector"""
    a = list(range(1000))
    b = [2 * x for x in a]
    expect_equal(vector_add(a, b), [3 * x for x in a])


def test_mismatched_lengths():
    """Different lengths raise ValueError"""
    expect_raises(ValueError, vector_add, [1, 2], [1])
    expect_raises(ValueError, vector_add, [], [1])


def test_inputs_unchanged():
    """a and b are not modified"""
    a, b = [1, 2], [3, 4]
    vector_add(a, b)
    expect_equal((a, b), ([1, 2], [3, 4]), 'vector_add modified its inputs.')
