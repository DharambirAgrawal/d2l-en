from submission import coupled_tests
from grader import expect_close


def result():
    return coupled_tests(0.0015, 0.1, 0.01, 0.02)


def test_joint_table():
    """P(D1, D2 | H=0) table"""
    expect_close(result()['table_healthy'], [[0.82, 0.08], [0.08, 0.02]])


def test_table_is_a_distribution():
    """Table entries sum to 1 and marginals equal the false positive rate"""
    t = result()['table_healthy']
    expect_close(sum(sum(row) for row in t), 1.0)
    expect_close(t[1][0] + t[1][1], 0.1)
    expect_close(t[0][1] + t[1][1], 0.1)


def test_posterior_one():
    """P(H=1 | D1=1)"""
    expect_close(result()['posterior_one'], 0.99 * 0.0015 / (0.99 * 0.0015 + 0.1 * 0.9985), rtol=1e-9)


def test_posterior_both():
    """P(H=1 | D1=1, D2=1)"""
    expect_close(result()['posterior_both'], 0.9801 * 0.0015 / (0.9801 * 0.0015 + 0.02 * 0.9985), rtol=1e-9)
