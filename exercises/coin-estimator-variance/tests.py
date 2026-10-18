from submission import simulated_variance, theoretical_variance
from grader import expect_close, expect_true


def test_theoretical():
    """p(1-p)/n"""
    expect_close(theoretical_variance(0.5, 100), 0.0025)
    expect_close(theoretical_variance(0.1, 10), 0.009)


def test_scaling():
    """10x more tosses means 10x smaller variance"""
    expect_close(theoretical_variance(0.3, 100) / theoretical_variance(0.3, 1000), 10.0)


def test_simulation_matches_theory():
    """The simulation is within 10% of theory"""
    for p, n in ((0.5, 50), (0.2, 400)):
        sim = simulated_variance(p, n, 20000, seed=1)
        expect_close(sim, theoretical_variance(p, n), rtol=0.1,
                     message=f'Simulation for p={p}, n={n} is too far from theory.')


def test_seed_reproducible():
    """Same seed, same answer"""
    expect_true(simulated_variance(0.5, 30, 1000, seed=7) == simulated_variance(0.5, 30, 1000, seed=7),
                'Use np.random.default_rng(seed) so results are reproducible.')
