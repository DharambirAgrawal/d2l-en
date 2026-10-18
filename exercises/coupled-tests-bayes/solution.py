def coupled_tests(prior, fpr, fnr, both_pos_healthy):
    """Joint table for healthy patients and posteriors after one/two positive tests."""
    one_pos_only = fpr - both_pos_healthy
    table = [[1 - both_pos_healthy - 2 * one_pos_only, one_pos_only],
             [one_pos_only, both_pos_healthy]]
    tpr = 1 - fnr
    one = tpr * prior / (tpr * prior + fpr * (1 - prior))
    both_sick = tpr * tpr
    both = both_sick * prior / (both_sick * prior + both_pos_healthy * (1 - prior))
    return {'table_healthy': table, 'posterior_one': one, 'posterior_both': both}


print(coupled_tests(0.0015, 0.1, 0.01, 0.02))
