import numpy as np


def calc_pmi_for_terms(p_wc, pw, pc):
    x = (pw * pc)
    a = float(p_wc) / float(x)
    a = max(a, 1e-14)
    return np.log(a)
