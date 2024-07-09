import scipy.stats


def scipy_rv_to_string(rv: scipy.stats.rv_continuous | scipy.stats.rv_discrete):
    dist_args_str = ", ".join(f"{x:.3f}" for x in rv.args)
    return f"X ~ {rv.dist.name.title()}({dist_args_str})"
