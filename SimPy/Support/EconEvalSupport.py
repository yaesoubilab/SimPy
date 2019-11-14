from scipy.optimize import minimize
import SimPy.StatisticalClasses as Stat
import SimPy.RandomVariantGenerators as RVG


def inmb_u(d_effect, d_cost):
    """ higher d_effect represents better health """
    return lambda w: w*d_effect - d_cost


def inmb_d(d_effect, d_cost):
    """ higher d_effect represents worse health """
    return lambda w: -w * d_effect - d_cost


def find_intersecting_wtp(w0, u_new, u_base):

    if u_new(w0) > u_base(w0):
        return None

    else:
        f = lambda w: (u_new(w)-u_base(w))**2
        res = minimize(f, (w0))
        w_star = res.x[0]

        if abs(u_new(w_star)-u_base(w_star)) > 0.01:
            return None

        if w_star > w0:
            return w_star
        else:
            return None


def utility_sample_stat(utility, d_cost_samples, d_effect_samples,
                        wtp_random_variate, n_samples, rnd):

    discrete_rnd = RVG.UniformDiscrete(
        l=0, u=len(d_cost_samples))

    samples = []
    for i in range(n_samples):
        j = discrete_rnd.sample(rnd)

        u = utility(d_effect=d_effect_samples[j],
                    d_cost=d_cost_samples[j])

        w = wtp_random_variate.sample(rnd)
        samples.append(u(w))

    return Stat.SummaryStat(name='', data=samples)
