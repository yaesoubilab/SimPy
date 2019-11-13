from scipy.optimize import minimize


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

        if abs(u_new(w_star)-u_base(w_star)) > 0.00001:
            return None

        if w_star >= w0:
            return w_star