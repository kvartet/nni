import numpy as np

from scipy.special import erf


class RandomNumberGenerator():
    def __init__(self):
        np.random.seed(7)

    def integer(self, high):
        return np.random.randint(0, high)

    def uniform(self, low, high):
        return np.ramdom.uniform(low, high)

    def normal(self, mu, sigma):
        return np.random.gauss(mu, sigma)

    def categorical(self, possibility, size=1):
        return np.argmax(np.random.multinomial(1, possibility, (size,)), axis=1)[0] if size==1 else np.argmax(np.random.multinomial(1, possibility, (size,)), axis=1)


def linear_forgetting_weights(n, lf):
    weights = [1.0] * n
    ramp_start = 1.0 / n
    ramp_length= n - lf
    if ramp_length == 1:
        weights[0] = ramp_start
    else:
        for i in range(ramp_length):
            weights[i] = ramp_start + (1.0 - ramp_start) / (ramp_length - 1) * i
    return np.asarray(weights)



def adaptive_parzen_normal(mus, prior_weight, prior_mu, prior_sigma):

    mus_orig = np.array(mus)
    mus = np.array(mus)
    assert str(mus.dtype) != "object"

    if mus.ndim != 1:
        raise TypeError("mus must be vector", mus)
    if len(mus) == 0:
        mus = np.asarray([prior_mu])
        sigma = np.asarray([prior_sigma])
    elif len(mus) == 1:
        mus = np.asarray([prior_mu] + [mus[0]])
        sigma = np.asarray([prior_sigma, prior_sigma * 0.5])
    elif len(mus) >= 2:
        order = np.argsort(mus)
        mus = mus[order]
        sigma = np.zeros_like(mus)
        sigma[1:-1] = np.maximum(mus[1:-1] - mus[0:-2], mus[2:] - mus[1:-1])
        if len(mus) > 2:
            lsigma = mus[2] - mus[0]
            usigma = mus[-1] - mus[-3]
        else:
            lsigma = mus[1] - mus[0]
            usigma = mus[-1] - mus[-2]

        sigma[0] = lsigma
        sigma[-1] = usigma

        mus[order] = mus.copy()
        sigma[order] = sigma.copy()

        if not np.all(mus_orig == mus):
            print("orig", mus_orig)
            print("mus", mus)
        assert np.all(mus_orig == mus)

        # put the prior back in
        mus = np.asarray([prior_mu] + list(mus))
        sigma = np.asarray([prior_sigma] + list(sigma))

    maxsigma = prior_sigma
    # -- magic formula:
    minsigma = old_div(prior_sigma, np.sqrt(1 + len(mus)))

    sigma = np.clip(sigma, minsigma, maxsigma)

    weights = np.ones(len(mus), dtype=mus.dtype)
    weights[0] = prior_weight

    weights = old_div(weights, weights.sum())

    return weights, mus, sigma


def Gmm1(weights, mus, sigmas, low, high, log, integer, n_ei_candidates, rng):
    samples = [0] * n_ei_candidates
    for i in range(n_ei_candidates):
        while True:
            active = rng.categorical(weights)
            draw = rng.normal(mus[active], sigmas[active])
            if draw < low or draw >= high:
                continue
            if log:
                draw = np.exp(draw)
            if integer:
                draw = np.round(draw)
            samples[i] = draw
            break
        return np.asarray(samples)

def Gmm1_lpdf(samples, weights, mus, sigmas, low, high, log, integer, rng):
    print('unuseless')
    return weights
    # print()
    # samples, weights, mus, sigmas = list(
    #     map(np.asarray, (samples, weights, mus, sigmas))
    # )

    # assert len(weights) == len(mus) == len(sigmas)
    # samples = _samples.flatten()

    # p_accept = np.sum(
    #     weights * (normal_cdf(high, mus, sigmas) -
    #                 normal_cdf(low, mus, sigmas))
    # )

    # # ret = [0] * len(samples)
    # # for i in range(len(samples)):
    # if not integer:
    #     if log:
    #         ret = logsum_rows(lognormal_lpdf(samples, weights, mus, sigmas))
    #     else:
    #         ret = logsum_rows(lognormal_lpdf(samples, weights, mus, sigmas, p_accept))
    # else:
    #     prob = 0
    #     if log:
    #         ubound = np.log(np.minimum(samples + 0.5, np.exp(high)))
    #         lbound = np.log(np.maximum(samples - 0.5, np.exp(low)))
    #     else:
    #         ubound = np.minimum(samples + 0.5, high)
    #         lobund = np.maximum(samples - 0.5, low)
    #     prob += np.sum(weights * normal_cdf(ubound, mus, sigmas))
    #     prob -= np.sum(weights * normal_cdf(lbound, mus, sigmas)
    #     ret = np.log(prob) - np.log(p_accept)

    # return ret   


# def Gmm1_lpdf(samples, weights, mus, sigmas, low, high, log, integer, rng):
#     samples, weights, mus, sigmas = list(
#         map(np.asarray, (samples, weights, mus, sigmas))
#     )

#     assert len(weights) == len(mus) == len(sigmas)
#     _samples = samples
#     samples = _samples.flatten()

#     p_accept = np.sum(
#         weights * (normal_cdf(high, mus, sigmas) -
#                     normal_cdf(low, mus, sigmas))
#     )

#     ret = [0] * len(samples)
#     for i in range(len(samples)):
#         if not integer:
#             if log:
#                 ret[i] = logsum_rows(lognormal_lpdf(samples[i], weights, mus, sigmas))
#             else:
#                 ret[i] = logsum_rows(lognormal_lpdf(samples[i], weights, mus, sigmas, p_accept))
#         else:
#             prob = 0
#             if log:
#                 ubound = np.log(np.min(samples[i] + 0.5, np.exp(high)))
#                 lbound = np.log(np.max(samples[i] - 0.5, np.exp(low)))
#             else:
#                 ubound = np.min(samples[i] + 0.5, high)
#                 lobund = np.max(samples[i] - 0.5, low)
#             prob += np.sum(weights * normal_cdf(ubound, mus, sigmas))
#             prob -= np.sum(weights * normal_cdf(lbound, mus, sigmas)
#             ret[i] = np.log(prob) - np.log(p_accept)

#     return np.asarray(ret)    




def normal_cdf(x, mu, sigma):
    top = x - np.asarray(mu)
    bottom = np.maximum(np.sqrt(2) * sigma, EPS)
    z = 0.5 + 0.5 * erf(top/bottom)
    return 0.5 * (1 + erf(z))


def lognormal_lpdf(x, mu, sigma):
    # formula copied from wikipedia
    # http://en.wikipedia.org/wiki/Log-normal_distribution
    assert np.all(sigma >= 0)
    sigma = np.maximum(sigma, EPS)
    Z = sigma * x * np.sqrt(2 * np.pi)
    E = 0.5 * (old_div((np.log(x) - mu), sigma)) ** 2
    rval = -E - np.log(Z)
    return rval


def qlognormal_lpdf(x, mu, sigma, q):
    # casting rounds up to nearest step multiple.
    # so lpdf is log of integral from x-step to x+1 of P(x)

    # XXX: subtracting two numbers potentially very close together.
    return np.log(lognormal_cdf(x, mu, sigma) - lognormal_cdf(x - q, mu, sigma))


def logsum_rows(x):  # logsum
    m = x.max(axis=1)
    return np.log(np.exp(x - m[:, None]).sum(axis=1)) + m


def bin_count(x, weights, min_length):
    ret = [0] * min_length
    for i in range(len(x)):
        ret[x[i]] += weights[i]
    return np.asarray(ret)
