import numpy as np

from scipy.special import erf

class RandomNumberGenerator():
    def __init__(self):
        np.random.seed(7)

    def integer(high):
        return np.random.randint(0, high-1)

    def uniform(low, high):
        return np.ramdom.uniform(low, high)

    def normal(mu, sigma):
        return np.random.gauss(mu, sigma)

    def categorical(possibility, size=1):
        if size == 1:
            return np.random.choice(possibility, size).item()
        else:
            return np.random.choice(possibility, size)


def linear_forgetting_weights(N, LF):
    assert N >= 0
    assert LF > 0
    if N == 0:
        return np.asarray([])
    if N < LF:
        return np.ones(N)
    ramp = np.linspace(old_div(1.0, N), 1.0, num=N - LF)
    flat = np.ones(LF)
    weights = np.concatenate([ramp, flat], axis=0)
    assert weights.shape == (N,), (weights.shape, N)
    return weights


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

        # XXX: is sorting them necessary anymore?
        # un-sort the mus and sigma
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

# TODO
def GMM1(weights, mus, sigmas, low=None, high=None, q=None, rng=None, size=()):
    """Sample from truncated 1-D Gaussian Mixture Model"""
    weights, mus, sigmas = list(map(np.asarray, (weights, mus, sigmas)))
    assert len(weights) == len(mus) == len(sigmas)
    n_samples = int(np.prod(size))
    # n_components = len(weights)
    if low is None and high is None:
        # -- draw from a standard GMM
        active = np.argmax(rng.multinomial(1, weights, (n_samples,)), axis=1)
        samples = rng.normal(loc=mus[active], scale=sigmas[active])
    else:
        # -- draw from truncated components, handling one-sided truncation
        low = float(low) if low is not None else -float("Inf")
        high = float(high) if high is not None else float("Inf")
        if low >= high:
            raise ValueError("low >= high", (low, high))
        samples = []
        while len(samples) < n_samples:
            active = np.argmax(rng.multinomial(1, weights))
            draw = rng.normal(loc=mus[active], scale=sigmas[active])
            if low <= draw < high:
                samples.append(draw)
    samples = np.reshape(np.asarray(samples), size)
    if q is None:
        return samples
    return np.round(old_div(samples, q)) * q


# TODO
def GMM1_lpdf(samples, weights, mus, sigmas, low=None, high=None, q=None):
    def print_verbose(s, x):
        return print(f"GMM1_lpdf:{s}", x)

    verbose = 0
    samples, weights, mus, sigmas = list(
        map(np.asarray, (samples, weights, mus, sigmas))
    )
    if samples.size == 0:
        return np.asarray([])
    if weights.ndim != 1:
        raise TypeError("need vector of weights", weights.shape)
    if mus.ndim != 1:
        raise TypeError("need vector of mus", mus.shape)
    if sigmas.ndim != 1:
        raise TypeError("need vector of sigmas", sigmas.shape)
    assert len(weights) == len(mus) == len(sigmas)
    _samples = samples
    samples = _samples.flatten()

    if verbose:
        print_verbose("samples", set(samples))
        print_verbose("weights", weights)
        print_verbose("mus", mus)
        print_verbose("sigmas", sigmas)
        print_verbose("low", low)
        print_verbose("high", high)
        print_verbose("q", q)

    if low is None and high is None:
        p_accept = 1
    else:
        p_accept = np.sum(
            weights * (normal_cdf(high, mus, sigmas) - normal_cdf(low, mus, sigmas))
        )

    if q is None:
        dist = samples[:, None] - mus
        mahal = (old_div(dist, np.maximum(sigmas, EPS))) ** 2
        # mahal shape is (n_samples, n_components)
        Z = np.sqrt(2 * np.pi * sigmas ** 2)
        coef = weights / Z / p_accept
        rval = logsum_rows(-0.5 * mahal + np.log(coef))
    else:
        prob = np.zeros(samples.shape, dtype="float64")
        for w, mu, sigma in zip(weights, mus, sigmas):
            if high is None:
                ubound = samples + old_div(q, 2.0)
            else:
                ubound = np.minimum(samples + old_div(q, 2.0), high)
            if low is None:
                lbound = samples - old_div(q, 2.0)
            else:
                lbound = np.maximum(samples - old_div(q, 2.0), low)
            # -- two-stage addition is slightly more numerically accurate
            inc_amt = w * normal_cdf(ubound, mu, sigma)
            inc_amt -= w * normal_cdf(lbound, mu, sigma)
            prob += inc_amt
        rval = np.log(prob) - np.log(p_accept)

    if verbose:
        print_verbose("rval:", dict(list(zip(samples, rval))))

    rval.shape = _samples.shape
    return rval



def normal_cdf(x, mu, sigma):
    top = x - mu
    bottom = np.maximum(np.sqrt(2) * sigma, EPS)
    z = old_div(top, bottom)
    return 0.5 * (1 + erf(z))


def normal_lpdf():
    pass



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


def logsum_rows(x): # logsum
    m = x.max(axis=1)
    return np.log(np.exp(x - m[:, None]).sum(axis=1)) + m


def bin_count(x, weights, min_length):
    ret = [0] * min_length
    for i in range(len(x)):
        ret[(int)x[i]] += weights[i]
    return ret
