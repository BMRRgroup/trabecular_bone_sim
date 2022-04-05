import numpy as np

def gss(f, a, b, tol=1e-5):
    """
    golden section search
    to find the minimum of f on [a,b]
    https://en.wikipedia.org/wiki/Golden-section_search

    f: a strictly unimodal function on [a,b]

    example:
    >> f = lambda x: (x-2)**2
    >> x = gss(f, 1, 5)
    >> x
    2.000009644875678
    """
    goldenratio = (1 + np.sqrt(5)) / 2
    c = b - (b - a) / goldenratio
    d = a + (b - a) / goldenratio
    if isinstance(a, (list, tuple, np.ndarray)):
        updateHere = np.abs(c - d) > tol
        while updateHere.any():
            b = np.where(updateHere & (f(c) < f(d)), d, b)
            a = np.where(updateHere & (f(c) >= f(d)), c, a)
            c = b - (b - a) / goldenratio
            d = a + (b - a) / goldenratio
            updateHere = np.abs(c - d) > tol
    else:
        while abs(c - d) > tol:
            if f(c) < f(d):
                b = d
            else:
                a = c
            c = b - (b - a) / goldenratio
            d = a + (b - a) / goldenratio
    return (b + a) / 2


if __name__ == '__main__':
    print(gss(lambda x: (x-2)**2, 1, 5))

    print(gss(lambda x: (x-2)**2,
              np.array([1, 1]),
              np.array([5, 5])))
