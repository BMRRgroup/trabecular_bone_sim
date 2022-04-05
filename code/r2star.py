from gss import gss
from scipy.optimize import curve_fit  # not threadsafe, https://stackoverflow.com/questions/22612213/python-threading-curve-fit-null-argument-to-internal-routine#22633971
import numpy as np

def fit_monoexpdecay_gss(x, y, bmin, bmax):
    """
    solve y = exp(-b x) via simple golden section search (gss)
    """
    def f(b):
        return np.exp(-b[..., np.newaxis] * x)

    def chi2(b):
        y_f = (y * f(b)).sum(axis=-1)
        f_f = (f(b)**2).sum(axis=-1)
        return np.real(-y_f.conj() * y_f / f_f)

    if y.ndim > 1:
        bmin = bmin * np.ones(y.shape[:-1])
        bmax = bmax * np.ones(y.shape[:-1])

    return gss(chi2, bmin, bmax)


def Lorentzian(x, I, g, x0):
    return I * g**2 / ((x-x0)**2 + g**2)


def fit_Lorentzian(xdata, ydata, p0=None):
    if p0 == None:
        I_guess = max(ydata)
        g_guess = 1 / (max(ydata))
        x0_guess = sum(xdata * ydata) / sum(ydata)

    p0 = (I_guess, g_guess, x0_guess)
    popt, pcov = curve_fit(Lorentzian, xdata, ydata, p0=p0)
    I, g, x0 = popt[0], popt[1], popt[2]

    return I, g, x0, pcov

def estimate_r2prime(rdf, mask):
    hist = np.histogram(rdf[~mask], bins='auto')
    xdata = hist[1][1:]
    ydata = hist[0]
    I, g, x0, pcov = fit_Lorentzian(xdata, ydata)
    return I, g, x0, pcov
