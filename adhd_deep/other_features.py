import numpy as np


def fd_higuchi(x, kmax="Nope"):
    """
    ---------------------------------------------------------------------
     Higuchi estimate in [1]
    ---------------------------------------------------------------------
    """
    N = len(x)
    if kmax == "Nope":
        kmax = int(np.floor(len(x) / 10))

    # what value of k to compute
    ik = 1
    k_all = np.empty(0)
    knew = 0

    while knew < kmax:
        if ik <= 4:
            knew = ik
        else:
            knew = np.floor(2 ** ((ik + 5) / 4))

        if knew <= kmax:
            k_all = np.append(k_all, knew)
        ik = ik + 1

    """
    ---------------------------------------------------------------------
    curve length for each vector:
    ---------------------------------------------------------------------
    """

    inext = 0
    L_avg = np.zeros(len(k_all))

    for k in k_all:
        L = np.zeros(k.astype(int))
        for m in range(k.astype(int)):
            ik = np.array(range(1, np.floor((N - m - 1) / k).astype(int) + 1))
            scale_factor = (N - 1) / (np.floor((N - m - 1) / k) * k)
            L[m] = np.nansum(
                np.abs(
                    x[m + np.array(ik) * k.astype(int)]
                    - x[m + (ik - 1) * k.astype(int)]
                )
            ) * (scale_factor / k)

        L_avg[inext] = np.nanmean(L)
        inext += 1

    """
    -------------------------------------------------------------------
     form log-log plot of scale v. curve length
    -------------------------------------------------------------------
    """

    x1 = np.log2(k_all)
    y1 = np.log2(L_avg)
    c = np.polyfit(x1, y1, 1)
    FD = -c[0]

    # y_fit = c[0]*x1 + c[1]
    # y_residuals = y1 - y_fit
    # r2 = 1 - np.sum(y_residuals**2) / ((N-1) * np.var(y1))

    # return FD, r2, k_all, L_avg
    return FD


def fd_katz(x, dum=0):

    """
    ---------------------------------------------------------------------
    Katz estimate in [2]
    ---------------------------------------------------------------------
    """
    N = len(x)
    p = N - 1

    # 1. line-length
    L = np.empty(N - 1)
    for n in range(N - 1):
        L[n] = np.sqrt(1 + (x[n] - x[n + 1]) ** 2)

    L = np.sum(L)

    # 2. maximum distance
    d = np.zeros(p)
    for n in range(N - 1):
        d[n] = np.sqrt((n + 1) ** 2 + (x[0] - x[n + 1]) ** 2)

    d = np.max(d)

    D = np.log(p) / (np.log(d / L) + np.log(p))

    return D


######################################"""pyeeeg"""############################################

import numpy


def hfd(X, Kmax):
    """ Compute Higuchi Fractal Dimension of a time series X. kmax
     is an HFD parameter
    """
    L = []
    x = []
    N = len(X)
    for k in range(1, Kmax):
        Lk = []
        for m in range(0, k):
            Lmk = 0
            for i in range(1, int(numpy.floor((N - m) / k))):
                Lmk += abs(X[m + i * k] - X[m + i * k - k])
            Lmk = Lmk * (N - 1) / numpy.floor((N - m) / float(k)) / k
            Lk.append(Lmk)
        L.append(numpy.log(numpy.mean(Lk)))
        x.append([numpy.log(float(1) / k), 1])

    (p, _, _, _) = numpy.linalg.lstsq(x, L)
    return p[0]


def pfd(X, D=None):
    """Compute Petrosian Fractal Dimension of a time series from either two
    cases below:
        1. X, the time series of type list (default)
        2. D, the first order differential sequence of X (if D is provided,
           recommended to speed up)

    In case 1, D is computed using Numpy's difference function.

    To speed up, it is recommended to compute D before calling this function
    because D may also be used by other functions whereas computing it here
    again will slow down.
    """
    if D is None:
        D = numpy.diff(X)
        D = D.tolist()
    N_delta = 0  # number of sign changes in derivative of the signal
    for i in range(1, len(D)):
        if D[i] * D[i - 1] < 0:
            N_delta += 1
    n = len(X)
    return numpy.log10(n) / (
        numpy.log10(n) + numpy.log10(n / n + 0.4 * N_delta)
    )


import numpy


def hjorth(X, D=None):

    if D is None:
        D = numpy.diff(X)
        D = D.tolist()

    D.insert(0, X[0])  # pad the first difference
    D = numpy.array(D)

    n = len(X)

    M2 = float(sum(D ** 2)) / n
    TP = sum(numpy.array(X) ** 2)
    M4 = 0
    for i in range(1, len(D)):
        M4 += (D[i] - D[i - 1]) ** 2
    M4 = M4 / n

    return numpy.sqrt(M2 / TP), numpy.sqrt(
        float(M4) * TP / M2 / M2
    )  # Hjorth Mobility and Complexity