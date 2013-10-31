#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed May 16 11:36:00 2012

@author: mag
"""

from numpy import arange, dtype, array, log10, tanh, ceil, \
    sqrt, cos, exp, pi, round, zeros, ones, reshape, \
    power, rint, where, int16
from time import time
from multiprocessing import Process, Queue

# Filter warnings, when evaluating log10 of a NaN value
import warnings

warnings.filterwarnings(action="ignore", category=RuntimeWarning)

import datetime

__author__ = 'Alexander Myasoedov'
__email__ = 'mag@rshu.ru'
__created__ = datetime.datetime(2012, 5, 16)
__modified__ = datetime.datetime(2012, 5, 16)
__version__ = "1.0"
__status__ = "Development"


def cmod4(u, windir, theta):
    '''
    ! ---------
    ! cmod4(u, windir, theta)
    !
    ! inputs:
    ! u in [m/s] wind velocity (always >= 0)
    ! windir in [deg] angle between azimuth and wind direction
    ! (= D - AZM)
    ! theta in [deg] incidence angle
    !
    ! output:
    ! sig Normalized Radar Cross Section in [linear units]
    !
    ! windir and theta must be Numpy arrays of equal sizes
    !
    ! This function calculates Normalized Radar Cross Section using CMOD4 model.
    !CMOD4 forward model - JHUAPL - Nathaniel Winstead - July 17, 2007.
    !Stoffelen&Anderson (1997) Scatterometer Data Interpretation:
    !Measurement Space and Inversion.
    !---------------------------------------------------------------------
    '''
    # c-coefficients
    c1 = -2.301523
    c2 = -1.632686
    c3 = 0.761210
    c4 = 1.156619
    c5 = 0.595955
    c6 = -0.293819
    c7 = -1.015244
    c8 = 0.342175
    c9 = -0.500786
    c10 = 0.014430
    c11 = 0.002484
    c12 = 0.074450
    c13 = 0.004023
    c14 = 0.148810
    c15 = 0.089286
    c16 = -0.006667
    c17 = 3.000000
    c18 = -10.000000

    # br-coefficients
    thetafac = array([
                         1.075, 1.075, 1.075, 1.072, 1.069, 1.066, 1.056,
                         1.030, 1.004, 0.979,
                         0.967, 0.958, 0.949, 0.941, 0.934, 0.927, 0.923,
                         0.930, 0.937, 0.944,
                         0.955, 0.967, 0.978, 0.988, 0.998, 1.009, 1.021,
                         1.033, 1.042, 1.050,
                         1.054, 1.053, 1.052, 1.047, 1.038, 1.028, 1.016,
                         1.002, 0.989, 0.965,
                         0.941, 0.929, 0.929, 0.929, 0.929], dtype='f8')

    # convert theta to int before calculating br to use indexes
    if type(theta).__name__ == 'ndarray' or type(theta).__name__ == 'ndarray':
        if theta.dtype == dtype('f4') or theta.dtype == dtype('f8'):
            x = (theta - 40) / 25
            theta = rint(theta)
            theta = theta.astype(int16)
            br = thetafac[round(theta - 16) + 1]
        elif theta.dtype == dtype('i4') or theta.dtype == dtype('i8'):
            br = thetafac[round(theta - 16) + 1]
            theta = theta.astype(float)
            x = (theta - 40) / 25
    else:
        x = (theta - 40) / 25
        theta = rint(theta)
        theta = theta.astype(int16)
        br = thetafac[round(theta - 16) + 1]

    P0 = 1
    P1 = x
    P2 = ((3 * (power(x, 2))) - 1) / 2

    F2x = tanh(2.5 * (x + 0.35)) - (0.61 * (x + 0.35))
    alfa = (c1 * P0) + (c2 * P1) + (c3 * P2)
    beta = (c7 * P0) + (c8 * P1) + (c9 * P2)
    gamma = (c4 * P0) + (c5 * P1) + (c6 * P2)

    b1 = c10 * P0 + c11 * u + (c12 * P0 + c13 * u) * F2x
    b2 = (c14 * P0) + (c15 * (1 + P1) * u)
    b3 = 0.42 * (1 + c16 * (c17 + x) * (c18 + u))

    y = u + beta

    if y.all() <= 0:
        F1x = 0
    elif y.all() <= 5:
        F1x = log10(y)
    else:
        F1x = sqrt(y) / 3.2
    b0 = br * 10 ** (alfa + (gamma * F1x))
    sig = b0 * (1 + b1 * cos(windir * pi / 180) + b3 * tanh(b2) * cos(
        2 * windir * pi / 180)) ** 1.6
    #    sig0 = 10*log10(sig) # in dB

    return sig


def cmod5n_forward(v, phi, theta):
    '''! ---------
    ! cmod5n_forward(v, phi, theta)
    ! inputs:
    ! v in [m/s] wind velocity (always >= 0)
    ! phi in [deg] angle between azimuth and wind direction
    ! (= D - AZM)
    ! theta in [deg] incidence angle
    ! output:
    ! CMOD5_N NORMALIZED BACKSCATTER (LINEAR)
    !
    ! All inputs must be Numpy arrays of equal sizes
    !
    ! A. STOFFELEN MAY 1991 ECMWF CMOD4
    ! A. STOFFELEN, S. DE HAAN DEC 2001 KNMI CMOD5 PROTOTYPE
    ! H. HERSBACH JUNE 2002 ECMWF COMPLETE REVISION
    ! J. de Kloe JULI 2003 KNMI, rewritten in fortan90
    ! A. Verhoef JAN 2008 KNMI, CMOD5 for neutral winds
    ! K.F.Dagestad OCT 2011 NERSC, Vectorized Python version
    !---------------------------------------------------------------------
    '''

    DTOR = 57.29577951
    THETM = 40.
    THETHR = 25.
    ZPOW = 1.6

    # NB: 0 added as first element below, to avoid switching from 1-indexing to 0-indexing
    C = [0, -0.6878, -0.7957, 0.3380, -0.1728, 0.0000, 0.0040, 0.1103, 0.0159,
         6.7329, 2.7713, -2.2885, 0.4971, -0.7250, 0.0450,
         0.0066, 0.3222, 0.0120, 22.7000, 2.0813, 3.0000, 8.3659,
         -3.3428, 1.3236, 6.2437, 2.3893, 0.3249, 4.1590, 1.6930]
    Y0 = C[19]
    PN = C[20]
    A = C[19] - (C[19] - 1) / C[20]

    B = 1. / (C[20] * (C[19] - 1.) ** (3 - 1))

    # ! ANGLES
    FI = phi / DTOR
    CSFI = cos(FI)
    CS2FI = 2.00 * CSFI * CSFI - 1.00

    X = (theta - THETM) / THETHR
    XX = X * X

    # ! B0: FUNCTION OF WIND SPEED AND INCIDENCE ANGLE
    A0 = C[1] + C[2] * X + C[3] * XX + C[4] * X * XX
    A1 = C[5] + C[6] * X
    A2 = C[7] + C[8] * X

    GAM = C[9] + C[10] * X + C[11] * XX
    S0 = C[12] + C[13] * X

    # V is missing! Using V=v as substitute, this is apparently correct
    V = v
    S = A2 * V
    S_vec = S.copy()
    SlS0 = [S_vec < S0]
    S_vec[SlS0] = S0[SlS0]
    A3 = 1. / (1. + exp(-S_vec))
    SlS0 = (S < S0)
    A3[SlS0] = A3[SlS0] * (S[SlS0] / S0[SlS0]) ** ( S0[SlS0] * (1. - A3[SlS0]))
    #A3=A3*(S/S0)**( S0*(1.- A3))
    B0 = (A3 ** GAM) * 10. ** (A0 + A1 * V)

    # ! B1: FUNCTION OF WIND SPEED AND INCIDENCE ANGLE
    B1 = C[15] * V * (0.5 + X - tanh(4. * (X + C[16] + C[17] * V)))
    B1 = C[14] * (1. + X) - B1
    B1 = B1 / (exp(0.34 * (V - C[18])) + 1.)

    # ! B2: FUNCTION OF WIND SPEED AND INCIDENCE ANGLE
    V0 = C[21] + C[22] * X + C[23] * XX
    D1 = C[24] + C[25] * X + C[26] * XX
    D2 = C[27] + C[28] * X

    V2 = (V / V0 + 1.)
    V2ltY0 = V2 < Y0
    V2[V2ltY0] = A + B * (V2[V2ltY0] - 1.) ** PN
    B2 = (-D1 + D2 * V2) * exp(-V2)

    # ! CMOD5_N: COMBINE THE THREE FOURIER TERMS
    CMOD5_N = B0 * (1.0 + B1 * CSFI + B2 * CS2FI) ** ZPOW

    return CMOD5_N


def cmod5n_inverse(sigma0_obs, phi, incidence, iterations=10):
    '''! ---------
    ! cmod5n_inverse(sigma0_obs, phi, incidence, iterations)
    ! inputs:
    ! sigma0_obs Normalized Radar Cross Section [linear units]
    ! phi in [deg] angle between azimuth and wind direction
    ! (= D - AZM)
    ! incidence in [deg] incidence angle
    ! iterations: number of iterations to run
    ! output:
    ! Wind speed, 10 m, neutral stratification
    !
    ! All inputs must be Numpy arrays of equal sizes
    !
    ! This function iterates the forward CMOD5N function
    ! until agreement with input (observed) sigma0 values
    !---------------------------------------------------------------------
    '''

    # First guess wind speed
    V = array([10.]) * ones(sigma0_obs.shape);
    step = 10.

    # Iterating until error is smaller than threshold
    for iterno in range(1, iterations):
        #print iterno
        sigma0_calc = cmod5n_forward(V, phi, incidence)
        ind = sigma0_calc - sigma0_obs > 0
        V = V + step
        V[ind] = V[ind] - 2 * step
        step = step / 2

    #mdict={'s0obs':sigma0_obs,'s0calc':sigma0_calc}
    #from scipy.io import savemat
    #savemat('s0test',mdict)

    return V


def interp1gsy(x, y, xi):
    """
    George Young's bottom-up interpolation function.
    A replacement for Matlab's interp1 for non-monotonic data.
    """
    #  method2use is "n" for nearest neighbor, "l" for linear regression
    #  See how many points there are
    npts = x.shape[0]
    #  Loop through list from start to finish looking for a bracket
    yi = y[x.argmax(axis=0)]

    for ipt in range(1, npts):
        goofbelow = xi - x[ipt - 1]
        goofabove = xi - x[ipt]
        a = where(goofbelow * goofabove < 0)
        yi[a] = (y[ipt] + (xi[a] - x[ipt][a]) * (y[ipt] - y[ipt - 1]) / (
        x[ipt][a] - x[ipt - 1][a]))

    return yi


def rcs2wind(sar=0.9146 * ones((1, 1)), cmdv=4, windir=0 * ones((1, 1)),
             theta=20 * ones((1, 1))):
    '''
    ! ---------
    ! rcs2wind(sar=0.9146 * ones((1, 1)), cmdv=4, windir=0 * ones((1, 1)),
                 theta=20 * ones((1, 1)))
    ! inputs:
    ! sar Normalized Radar Cross Section [linear units]
    ! cmdv - cmod version, 4 or 5
    ! windir in [deg] angle between azimuth and wind direction
    ! (= D - AZM)
    ! theta in [deg] incidence angle
    !
    ! output:
    ! Wind speed, 10 m, neutral stratification
    !
    ! Theta and windir must be same size as inputed sar
    !
    ! This function calculates wind speed at 10 m, using CMOD4 or CMOD5 model.
    !---------------------------------------------------------------------
    '''

    # Set the maximum wind to be retrieved
    maxwind = 35.0
    # Create a list of winds to be retrieved - linear interpolation of wind
    # speed given rcs will done between values spaced every m/s from 0 to
    # maxwind
    ws = arange(0, maxwind, 1.0)
    #    alpha = 0.6
    # Start timer
    print "Calculating CMOD..."
    currtime = time()

    sig = zeros((ws.size, sar.shape[0], sar.shape[1]))
    # Loop over all of the available pixels, calling CMOD for each wind speed
    if cmdv == 4:
        for ind in range(ws.size):
            sig[ind, :, :] = cmod4(u=ws[ind], windir=windir, theta=theta)
            # Use linear interpolation to look up the right wind in the sima table.
        print "Sigma to Wind LUT..."
        w = interp1gsy(x=sig, y=ws, xi=sar)
    elif cmdv == 5:
        w = cmod5n_inverse(sar, windir, theta, iterations=10)
    else:
        print "Illegal CMOD version specified"

    print 'CMOD elapsed time: %f' % (time() - currtime)
    return w


def rcs2windPar(sar=0.9146 * ones((1, 1)), \
                cmdv=4, windir=0 * ones((1, 1)), theta=20 * ones((1, 1)),
                nprocs=4):
    '''
    ! ---------
    ! rcs2windPar(sar=0.9146 * ones((1, 1)), cmdv=4, windir=0 * ones((1, 1)),
                 theta=20 * ones((1, 1)), nprocs=4)
    ! inputs:
    ! sar Normalized Radar Cross Section [linear units]
    ! cmdv - cmod version, 4 or 5
    ! windir in [deg] angle between azimuth and wind direction
    ! (= D - AZM)
    ! theta in [deg] incidence angle
    ! nprocs - number of processes
    !
    ! output:
    ! Wind speed, 10 m, neutral stratification
    !
    ! Theta and windir must be same size as inputed sar
    !
    ! This function calculates  in parallel wind speed at 10 m, using CMOD4
    ! or CMOD5 model.
    !---------------------------------------------------------------------
    '''
    def worker(sar, cmdv, windir, theta, out_q=None):
        maxwind = 35.0
        # Create a list of winds to be retrieved - linear interpolation of wind
        # speed given rcs will done between values spaced every m/s from 0 to
        # maxwind
        ws = arange(0, maxwind, 1.0)
        sig = zeros((ws.size, theta.size))
        # Loop over all of the available pixels, calling CMOD for each
        #  wind speed
        if cmdv == 4:
            for ind in range(ws.size):
                sig[ind, :] = cmod4(u=ws[ind], windir=windir, theta=theta)
            w = interp1gsy(x=sig, y=ws, xi=sar)
        elif cmdv == 5:
            w = cmod5n_inverse(sar, windir, theta, iterations=10)
        else:
            print "Illegal CMOD version specified"
        out_q.put(w)

    # Start timer
    print "Calculating CMOD..."
    currtime = time()

    sarR = reshape(sar, sar.size)
    thetaR = reshape(theta, sar.size)
    windirR = reshape(windir, sar.size)
    # Each process will get 'chunksize' nums and a queue to put his out
    # dict into
    chunksize = int(ceil(len(thetaR) / float(nprocs)))
    procs = []
    out_q = []
    for i in range(nprocs):
        q = Queue()
        p = Process(
            target=worker,
            args=(sarR[chunksize * i:chunksize * (i + 1)], cmdv, \
                  windirR[chunksize * i:chunksize * (i + 1)], \
                  thetaR[chunksize * i:chunksize * (i + 1)], \
                  q))
        procs.append(p)
        out_q.append(q)
        p.start()
        # Collect all results into a single result dict.
    # We know how many dicts with results to expect.
    #resultS = zeros((ws.size, sar.shape[0], sar.shape[1]))
    resultW = zeros(sar.size)
    i = 0
    for q in out_q:
        resultW[chunksize * i:chunksize * (i + 1)] = q.get()
        i += 1
        # Processes that raise an exception automatically get an exitcode of 1.
    procs[-1].terminate()
    # Wait for all worker processes to finish
    for p in procs:
        p.join()
    print "Sigma to Wind LUT..."
    # w = zeros(sar.size)
    resultW = reshape(resultW, sar.shape)

    print 'CMOD elapsed time: %f' % (time() - currtime)
    return resultW


if __name__ == "__main__":
    wind = rcs2wind(sar=0.9146 * ones((1, 1)), cmdv=4,
                    windir=0 * ones((1, 1)),
                    theta=20 * ones((1, 1)))
    print "Testing CMOD4 passed, Wind = %f" % wind
    wind = rcs2wind(sar=0.9146 * ones((1, 1)), cmdv=5,
                    windir=0 * ones((1, 1)),
                    theta=20 * ones((1, 1)))
    print "Testing CMOD5 passed, Wind = %f" % wind