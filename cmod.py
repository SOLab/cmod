#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed May 16 11:36:00 2012

@author: mag
"""

from numpy import arange, dtype, array, log10, tanh, ceil, \
                  sqrt, cos, pi, round, zeros, ones, reshape
from time import time
from multiprocessing import Process, Queue

# Filter warnings, when evaluating log10 of a NaN value
import warnings
warnings.filterwarnings(action = "ignore", category=RuntimeWarning)

import datetime

__author__    =  'Alexander Myasoedov'
__email__    = 'mag@rshu.ru'
__created__   =  datetime.datetime(2012, 5, 16)
__modified__ = datetime.datetime(2012, 5, 16)
__version__   =  "1.0"
__status__   = "Development"

def cmod4(u=10, windir=0, theta=arange(0,46)):
    """
    Calculates Normalized Radar Cross Section using CMOD4 model.
    CMOD4 forward model - JHUAPL - Nathaniel Winstead - July 17, 2007.
    Stoffelen&Anderson (1997) Scatterometer Data Interpretation:
    Measurement Space and Inversion.

    INPUT:
    u           - Wind speed (m/s) at 10m height (?)
    windir      - Angle between wind vector and radar look vector (degrees)
                  NB! windir = 0 when wind blows toward radar!
    theta       - Radar incidence angle (degrees)

    OUTPUT:
    cmod4    - NRCS (dB)
    """
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
    thetafac = array([ \
                1.075,1.075,1.075,1.072,1.069,1.066,1.056,1.030,1.004,0.979, \
                0.967,0.958,0.949,0.941,0.934,0.927,0.923,0.930,0.937,0.944, \
                0.955,0.967,0.978,0.988,0.998,1.009,1.021,1.033,1.042,1.050, \
                1.054,1.053,1.052,1.047,1.038,1.028,1.016,1.002,0.989,0.965, \
                0.941,0.929,0.929,0.929,0.929], dtype='f8')

    # convert theta to int before calculating br to use indexes
    if type(theta).__name__ == 'ndarray' or type(theta).__name__ == 'ndarray':
        if theta.dtype == dtype('f4') or theta.dtype == dtype('f8'):
            x = (theta - 40)/25
            theta = array(map(int, theta))
            br = thetafac[round(theta-16)+1]
        elif theta.dtype == dtype('i4') or theta.dtype == dtype('i8'):
            br = thetafac[round(theta-16)+1]
            theta = array(map(float, theta))
            x = (theta - 40)/25
    else:
        x = (theta - 40)/25
        theta = int(theta)
        br = thetafac[round(theta-16)+1]

    P0 = 1
    P1 = x
    P2 = ((3*(x**2))-1)/2

    F2x = tanh(2.5*(x+0.35))-(0.61*(x+0.35))
    alfa = (c1*P0)+(c2*P1)+(c3*P2)
    beta = (c7*P0)+(c8*P1)+(c9*P2)
    gamma = (c4*P0)+(c5*P1)+(c6*P2)

    b1 = (c10*P0)+(c11*u)+((c12*P0)+(c13*u))*F2x
    b2 = (c14*P0)+(c15*(1+P1)*u)
    b3 = 0.42*(1+c16*(c17+x)*(c18+u))

    y = u+beta

    if y.all() <= 0:
        F1x  =  0
    elif y.all() <= 5:
        F1x  =  log10(y)
    else:
        F1x = sqrt(y)/3.2

    b0 = br*10**(alfa+(gamma*F1x))
    sig = b0*(1+b1*cos(windir*pi/180)+b3*tanh(b2)*cos(2*windir*pi/180))**1.6
#    sig0 = 10*log10(sig) # in dB

    return sig

#import cmod
#reload(cmod)
#from cmod import cmod4
#sig0 = cmod4()

def interp1gsy(x,y,xi,method2use):
    """
    George Young's bottom-up interpolation function.
    A replacement for Matlab's interp1 for non-monotonic data.
    """
    #  method2use is "n" for nearest neighbor, "l" for linear regression
    #  See how many points there are
    npts = x.size
    #  Loop through list from start to finish looking for a bracket
    yi = 0
    for ipt in range(1,npts):
        goofbelow = xi-x[ipt-1]
        goofabove = xi-x[ipt]
        #  Test for a bracket
        if goofbelow*goofabove < 0:
            if method2use == 'n':
                if abs(goofbelow) < abs(goofabove):
                    yi = y[ipt-1]
                else:
                    yi = y[ipt]
            else:  # default to linear interpolation
                yi = y[ipt] + ((xi-x[ipt])*(y[ipt]-y[ipt-1])/(x[ipt]-x[ipt-1]))
            #  Break out of the loop if there is a bracket.
            #  This makes sure you only find the slowest wind speed that brackets
            break
    #  If the backscatter is higher than "possible" assign the wind for the highest backscatter
    xmax = x.max()
    xmaxi = x.argmax()
    if (yi == 0) and (xi > xmax):
        yi = y(xmaxi)
    return yi

def rcs2wind(sar=-0.3877*ones((1,1)), cmdv=4, windir=0*ones((1,1)), theta=20*ones((1,1))):
    """
    Note that input sar is in dB.
    Theta and windir must be same size as inputed sar
    Check that with sar=-0.387 wind speed is 10m/s
    """
    # Set the maximum wind to be retrieved
    maxwind = 35.0
    # Create a list of winds to be retrieved - linear interpolation of wind
    # speed given rcs will done between values spaced every m/s from 0 to
    # maxwind
    ws = arange(0,maxwind,1.0)
#    alpha = 0.6
    # Start timer
    print "Calculating CMOD..."
    currtime = time()
    # reshape sar, windir and theta to vector line
    sarR = reshape(sar, sar.size)
    theta = reshape(theta, sar.size)
    windir = reshape(windir, sar.size)
    # preallocate sig0 with size (sar,ws)
    sig=zeros((sar.size, ws.size))
    # Loop over all of the available pixels, calling CMOD once per pixel
    for i in range(sar.size):
        if cmdv == 4:
            sig[i,:] = cmod4(u=ws, windir=windir[i], theta=theta[i])
#        elif cmdv == 5:
#            sig[i,:] = cmod5(sarR[i])
        else:
            print "Illegal CMOD version specified"

    # Use linear interpolation to look up the right wind in the sima table.
    # Start timer
    print "Sigma to Wind LUT..."
    w = zeros(sar.size)
    for i in range(sar.size):
        temp = sig[i,:]
        wtemp = ws
        xi = 10**(sarR[i]/10)
        # Because Matlab's interpolation function can fail, particularly on
        # non-monotonic functions, we use our own one-dimensional linear
        # interpolation function
        w[i] = interp1gsy(x=temp,y=wtemp,xi=xi,method2use='l')
    w = reshape(w,sar.shape)
    print 'CMOD elapsed time: %f', ( time() - currtime )
    return w

def rcs2windPar(sar=-0.3877*ones((1,1)), \
                cmdv=4, windir=0*ones((1,1)), theta=20*ones((1,1)), nprocs=4):
    def worker(sar, cmdv, windir, theta, out_q=None):
        """
        Note that input sar is in dB.
        Theta and wind must be same size as inputed sar
        Check that with sar=-0.387 wind speed is 10m/s
        """
        # Set the maximum wind to be retrieved
        maxwind = 35.0
        # Create a list of winds to be retrieved - linear interpolation of wind
        # speed given rcs will done between values spaced every m/s from 0 to
        # maxwind
        ws = arange(0,maxwind,1.0)
        # preallocate sig0 with size (sar,ws)
        sig=zeros((sar.size, ws.size))
        # Loop over all of the available pixels, calling CMOD once per pixel
        for i in range(sar.size):
            if cmdv == 4:
                sig[i,:] = cmod4(u=ws, windir=windir[i], theta=theta[i])
    #        elif cmdv == 5:
    #            sig[i,:] = cmod5(sarR[i])
            else:
                print "Illegal CMOD version specified"

        # Use linear interpolation to look up the right wind in the sigma table.
        w = zeros(sar.size)
        for i in range(sar.size):
            temp = sig[i,:]
            wtemp = ws
            xi = 10**(sar[i]/10)
            # Because Matlab's interpolation function can fail, particularly on
            # non-monotonic functions, we use our own one-dimensional linear
            # interpolation function
            w[i] = interp1gsy(x=temp,y=wtemp,xi=xi,method2use='l')
        out_q.put(w)

    # Start timer
    print "Calculating CMOD..."
    currtime = time()
    # reshape sar, windir and theta to vector line
    sarR = reshape(sar, sar.size)
    thetaR = reshape(theta, sar.size)
    windirR = reshape(windir, sar.size)
    # Each process will get 'chunksize' nums and a queue to put his out
    # dict into
    chunksize = int(ceil(len(sarR) / float(nprocs)))
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
    resultW = zeros((sar.size))
    i=0
    for q in out_q:
        resultW[chunksize * i:chunksize * (i + 1)] = q.get()
        i+=1
    # Processes that raise an exception automatically get an exitcode of 1.
    procs[-1].terminate()
    # Wait for all worker processes to finish
    for p in procs:
        p.join()
    resultW = reshape(resultW,sar.shape) # reshape back to initial array
    print 'CMOD elapsed time: %f', ( time() - currtime )
    return resultW


if __name__ == "__main__":
    rcs2wind(sar=-0.3877*ones((1,1)), cmdv=4, windir=180*ones((1,1)), theta=20*ones((1,1)))
