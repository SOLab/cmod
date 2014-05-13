#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy

import pyopencl as cl
from pyopencl import array

from time import time

__author__ = 'Anna Monzikova (Monzik), Alexander Myasoedov (magican)'
__email__ = 'monzik@rshu.ru, mag@rshu.ru'
__created__ = datetime.datetime(2014, 5, 13)
__modified__ = datetime.datetime(2014, 5, 13)
__version__ = "1.0"
__status__ = "Development"


def rcs2windOpenCl(sar, windir, theta):
    
    """
    Returns wind speed at 10 m, neutral stratification.

    This function calculates wind speed at 10 m,
    using CMOD5 model. The calculations are performed 
    on a GPU using PyOpenCL module.

    Parameters
    ----------
    sar : numpy array
        Normalized Radar Cross Section in linear units

    windir : numpy array
        Angle between azimuth and wind direction in degrees.

    theta: numpy array
        Incidence angle in degrees.

     Returns
    -------
    v_result : numpy array
         Wind speed, 10 m, neutral stratification in m/s.
    """
    KERNEL_CODE = """
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable \n
    double  cmod5n_forward(double v, double phi, double theta)
    
    {        
            const double DTOR = 57.29577951;
            const double THETM = 40.;
            const double THETHR = 25.;
            const double ZPOW = 1.6;
    
            double C[] = {0, -0.6878, -0.7957, 0.3380, -0.1728, 0.0000, 0.0040, 0.1103, 0.0159,\
                                6.7329, 2.7713, -2.2885, 0.4971, -0.7250, 0.0450,\
                                0.0066, 0.3222, 0.0120, 22.7000, 2.0813, 3.0000, 8.3659,\
                                -3.3428, 1.3236, 6.2437, 2.3893, 0.3249, 4.1590, 1.6930};
    
    
            double Y0 = C[19];
            double PN = C[20];
            double A = C[19] - (C[19] - 1) / C[20];
    
            double B = 1. / (C[20] * pow((C[19] - 1.) , (3 - 1)));
    
            double FI = phi / DTOR;
            double CSFI = cos(FI);
            double CS2FI = 2.00 * CSFI * CSFI - 1.00;
            double X = (theta - THETM) / THETHR;
            double XX = X * X;
            double A0 = C[1] + C[2] * X + C[3] * XX + C[4] * X * XX;
            double A1 = C[5] + C[6] * X;
            double A2 = C[7] + C[8] * X;
            double GAM = C[9] + C[10] * X + C[11] * XX;
            double S0 = C[12] + C[13] * X;
            double S = A2 * v;
            double A3;
            if(S<S0)
            {
                A3 = 1. / (1. + exp(-S0));
                A3 = A3 * pow((S / S0), ( S0 * (1. - A3)));
            }
            else
            {
                A3 = 1. / (1. + exp(-S));
            }
            double B0 = (pow(A3, GAM)) * pow(10. , (A0 + A1 * v));
            double B1 = C[15] * v * (0.5 + X - tanh(4. * (X + C[16] + C[17] * v)));
            B1 = C[14] * (1. + X) - B1;
            B1 = B1 / (exp(0.34 * (v - C[18])) + 1.);
            double V0 = C[21] + C[22] * X + C[23] * XX;
            double D1 = C[24] + C[25] * X + C[26] * XX;
            double D2 = C[27] + C[28] * X;
            double V2 = (v / V0 + 1.);
            if (V2<Y0)
            {
                V2 = A + B * pow((V2 - 1.), PN);
            }
    
    
            double B2 = (-D1 + D2 * V2) * exp(-V2);
    
    
            double CMOD5_N = B0 * pow((1.0 + B1 * CSFI + B2 * CS2FI), ZPOW);
    
            return CMOD5_N;
        }
    
    __kernel void cmod5n_inverse( __global const double* sigma0_obs, __global const double* phi, __global const double* incidence, __global double* V)
    
    {
        int gid =get_global_id(1) * get_global_size(0) + get_global_id(0);
        double  sigma0_calc;
        double step = 10.;
        V[gid]=10.;
        for (int it = 1; it<10; ++it)
        {
            sigma0_calc = cmod5n_forward(V[gid], phi[gid], incidence[gid]);
            V[gid] = V[gid] + step;
            if(((sigma0_calc - sigma0_obs[gid]) == 0)|| ((sigma0_calc - sigma0_obs[gid]) == NAN) || ((sigma0_calc - sigma0_obs[gid]) == INFINITY) )
            {
                V[gid] = V[gid] - step;
            }
            if(((sigma0_calc - sigma0_obs[gid]) > 0) && ((sigma0_calc - sigma0_obs[gid]) != NAN) &&((sigma0_calc - sigma0_obs[gid]) != INFINITY))
            {
                V[gid] = V[gid] - 2 *step;
            }
    
            step = step / 2;
        }
    
        
    } """
    
    start = time()
    
    sar = sar.astype(numpy.float64)
    windir = windir.astype(numpy.float64)
    theta = theta.astype(numpy.float64)
    v_result = numpy.empty_like(sar)
        
    ## Step #1. Obtain an OpenCL platform.
    platform = cl.get_platforms()[0]
     
    ## Step #2. Obtain a device id for at least one device (accelerator).
    device = platform.get_devices()[0]
     
    ## Step #3. Create a context for the selected device.
    context = cl.Context([device])
     
    ## Step #4. Create the accelerator program from source code.
    ## Step #5. Build the program.
    ## Step #6. Create one or more kernels from the program functions.
    program = cl.Program(context, KERNEL_CODE).build()
     
    ## Step #7. Create a command queue for the target device.
    queue = cl.CommandQueue(context, properties=cl.command_queue_properties.PROFILING_ENABLE)
     
    ## Step #8. Allocate device memory and move input data from the host to the device memory.
    mem_flags = cl.mem_flags
    sar_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=sar)
    windir_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=windir)
    theta_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=theta)
    v_result_buf = cl.Buffer(context, mem_flags.WRITE_ONLY, v_result.nbytes)
     
    ## Step #9. Associate the arguments to the kernel with kernel object.
    ## Step #10. Deploy the kernel for device execution.
    exec_evt = program.cmod5n_inverse(queue,v_result.shape, None, sar_buf, windir_buf, theta_buf, v_result_buf)
    
    ## Step #11. Move the kernel’s output data to host memory.
    cl.enqueue_copy(queue, v_result, v_result_buf)
    end = time()
    print "Execution time of CMOD5 with PyOpenCl: %g s" % (end-start)  
    print "Mean wind speed: %s m/s" % (v_result.mean())
    ## Step #12. Release context, program, kernels and memory.
     
    return v_result


if __name__ == "__main__":
    wind = rcs2windOpenCL(sar=0.9146 * ones((1, 1)),
                    windir=0 * ones((1, 1)),
                    theta=20 * ones((1, 1)))
    print "Testing CMOD5 passed, Wind = %f" % wind.mean()
    
