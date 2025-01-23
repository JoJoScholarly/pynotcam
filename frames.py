#!/usr/bin/env python3

import numpy as np
import astropy.io.fits as fits
from numba import njit, prange
from sys import argv


@njit( parallel=True )
def linFit( frames, N, t, ignore=1 ):
    ignore = ignore # Don't include to first reads in the fit.
    frames = frames[ignore:,:,:]
    x = np.linspace(t+t*ignore, N*t, (N-ignore))
    
    lb = np.zeros( (1024, 1024) )
    for i in prange(0, frames.shape[1]):
        for j in range(0, frames.shape[2]):
            y = frames[:, i, j]
            #weights = ((10**0.5)**2 + (np.nan_to_num( frames[:,i,j]**(-0.5), nan=1e-9, posinf=1e-9, neginf=1e-9 ))**2)**0.5
            #lb[i,j] = np.polyfit(x,y,1,w=weights)[0]
            lb[i,j] = ((N-ignore)*(x*y).sum() - x.sum()*y.sum()) / ((N-ignore)*(x*x).sum() - x.sum()*x.sum())
    return lb*N*t


def exptime( expModeKeyword ):
    """Extract exposure time from readmode fits header keyword. NOTCam has to read modes:
       1.) reset-read-read (rrr)
       2.) ramp-sample (rs) with n reads (n_max=14)
       For rrr, the total exposure time consist of time interval between the two reads (1 x exptime). 
       For rs, the total exposure time consist of n times the time interval between two reads(n x exptime)
       This function return tuple with where first element is exposure time, and second number of subexposures.
    """
    if 'frames' in expModeKeyword:
        t = float(expModeKeyword.split(' ')[1])
        N = int(expModeKeyword.split(' ')[2]) 
    else:
        print('Not exposed with NOTCam frame or dframe command.')
        t = 0
        N = 0
    return ( t, N )


if __name__ == "__main__":
    filename = argv[1]

    with fits.open( filename ) as hdul:
        hdr = hdul[0].header # Get the primary header, no data in hdu[0] in case of NOTCam
        resetLevel = hdul[-1].data # Reset level is stored to las hdu

        headers = []
        for hdu in hdul:
            headers.append( hdu.header )

        t, N = exptime( hdr['EXPMODE'] )
        
        frames = []
        # Skip
        # hdu[0] = (no data)
        # hdu[1] = (controller linear fit)
        # hdu[-1] =  (reset level)
        for i in np.arange(2, len(hdul)-1):
            frames.append(hdul[i].data.astype(np.int32))
        frames = np.array(frames)

        lbNt = linFit( frames, N, t )

        out_file_name = filename.split('.')[0] + '_linFit.fits'

        hdusOut = [fits.PrimaryHDU( header=headers[0] )] # Like the original, no data in primary HDU
        hdusOut.append( fits.ImageHDU( lbNt, header=headers[1] )) # New linear fit
        
        # Add sub-reads and attach headers
        for i in range(0, len(frames)):
            hdusOut.append( fits.ImageHDU( frames[i], header=headers[i+2] ))
        
        hdusOut.append( fits.ImageHDU( resetLevel, header=headers[-1] ))

        hdul = fits.HDUList(hdusOut)
        hdul.writeto(out_file_name, overwrite=True)
