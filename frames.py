#!/usr/bin/env python3

import numpy as np
import astropy.io.fits as fits
from numba import njit, prange
from sys import argv


@njit( parallel=True )
def linFit( frames, N, t ):
    x = np.linspace(0, N*t, N)
    lb = np.zeros( (1024, 1024) )
    for i in prange(0, frames.shape[1]):
        for j in range(0, frames.shape[2]):
            y = frames[:, i, j]
            lb[i,j] = (N*(x*y).sum() - x.sum()*y.sum()) / (N*(x*x).sum() - x.sum()*x.sum())
    return lb*N*t


if __name__ == "__main__":
    filename = argv[1]

    with fits.open( filename ) as hdul:
        hdr = hdul[0].header # Get the primary header, no data in hdu[0] in case of NOTCam
        resetLevel = hdul[-1].data # Reset level is stored to las hdu

        N = len(hdul) - 2 - 1# Only sub-reads
        t = 43.0 # This should be scraped from keywords, best calc from timestamp
        
        frames = []
        # Skip
        # hdu[0] = (no data)
        # hdu[1] = (controller linear fit)
        # hdu[-1] =  (reset level)
        for i in np.arange(2, len(hdul)-1):
            frames.append(hdul[i].data.astype(np.int16))
        frames = np.array(frames)

        lbNt = linFit( frames, N, t )

        out_file_name = filename.split('.')[0] + '_linFit.fits'

        hdusOut = [fits.PrimaryHDU( lbNt )]
        for frame in frames:
            hdusOut.append( fits.ImageHDU( frame ))
        hdusOut.append( fits.ImageHDU( resetLevel ))

        hdul = fits.HDUList(hdusOut)
        hdul.writeto(out_file_name, overwrite=True)
