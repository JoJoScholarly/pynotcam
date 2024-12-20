#!/usr/bin/env python3

import numpy as np
import astropy.io.fits as fits
from numba import njit, prange
from sys import argv


def twosComplements( frames ):
    """Iterate sub-frames, convert with two's complement."""
    for i in range( frames.shape[0] ):
        frames[i] = twosComplement( frames[i] )
    return frames


@njit( parallel=True )
def twosComplement( frame ):
    for i in prange( frame.shape[0] ):
        for j in range( frame.shape[1] ):
            if frame[i,j] > 62000:
                frame[i,j] = ( 65536 - frame[i,j] )
    return frame


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
        hdr = hdul[0].header
        resetLevel = hdul[-1].data

        N = len(hdul) - 2 - 1 # Only sub-reads
        t = 43.0 # This should be scraped from keywords, best calc from timestamp
        
        frames = []
        for i in np.arange(2, len(hdul)-1):
            frames.append(hdul[i].data)
        frames = np.array(frames)

        frames = twosComplements( frames )
        lbNt = linFit( frames, N, t )

        out_file_name = filename.split('.')[0] + '_linFit.fits'

        hdus = [fits.PrimaryHDU( lbNt )]
        for frame in frames:
            hdus.append( fits.ImageHDU( frame ))
        hdus.append( fits.ImageHDU( resetLevel ))

        hdul = fits.HDUList(hdus)
        hdul.writeto(out_file_name, overwrite=True)
