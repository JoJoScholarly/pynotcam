#!/usr/bin/env python3

import astropy.io.fits as fits
import numpy as np
from sys import argv

filename = argv[1]
lowerClip = float(argv[2])
upperClip = float(argv[3])

with fits.open( filename ) as hdul:
    dark = hdul[0].data

mask = np.zeros( dark.shape )
mask[dark<lowerClip] = 1
mask[dark>upperClip] = 1
mask[dark==0] = 1
mask[512,:] = 1
mask[:,512] = 1
mask[507:516, 507:516] = 1

hdu = fits.PrimaryHDU( mask )
hdu.writeto( 'badpix.fits', overwrite=True)