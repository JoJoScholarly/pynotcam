#!/usr/bin/env python3

import astropy.io.fits as fits
import numpy as np
from sys import argv


def fixPix(
        image: np.ndarray,
        mask: np.ndarray,
        method: str = 'nearest',
        fill_value: int = 0
):
    """
    :param image: a 2D image
    :param mask: a 2D boolean image, True indicates missing values
    :param method: interpolation method, one of
        'nearest', 'linear', 'cubic'.
    :param fill_value: which value to use for filling up data outside the
        convex hull of known pixel values.
        Default is 0, Has no effect for 'nearest'.
    :return: the image with missing values interpolated
    """
    from scipy import interpolate

    h, w = image.shape[:2]
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))

    known_x = xx[~mask]
    known_y = yy[~mask]
    known_v = image[~mask]
    missing_x = xx[mask]
    missing_y = yy[mask]

    interp_values = interpolate.griddata(
        (known_x, known_y), known_v, (missing_x, missing_y),
        method=method, fill_value=fill_value
    )

    interp_image = image.copy()
    interp_image[missing_y, missing_x] = interp_values

    return interp_image

if __name__ == "__main__":
    filename = argv[1]

    with fits.open( filename ) as hdul:
        data = hdul[1].data
        hdr = hdul[0].header
    
    with fits.open( 'badpix.fits' ) as hdul:
        badpix = hdul[0].data.astype(bool)

    data = fixPix(data, badpix)

    hdu = fits.PrimaryHDU( (data), header=hdr) 
    hdu.writeto( filename.split('.')[0] + '_badpx.fits', overwrite=True)
    