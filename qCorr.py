#!/usr/bin/env python3

import numpy as np
import astropy.io.fits as fits
from sys import argv
from copy import deepcopy

darkfile = argv[1]

with fits.open(darkfile) as hdul:
    image = hdul[1].data.astype('float')
    hdr = hdul[0].header

mdark = deepcopy(image)

#image = np.flip(image)

q0 = image[0:511,0:511]
q1 = image[512:1023,0:511]
q2 = image[512:1023,512:1023]
q3 = image[0:511,512:1023]

q0_mask = np.zeros(q0.shape, dtype=bool)
q0_mask[:,:] = True
q0_mask[0:50,0:50] = False
q0_mask[0:150,110:300] = False
q0_mask[0:60,470:511] = False

q1_mask = np.zeros(q1.shape, dtype=bool)
q1_mask[:,:] = True
q1_mask[0:60,(512-512):(560-512)] = False
q1_mask[0:150,(700-512):(880-512)] = False

q2_mask = np.zeros(q2.shape, dtype=bool)
q2_mask[:,:] = True
q2_mask[(980-512):(1023-512),(512-512):(570-512)] = False
q2_mask[(900-512):(1023-512),(690-512):(880-512)] = False

q3_mask = np.zeros(q3.shape, dtype=bool)
q3_mask[:,:] = True
q3_mask[(970-512):(1023-512),460:511] = False
q3_mask[(850-512):(1023-512),200:400] = False


qAll = np.median([q0,q1,q2,q3], axis=0)

#q0_1dy = np.median(qAll, axis=1, keepdims=True)
#q1_1dy = np.median(qAll, axis=1, keepdims=True)
#q2_1dy = np.median(qAll, axis=1, keepdims=True)
#q3_1dy = np.median(qAll, axis=1, keepdims=True)

image[0:511,0:511] = image[0:511,0:511] - qAll
image[0:511,512:1023] = image[0:511,512:1023] - qAll
image[512:1023,512:1023] = image[512:1023,512:1023] - qAll
image[512:1023,0:511] = image[512:1023,0:511] - qAll

#q0 = image[0:512,0:512]
#q1 = image[0:512,512:1024]
#q2 = image[512:1024,512:1024]
#q3 = image[512:1024,0:512]

#q0_1dx = np.median(q0[201:500,:], axis=0)
#q1_1dx = np.median(q1[201:500,:], axis=0)
#q2_1dx = np.median(q2[0:300,:], axis=0)
#q3_1dx = np.median(q3[0:300,:], axis=0)

#image[0:512,0:512] = image[0:512,0:512] - q0_1dx
#image[0:512,512:1024] = image[0:512,512:1024] - q1_1dx
#image[512:1024,512:1024] = image[512:1024,512:1024] - q2_1dx
#image[512:1024,0:512] = image[512:1024,0:512] - q3_1dx

out_hdu = fits.PrimaryHDU((image.astype('float')), header=hdr)
out_hdul = fits.HDUList([out_hdu])
out_hdul.writeto(darkfile.split('.')[0]+"_qCorr.fits", overwrite='True')
