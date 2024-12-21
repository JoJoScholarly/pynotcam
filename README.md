# pynotcam
`pynotcam` is meant for reducing NOT's NOTCam instrument's data.

## Description  
`frames` re-processes ramp-sampling images, and can recover sub-reads where negative values have been calculated by the detector controller. The controller stores 16-bit unsigned integer values, and the negative valuse appear as values close to 2^16 (two's complement). `qCorr` uses all four quandrants to measure the DC voltage offsets. 

## Requirements
`numpy`, `astropy`
