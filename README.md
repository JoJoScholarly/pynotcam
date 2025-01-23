# pynotcam
`pynotcam` is meant for reducing NOT's NOTCam instrument's data.

## Description  
`frames` re-processes ramp-sampling images. Differences compared to the detector generated linear least-squares fit are ability to recover sub-reads where controller has recorded below reset level values, and ingnoring the first reads to limit effect of reset anomaly. See below for more on these features.  

### Re-wrap negative values
The BIAS controller (Klougart, 1994) stores data as 16-bit unsigned integer values. In a ramp-sampling integration, all sub-reads are stored as a difference to the reset-level. In long exposures, it is possible to encounter negative values in the sub-read. The negative valuse appear as values close to 2^16 (two's complement).

### Linear fitting
User selectable number of first sub-reads can be skipped (default skip first).


## Other
`qCorr` uses all four quandrants to measure the DC voltage offsets. 

## Requirements
`numpy`, `astropy`
