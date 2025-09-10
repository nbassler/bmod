# Recording Spot Positions

 - Equiptment: XRV4000
 - No range shifter.
 - In QA Plan Capture tab, set to "General Beamcenter (automatic)"
 - In "Script Control" tab: Set FPS = 15; Gain = 0
 - Measurement Positions relative to isocenter plane z = 0: `z =[-100,-50,0,50,100,150,200,250]`
 - Request multifield plan `XRV4000_tr4.dcm`
 - When PROTON READY - Start script, and let it run uninterrupted for all 3 fields.

## Notes

### Calibration Values
RefValues20240826.txt, which contains:

XRVcalH0pos: Horizontal pixel offset (origo)
XRVcalV0pos: Vertical pixel offset (origo)
XRVcalHscaling: Horizontal pixel-to-mm scaling factor
XRVcalVscaling: Vertical pixel-to-mm scaling factor


### FunctionFindAndAnalyse
- Finds and fits spots in images.
- Calibrates pixel measurements to millimeters.
- Compares results to reference/commissioning data.
- Saves and further analyzes data for each image.

Needs:
- Energies (MeV) for each image.
- z position (mm) of the detector.
- Treatment Room (TR) and Range Shifter (RS) used.

### FunctionFindPeaksAndFit_v20220308
- Reads images.
- Fits 2D Gaussian functions to each spot.
- Calibrates pixel measurements to millimeters.
- Optionally saves images and fit results.
- Outputs all data for further analysis.


### Functionfit2DgaussFunctionPixels20230321
 - median filtering, detect peak over threshold



## Notes for new implementation
- numpy: For array operations and numerical computations.
- scipy: For optimization, fitting, and signal processing.
- scipy.optimize.curve_fit: For 2D Gaussian fitting.
- scipy.ndimage: For filtering and peak detection.
- skimage (scikit-image): For image processing (e.g., filtering, thresholding, peak finding).
- - - matplotlib: For visualization.
- pandas: For organizing and saving results (e.g., CSV files).
- tifffile: For reading .tif files.

Astronomical Peak-Finding Algorithms
- DAOStarFinder (from photutils)
- Mexican Hat Wavelet (from skimage or astropy)
- SEP (Software for Extracting and Photometering Sources)
- Top-Hat Filtering
- Sigma-Clipped Statistics + Centroiding