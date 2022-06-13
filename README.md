# Master
The raw data and code used to extract magnetization data from three angular hysteresis measurements, as well as some imaged as-grown deformed pinwheel states, are found here. 


All functions the functions used to extract magnetization information from the raw MFM-data is found in the img.py-file, which defines the image-class. This class contains almost all functions needed to process the data. Creating the vignette for windowing of the MFM-images and the mask for extracting only periodicities along the diagonals of the image (used in the 2D FFT image processing procedure) are done in separate functions, not in the image-class, which are also defined in img.py.

Note that before the raw MFM-data can be processed using the 1D and 2D FFT image processing procedures, they must be preprocessed (using the SPM analysis software Gwyddion, or similar) and reshaped. Reshaping is done using the simple-reshape.py-script. 

A proper guide for using the code to analyze the data has not been completely finished. Chapter 4.4 in my master's thesis should be enough to understand how the image processing procedures work. This chapter will be reformulated as a "user's guide" for this code, and uploaded to this repository, but not before grading of the master's thesis is finished. 

The raw MFM-data is included in the zip-file raw-data.zip. The raw data is sorted into folders based on the dates on which they were taken. Each date-folder should have its own log-file, stating scan-speeds and lift heights, as well as information on the imaged structures and the applied fields.

The reshaped data of the as-grown images and the three angular hysteresis measurements is included in the hys-data.zip-dile, and are found in their respective folders. The folders for the three angular hysteresis measurements also include jupyter-notebooks, in which all analysis and plotting is done. The results are also found in text-files in these folders, and so are all produced plots. Each subfolder of the hysteresis measurement-folders includes data corresponding to a single deformed ASI-structure, and these should include a log-file stating the date which the images were taken, making it easy to find the binary raw data in the dated folders. 

The simulated MFM-data used to test the imaging procedures is also included, in the zip-file sim-data.zip. All simulated MFM-data are produced using the Mumax3 micromagnetic simulation package. 
