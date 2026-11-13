The Python functions, supporting the analyses of the manuscript: A trade-off shaping leaf mesophyll architecture emerges from a stochastic geometry model (Rahimi-Majd et al.).

The functions in "SCellPack_simulation.py" implement the main functionalities required to perform SCellPack simulations.
The script "SCellPack_run.py" contains the code to reproduce the simulations performed in this study using the dataset published by "Borsuk et al. (2022)", available at: https://osf.io/u59zt.

Running "SCellPack_run.py" requires a "data" folder containing the image data and measurements derived from the published dataset and supplementary tables of Borsuk et al. (2022).
The "data" folder must include the following:
1. An Excel file named "pixel_info.xlsx", containing the following columns:
   - "species"
   - "d1"
   - "d2"
   - "d3"
   - "Image Pixel Size (mm)"
   - "Arm Diameter Mean (mm)"
   - "Arm Diameter SD (mm)"
   - "Arm Length Mean (mm)"
   - "Arm Length SD (mm)"
   The columns "d1", "d2", and "d3" define the dimensions of the 3D image data:
   - "d1" and "d3" correspond to the dimensions (in pixels) of the paradermal slices, determined from the corresponding ".tif" image files.
   - "d2" corresponds to the number of paradermal slices and therefore represents the depth dimension of the 3D dataset.
   The remaining columns contain the image pixel size and the mean and standard deviation of arm diameter and arm length for each species. These measurements are     obtained from the supplementary tables of Borsuk et al. (2022).
2. A subfolder for each species, containing the corresponding paradermal slice images in ".tif" format. Each subfolder must have the same name as the corresponding species entry in the `species` column of "pixel_info.xlsx".

The analyses were tested using the following versions of Python and associated libraries: Python (version 3.12.12), numpy (version 2.3.0), pandas (version 2.3.3), scipy (version 1.16.3), shapely (version 2.1.2), rasterio (version 1.5.0), imageio (version 2.37.0), skimage (version 0.26.0), matplotlib (version 3.10.8), sklearn (version 1.8.0).
