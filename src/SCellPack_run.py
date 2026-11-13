from __future__ import annotations
import os
import re
import imageio.v2 as imageio
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from scipy.ndimage import label as scipy_label
from skimage import measure
from skimage.measure import regionprops
import SCellPack_simulation as cp  
from pathlib import Path
import matplotlib.pyplot as plt
#######################
specific_columns = ["Sm", "Sc", "P_total", "p_cell_av", "p_cell_std"]
unitless_columns = ["fIAS", "floss", "N", "A_total", "ncell_comp", "nair_comp", "a_cell_av", "a_cell_std"]
###############################################################################
#################################################### Extract The species info
##############################
def load_species_name(id0, pixel_info_path) :
    """
    To get the species name based on its id number (row number) in the pixel_info table.
    """
    df = pd.read_excel(pixel_info_path)
    species_name = df[df['id']==id0]['species'].values[0]
    return species_name
  
def load_box_params(pixel_info_path, id0, dscale):
    """
    Load box dimensions (a, b, pixel units), pixel size (S_pixel),
    and cell-size ratios (Cmin/Cmax, Dmin/Dmax, f_dscale) for one species id. 
    """
    df = pd.read_excel(pixel_info_path)
    df = df.rename(columns={
        "Image Pixel Size (mm)": "S_pixel",
        "Arm Diameter Mean (mm)": "D_arm_av",
        "Arm Diameter SD (mm) ": "D_arm_std",
        "Arm Length Mean (mm)": "L_arm_av",
        "Arm Length SD (mm)": "L_arm_std", })
    df["L_mm"] = np.sqrt((df["S_pixel"] * df["d1"]) ** 2 + (df["S_pixel"] * df["d3"]) ** 2) 
    row = df[df["id"] == id0].iloc[0]

    a, b = float(row["d1"]), float(row["d3"])
    n_slices = row['d2']
    S_pixel = float(row["S_pixel"])
    L_mm = float(row["L_mm"])
    C = float(row["L_arm_av"] / L_mm) 
    Cerr = float(row["L_arm_std"] / L_mm)
    D = float(row["D_arm_av"] / L_mm)
    Derr = float(row["D_arm_std"] / L_mm)

    Cmin, Cmax = max(C - Cerr, 0.001), C + Cerr
    Dmin, Dmax = D - Derr, D + Derr
    f_dscale = (D / C) if dscale else None

    return {
        "a": a, "b": b, 'n_slices': n_slices, "S_pixel": S_pixel,
        "Cmin": Cmin, "Cmax": Cmax, "Dmin": Dmin, "Dmax": Dmax,
        "C": C, "D": D, "f_dscale": f_dscale,}
##############################
#################################################### Extract the species info
###############################################################################


###############################################################################
#################################################### MicroCT data satas
##############################
def read_slices(folder_path):
    """
    Read the paired original and watershed-segmented TIFF slices in the species data folder.

    Here the file names are in the form of "paradermal_{slice_number}.tif" and
    "paradermal_{slice_number}_watershed.tif" in folder_path.
    To have the correct ordering of the slices, a similar way of naming is expected
    (sorting is based on the first number found in the filename).
    Each list is sorted independently. To have a correct matching for the later analyses,
    numbering must match 1:1 and in the same order between the original and watershed files.
    """
    def _sorted_tifs(suffix_exclude=None, suffix_only=None):
        files = [f for f in os.listdir(folder_path) if f.endswith(".tif")]
        if suffix_only is not None:
            files = [f for f in files if f.endswith(suffix_only)]
        if suffix_exclude is not None:
            files = [f for f in files if not f.endswith(suffix_exclude)]
        return sorted(files, key=lambda x: int(re.search(r"\d+", x).group()))

    file_list_orig = _sorted_tifs(suffix_exclude="watershed.tif")
    file_list_watershed = _sorted_tifs(suffix_only="watershed.tif")

    slices_orig = [imageio.imread(os.path.join(folder_path, f)) for f in file_list_orig]
    slices_watershed = [imageio.imread(os.path.join(folder_path, f)) for f in file_list_watershed]
    return slices_orig, slices_watershed


def compute_data_slice_stats(binary_image_orig, binary_image_watershed, S_pixel):
    """
    The stats for the given data parademal slice.
    """
    aggregate_mask = (binary_image_orig > 0).astype(np.uint8)
    labeled_cells, _ = scipy_label((binary_image_watershed > 0).astype(np.uint8))
    raw = image_stats(aggregate_mask, labeled_cells)
    row = convert_units(raw, S_pixel)
    return row

def compute_data_all_stats(id0, species_name, dir_path, S_pixel):
    """
    To compute the stats for all the paradermal slices of the given species id.
    """
    folder_path = os.path.join(dir_path, species_name)
    slices_orig, slices_watershed = read_slices(folder_path) # slices_watershed may also be segmented here directly, e.g. via a Python segmentation function, instead of using pre-segmented data
    rows = []
    for sn, (orig, watershed) in enumerate(zip(slices_orig, slices_watershed)):
        row = compute_data_slice_stats(orig, watershed, S_pixel)
        rows.append({"id": id0, "species": species_name, "slice": sn, **row})
    return pd.DataFrame(rows) 


def precompute_all_species_stats(species_ids, dir_path, pixel_info_path="pixel_info.xlsx",
                                  out_path="all_species_real_stats.xlsx"):
    """
    To compute_data_all_stats for every id in species_ids and save and return the
    combined result to one file. 
    """
    frames = []
    for id0 in species_ids:
        species_name = load_species_name(id0, pixel_info_path=pixel_info_path)
        box = load_box_params(pixel_info_path, id0, dscale=True) # dacele value does not matter for the stats we need here.
        print(f"Compute image stats {id0} {species_name}")
        frames.append(compute_data_all_stats(id0, species_name, dir_path, box["S_pixel"]))
      
    all_real_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    all_real_df.to_excel(out_path, index=False)
    return all_real_df
##############################
#################################################### MicroCT data satas
###############################################################################


###############################################################################
#################################################### Binray image satas
##############################
def convert_units(raw_stats, S_pixel):
    """
    To convert specific_columns entries (in units of 1/pixel) into both pixel and physical (mm) versions.
    Note: 
        - S_pixel is the pixel size in mm per pixel (mm/pixel).
        - unitless_columns entries pass through unchanged.
        - Conversion from 1/pixel to 1/mm: value_mm = value_px / S_pixel.
    """
    out = {}
    for name in specific_columns:
        out[f"{name}_px"] = raw_stats[name]
        out[f"{name}_mm"] = raw_stats[name] / S_pixel
    for name in unitless_columns:
        out[name] = raw_stats[name]
    return out

def _perimeter_on_border(binary_mask):
    """
    Count of foreground pixels sitting on the image's edge.
    Note: this function assumes "binary_mask" is a binary mask (with background in zero) with at least 2 rows and 2 columns.
    """
    b = binary_mask.astype(float)
    b[1:-1, 1:-1] = np.nan
    return float(np.sum(b > 0))

def image_stats(aggregate_mask, labeled_cells):
    """
    Compute image stats in raw pixel unit.

    aggregate_mask : binary array, 1 = any cell, 0 = background/IAS.
    labeled_cells : segmented array, 0 = background, each cell has its own
        unique positive label (from watershed segmentation for real data,
        or from rasterize_labeled_cells for simulated data).
    """
    aggregate_mask = (aggregate_mask > 0).astype(np.uint8)
    height, width = aggregate_mask.shape
    a_box = float(height * width)

    fIAS = float(np.mean(aggregate_mask == 0))

    ########### Aggregate-level exposed-to-air / touching-border split
    raw_perimeter = measure.perimeter(aggregate_mask, neighborhood=4)
    border_perimeter = _perimeter_on_border(aggregate_mask)
    Sm = (raw_perimeter - border_perimeter) / a_box
    _, ncell_comp = scipy_label(aggregate_mask)
    _, nair_comp = scipy_label(1 - aggregate_mask)

    ########## Per-cell stats, from the segmented image  
    props = regionprops(labeled_cells)
    N = len(props)
    
    if N>0:
        perimeters = np.array([p.perimeter for p in props])
        areas = np.array([p.area for p in props])
        total_perimeter = perimeters.sum()
        total_area = areas.sum()
        P_total = total_perimeter / a_box
        p_cell_av = perimeters.mean() / a_box
        p_cell_std = perimeters.std() / a_box 
        A_total = total_area / a_box
        a_cell_av = areas.mean() / a_box
        a_cell_std = areas.std() / a_box
    else:
        perimeters = areas = np.array([])
        total_perimeter = total_area = P_total = A_total = 0.0
        p_cell_av = p_cell_std = a_cell_av = a_cell_std = np.nan 
   
    fIAS_segmented = 1 - A_total


    Sc = (total_perimeter - (raw_perimeter - border_perimeter) - border_perimeter) / a_box #  Sc = (P_total_raw - Sm_raw - Sm_border_raw) 
    floss = Sc / P_total if P_total > 0 else np.nan
    

    return {
        "fIAS": fIAS, 'fIAS_segmented' : fIAS_segmented, "floss": floss, "N": N, 
        "P_total": P_total, "p_cell_av": p_cell_av, "p_cell_std": p_cell_std,
        "A_total": A_total, "a_cell_av": a_cell_av, "a_cell_std": a_cell_std,
        "Sm": Sm, "Sc": Sc, "ncell_comp": ncell_comp, "nair_comp": nair_comp}
##############################
#################################################### Binray image satas
###############################################################################

###############################################################################
#################################################### Simulation
##############################
def shapely_to_image(units, a, b, Cmin, resolution = 2):
    """
    Shaply units to pixels.
    """
    L = np.sqrt(a**2 + b**2)
    pixel_scale = resolution + int(1 / (Cmin * L)) 
    # The second term is only active if a, b (and Cmin*L) are very small. In our analyses, this pixel-unit convention it's always 0. 
    # Kept as a safety margin for reuse with smaller domains, though very small a, b can still cause other artifacts even with this term active.
    
    grid_size_x = int(a * pixel_scale)
    grid_size_y = int(b * pixel_scale)
    
    if not np.isclose(a / grid_size_x, b / grid_size_y, rtol=1e-5, atol=1e-8):
        raise ValueError("Aspect ratio mismatch between grid dimensions! a and b must be integers")
        
    transform = rasterio.transform.from_origin(0, b, a / grid_size_x, b / grid_size_y)
    shapes = [(unit, i + 1) for i, unit in enumerate(units)]
    labeled = rasterize(shapes, out_shape=(grid_size_y, grid_size_x), transform=transform, fill=0, dtype=np.int32)
    return labeled, pixel_scale
   

def simulated_slice_stats(units, a, b, Cmin, S_pixel, resolution = 2):
    """
    One ensemble member's descriptor row, measured on its rasterized cells.
    """
    labeled_cells, pixel_scale = shapely_to_image(units, a, b, Cmin, resolution)
    aggregate_mask = (labeled_cells > 0).astype(np.uint8)

    raw = image_stats(aggregate_mask, labeled_cells)

    # specific_columns are perimeter/area ratios, so raw units are 1/raster-pixel. 
    # Multiply by pixel_scale to convert to 1/original-image-pixel, the unit convert_units expects.
    for name in specific_columns:
        raw[name] = raw[name] * pixel_scale

    # Convert the specific parameters (with units of "per pixel") to the physical unit 1/L.
    row = convert_units(raw, S_pixel)
    return row, labeled_cells


def run_ensemble_for_shape(shape_t, ensemble, a, b, S_pixel, target_fIAS, Cmin, Cmax, max_attempts, T, dscale, Dmin, Dmax, f_dscale,):
    """
    run SCellPack for the given parameters in the ensemble.
    """
    rows = []
    for ens in range(ensemble):
        grid_res = cp.make_grid_fIAS(a, b, target_fIAS=target_fIAS,
            Cmin=Cmin, Cmax=Cmax, max_attempts=max_attempts, T=T,
            shape_type=shape_t, dscale=dscale, Dmin=Dmin, Dmax=Dmax, f_dscale=f_dscale,)
        units, *_ , sim_status = grid_res
        row, sim_slice = simulated_slice_stats(units, a, b, Cmin, S_pixel)
        row = {"shape_type": shape_t, "ensemble": ens, "sim_status": sim_status, **row} 
        rows.append(row)
    return pd.DataFrame(rows), sim_slice # sim_slice: The slice for the last run in the ensemble
##############################
#################################################### Simulation
###############################################################################

   
###############################################################################
#################################################### Pipeline
##############################
def run_SCellPack_pipeline(id0, image_dir_path, pixel_info_path, shape_types, ensemble, 
    max_attempts = 200, T = 1.0, dscale = True, data_stat_path = None,):
    """
    Run the full per-species pipeline: 
    - get real-image stats for each slice,
    - generate and measure simulated slices for each requested cell shape.
    Note: These simulations are entirely based on the species metadata and 
    microCT images slices from Borsuk et. al (2022) study.

    Parameters
    ----------
    id0 : int
        The species id from 0 to 39.
        Note: the order is based on the alphabetic order of the species according to Borsuk et. al (2022) original data set. Not sorted by arm length.
    image_dir_path : str
        Path to the parent folder containing a subfolder for the given species,
        containing both original ("paradermal_{slices_number}.tif") and
        watershed-segmented ("paradermal_{slices_number}_watershed.tif") paradermal slices.
        Only read when data_stat_path is None.
    pixel_info_path : str
        Path to the pixel_info.xlsx file holding per-species box dimensions,
        pixel size, and arm length/diameter statistics used to calculate
        Cmin/Cmax, Dmin/Dmax, and f_dscale when not given explicitly. 
        This file is directly collected form the metadata and images of the 
        Borsuk et. al (2022) study.
    shape_types : list[str]
        Cell shape types to simulate for every slice.
        Each cell shape morphology is run as a separate ensemble.
        Note: shape_types must be a list of valid shapes defined in SCellPack
        ("circle", "1armed", "2armed180", "2armed", "3armed", "4armed"). 
    ensemble : int
        Number of independent simulated replicates generated per slice per
        shape type.
    max_attempts : int, optional
        Maximum number of consecutive failed placement attempts (rejected
        due to overlap or out-of-bounds clipping) before a slice's
        simulation is stopped early and flagged as failed.
        The default is 200.
    T : float, optional
        Upper bound allowed truncation fraction beacuse of overlap with existing 
        cells or the box boundary for a new cell to be accepted, on a scale of 
        0 (no truncation allowed) to 1 (any truncation smallet than full cell allowed).
        The default is 1.0.
    dscale : bool, optional
        If True, each cell's arm diameter is scaled directly from its arm
        length via f_dscale. If False, arm diameter is instead sampled
        independently and uniformly between Dmin and Dmax.
        The default is True.
    data_stat_path : str, optional
        If None, real-image stats are computed here directly from the imageslices
        under image_dir_path.
        If a path string, real-image stats are instead loaded from that
        precomputed file (as saved once by precompute_all_species_stats)
        and filtered to this id0; image_dir_path's slices are not read in
        that case.
        The default is None.

    Returns
    -------
    pandas DataFrame
        A combined DataFrame with one row per (slice, shape_type, ith ensemble), 
        containing the simulated slice's measured stats alongside
        the corresponding real slice's expected stats.
    """
    ############### Get the id name of the species
    species_name = load_species_name(id0, pixel_info_path=pixel_info_path)
    ###############
    
    ############### Get the species global paramters (the same for all the slices) - these are the configuration parameters for the slice-level simulation
    box = load_box_params(pixel_info_path, id0, dscale=dscale)
    a, b, S_pixel, n_slices = box["a"], box["b"], box["S_pixel"], box['n_slices']
    Cmin = box["Cmin"] # The unitless min cell arm lenght relative to the box diagonal
    Cmax = box["Cmax"] # The unitless max cell arm lenght relative to the box diagonal
    Dmin = box["Dmin"] # The unitless min cell arm diamter relative to the box diagonal
    Dmax = box["Dmax"] # The unitless max cell arm diameter relative to the box diagonal
    f_dscale = box["f_dscale"] # The ratio of D(average)/C(average). This is an alternative way of defining cell diameter, based on Cmin and Cmax rather than Dmin and Dmax
    ###############
        
    ############### To comput the iamges stats for the original slices  
    if data_stat_path is None:
        print(f"[{id0}] {species_name}: computing real-image stats for every slice...")
        real_df = compute_data_all_stats(id0, species_name, image_dir_path, S_pixel)
    else:
        data_df = pd.read_excel(data_stat_path)
        real_df = data_df[data_df["id"] == id0].reset_index(drop=True)
    real_cols = [c for c in real_df.columns if c not in ("id", "species", "slice")] 
    ############### 
    
    ############### To simulate slices based on the image parameters in ensembles
    all_rows = []
    for _, real_row in real_df.iterrows():
        slice0 = real_row["slice"]
        target_fIAS = float(real_row["fIAS"])
        print(f"[{id0}] {species_name} slice {slice0}/{n_slices}: target fIAS = {target_fIAS:.4f}")

        for shape_t in shape_types:
            print(f"  ...ensemble for shape '{shape_t}'")
            sim_df, sim_image = run_ensemble_for_shape(shape_t, ensemble, a, b, S_pixel, target_fIAS, 
                                            Cmin, Cmax, max_attempts, T, dscale, Dmin, Dmax, f_dscale,) # Note: that sim_image is the simulated slice for the last ensemble run.
            sim_df.insert(0, "id", id0)
            sim_df.insert(1, "species", species_name)
            sim_df.insert(2, "T", T)
            sim_df.insert(3, "slice", slice0)
            sim_df.insert(4, "imaged1", a)
            sim_df.insert(5, "imaged2", b)
            sim_df.insert(6, "D/L_scale", dscale) 
            for col in real_cols:
                sim_df[f"expected_{col}"] = real_row[col]
            all_rows.append(sim_df)
    ############### 
    return pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()

###############
def run_independent_SCellPack_simulation(shape_types, ensemble, a, b, target_fIAS,
    Cmin, Cmax, max_attempts=200, T=1.0, dscale=True, Dmin=None, Dmax=None,
    f_dscale=None, S_pixel=1.0, plot = True):
    """
    Run an ensemble simulation directly from explicit arbitrary parameters (no need
    to any species' real image data.)
    
    Note: a and b must be given in pixel units, since the simulation
    geometry (cell placement and rasterization) operates entirely in
    pixel units. 
    Cmin, Cmax, Dmin, Dmax, f_dscale are unitless ratios
    relative to L = sqrt(a**2 + b**2).
    shape_types must be a list of valid shapes defined in SCellPack
    """
    all_rows = []
    for shape_t in shape_types:
        sim_df, sim_image = run_ensemble_for_shape(shape_t, ensemble, a, b, S_pixel, target_fIAS,
            Cmin, Cmax, max_attempts, T, dscale, Dmin, Dmax, f_dscale)
        sim_df.insert(0, "T", T)
        sim_df.insert(1, "imaged1", a)
        sim_df.insert(2, "imaged2", b)
        sim_df.insert(3, "D/L_scale", dscale) 
        all_rows.append(sim_df)
        if plot:
            plt.figure(dpi=200)
            plt.imshow(sim_image)
            plt.xlabel(r'$x_1$')
            plt.ylabel(r'$x_2$')
            plt.title(f'SCellPack sample simulation for {shape_t} cells')
    print(all_rows)
            
    return pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
##############################
#################################################### Pipeline
###############################################################################
if __name__ == "__main__":
    current_dir = Path(__file__).resolve().parent      
    root_dir = current_dir.parent         
    
    species_ids = np.arange(40)
    image_dir_path = root_dir / "data" / "paradermal_slices"
    pixel_info_path = root_dir / "data" / "pixel_info.xlsx"
    data_res_path = root_dir / "results" / "all_species_real_stats.xlsx" 
    data_res_path.parent.mkdir(parents=True, exist_ok=True)
    ################################### A sample SCellPack run with arbitrary parameters
    run_independent_SCellPack_simulation(shape_types=['circle','3armed'], ensemble=1, a=200, b=100, target_fIAS=0.3,
        Cmin=0.03, Cmax=0.05, max_attempts=200, T=1.0, dscale=True, Dmin=None, Dmax=None,
        f_dscale=1.5, S_pixel=1.0, plot = True)
    ################################### Image-based runs 
    cell_shapes = ["circle", "1armed", "2armed180", "2armed", "3armed", "4armed"]
    # cell_shapes = ["circle"]
    ensemble = 10    
    if os.path.exists(data_res_path):
        print(f"{data_res_path} already exists, skipping precompute...")
        all_real_df = pd.read_excel(data_res_path)
    else:
        all_real_df = precompute_all_species_stats(
            species_ids=species_ids, dir_path=image_dir_path,
            pixel_info_path=pixel_info_path, out_path=data_res_path)

    out_dir = root_dir / "results" / "SCellPack_runs"
    os.makedirs(out_dir, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    ### Run the SCellPAck for the given species and truncation fraction
    species_id = 20
    T_fraction = 1
    combined_df = run_SCellPack_pipeline(id0=species_id, image_dir_path=image_dir_path, 
        shape_types=cell_shapes, ensemble=ensemble, T=T_fraction,
        pixel_info_path=pixel_info_path, data_stat_path=data_res_path)
    combined_df.to_csv(out_dir / f"SCellPack_id{species_id}_T{T_fraction}.csv", index=False)
 