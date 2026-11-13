import time
import numpy as np
from shapely.affinity import rotate
from shapely.geometry import LineString, MultiPolygon, Point, Polygon
from shapely.ops import unary_union


############################################################################### 
#*************************** Single cell generation
########################
# Angle (in degrees) of each arm, measured from the cell center, for  armed cell shape. 
cell_arm_angles  = {
    "1armed": [0],
    "2armed": [0, 120],
    "2armed180": [0, 180],
    "3armed": [0, 120, 240],
    "4armed": [0, 90, 180, 270],
    }


def make_circle_cell(x, y, radius):
    """
    Construct a circular cell centered at (x, y) with the given radius.
    """
    return Point(x, y).buffer(radius, resolution=32)


def make_armed_cell(x, y, arm_length, arm_diameter, angles):
    """
    Construct a armed-shaped cell from one or more straight arms extending from
    (x, y), each of length `arm_length` and thickness `arm_diameter`,
    positioned at `angles` (degrees).
    The resulting shape is then randomly rotated about its center.
    """
    center = (x, y)
    rotation_angle = np.random.uniform(0, 360)

    arms = []
    for angle in angles:
        radians = np.radians(angle)
        end_x = x + arm_length * np.cos(radians)
        end_y = y + arm_length * np.sin(radians)
        arm = LineString([center, (end_x, end_y)]).buffer(arm_diameter / 2, resolution=32)
        arms.append(arm)

    shape = unary_union(arms) 
    return rotate(shape, rotation_angle, origin=center)


def _make_cell_shape(shape_type, x, y, radius, L, dscale, f_dscale= None, Dmin= None, Dmax= None):
    """
    Dispatch to the right shape constructor for one cell generation attempt.
    """
    if shape_type == "circle":
        return make_circle_cell(x, y, radius)

    if shape_type not in cell_arm_angles:
        raise ValueError(f"shape_type: {shape_type!r} has not been defined in SCellPack")

    if dscale:
        arm_diameter = radius * f_dscale
    else:
        D = np.random.uniform(Dmin, Dmax)
        arm_diameter = D * L

    return make_armed_cell(x, y, radius, arm_diameter, cell_arm_angles[shape_type])

############################################################################### 
#*************************** Full slice packing
########################
def make_grid_fIAS(a, b, target_fIAS,
    Cmin, Cmax, max_attempts, T, shape_type = "circle",  dscale = True,  Dmin = None,  Dmax = None, f_dscale = None,):
    """   
    Randomly place cells until the fraction of intercellular air space (IAS)
    reaches `target_fIAS`, or `max_attempts` consecutive placements fail.    
    """
    
    rectangle = Polygon([(0, 0), (a, 0), (a, b), (0, b)])
    
    start_time = time.perf_counter()
    L = np.sqrt(a**2 + b**2)

    units = []
    perimeters = []
    areas = []
    total_perimeter = 0.0
    total_area = 0.0
    failed_attempts = 0
    all_attempts = 0
    failed = False

    expected_cell_area = (1 - target_fIAS) * rectangle.area

    while total_area < expected_cell_area:
        if failed_attempts >= max_attempts:
            failed = True
            break

        all_attempts += 1
        x, y = np.random.uniform(0, a), np.random.uniform(0, b)
        C = np.random.uniform(Cmin, Cmax)
        radius = C * L

        new_shape = _make_cell_shape(shape_type, x, y, radius, L, dscale, f_dscale, Dmin, Dmax)
        shape_area = new_shape.area

        # Clip against the box boundary 
        if not rectangle.covers(new_shape):
            intersection = new_shape.intersection(rectangle)
            overlap_fraction = 1 - (intersection.area / shape_area) 
            if overlap_fraction >= T: 
                failed_attempts += 1
                continue
            new_shape = intersection

        ############################## 
        # Check overlap against previously placed cells 
        full_cover = False
        overlap_shapes = []
        for unit in units:
            if unit.covers(new_shape):
                full_cover = True
                break

            if new_shape.covers(unit):
                full_cover = True
                break
            if unit.intersects(new_shape):
                overlap_shapes.append(unit)
        if full_cover:
            failed_attempts += 1
            continue 
        #############################
        #############################
        if overlap_shapes:
            overlap_union = unary_union(overlap_shapes)
            remaining_splits = new_shape.difference(overlap_union)
            
            if remaining_splits.is_empty: 
                failed_attempts += 1
                continue
            
            if isinstance(remaining_splits, MultiPolygon):
                remaining_parts = list(remaining_splits.geoms)
            elif isinstance(remaining_splits, Polygon):
                remaining_parts = [remaining_splits]
            else:
                raise TypeError(f"Unexpected geometry type from difference(): {remaining_splits.geom_type}.")

            largest_part = max(remaining_parts, key=lambda part: part.area)           
            overlap_fraction = 1 - (largest_part.area / shape_area) 
            if overlap_fraction > 0 and overlap_fraction >= T: # The condition "overlap_fraction > 0" is a safety check specifically for handling T=0, for touching-only cells.
                failed_attempts += 1
                continue
            
            remaining_shape = largest_part
        else:
            remaining_shape = new_shape 

        ############################## Accept the cell 
        units.append(remaining_shape)  
        unit_perimeter = remaining_shape.length
        unit_area = remaining_shape.area
        perimeters.append(unit_perimeter)
        areas.append(unit_area)
        total_perimeter += unit_perimeter
        total_area += unit_area
        failed_attempts = 0

    elapsed_time = time.perf_counter() - start_time
    fIAS = 1 - (total_area / rectangle.area)

    return (units, fIAS, total_perimeter, np.mean(perimeters), np.std(perimeters), total_area, np.mean(areas), np.std(areas), all_attempts, elapsed_time, failed)


