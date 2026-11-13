"""
Heightmap-based model (HBM) for pushing granular material.

A pile is represented as a 2D height map (`elevationMap2D`). A push is predicted in two steps:
  1. Geometric displacement: the material inside the area swept by the tool is moved into a
     wedge at the end of the push.
  2. Erosion: wherever the slope between
     neighbouring cells exceeds the angle of repose, material moves downhill until the pile
     is stable.

Main entry point: `elevationMap2D.sweepModelDisplacementAndErosion`.
Also defines the erosion stencils and the error metrics used in the evaluation.

There is no single standard HBM; related heightmap models include, for example:
  - W. Kim, C. Pavlov, A. M. Johnson, "Developing a Simple Model for Sand-Tool Interaction
    and Autonomously Shaping Sand", arXiv:1908.02745, 2019.
  - R. W. Sumner, J. F. O'Brien, J. K. Hodgins, "Animating Sand, Mud, and Snow",
    Computer Graphics Forum, 1999.
"""
import copy
import numpy as np
from numba import jit
import matplotlib.pyplot as plt 
from skimage.morphology import disk
import time 
import math 
import warnings
# Note: hides all warnings, also in every script that imports this module (as in the paper's runs).
warnings.filterwarnings("ignore")

# Note: sets the default figure size for every script that imports this module.
plt.rcParams["figure.figsize"] = (10, 10)


# ---- HBM erosion: setup ----

def generate_erosion_update_path(map_rows, map_cols, N, random=True, random_seed=None):
    """
    Order in which the erosion visits the cells, for each of `N` iterations.

    Returns an int array (N, map_rows * map_cols, 2) of (row, col) pairs. With `random=False`
    the order is fixed (column by column); with `random=True` it is shuffled per iteration.
    """
    np.random.seed(random_seed)

    total = map_rows * map_cols
    paths = np.empty((N, total, 2), dtype=np.int64)
    
    for n in range(N):
        if random:
            order = np.arange(total)
            np.random.shuffle(order)
        else:
            order = np.arange(total)
        coords = np.column_stack((order % map_rows, order // map_rows))
        paths[n] = coords
    return paths

def make_kernel_LUT(map_rows, map_cols, kernel):
    """
    For every cell, precompute the (row, col) of its neighbours under the stencil `kernel`.

    `kernel` is a 2D 0/1 erosion stencil whose center is the cell itself.
    Returns (neighbors[rows, cols, max_neighbors, 2], max_neighbors); neighbours that fall
    outside the map are marked with -1.
    """
    kh, kw = kernel.shape
    ci, cj = kh // 2, kw // 2  # center index

    # Offsets where kernel == 1, excluding the center
    offsets = []
    for i in range(kh):
        for j in range(kw):
            if kernel[i, j] != 0 and not (i == ci and j == cj):
                offsets.append((i - ci, j - cj))

    max_neighbors = len(offsets)
    neighbors = -np.ones((map_rows, map_cols, max_neighbors, 2), dtype=np.int32)

    for i in range(map_rows):
        for j in range(map_cols):
            k = 0
            for di, dj in offsets:
                ni, nj = i + di, j + dj
                if 0 <= ni < map_rows and 0 <= nj < map_cols:
                    neighbors[i, j, k, 0] = ni
                    neighbors[i, j, k, 1] = nj
                    k += 1
    return neighbors, max_neighbors

def parse_erosion_stencil(arg):
    """
    Build an erosion stencil (2D 0/1 array, center excluded) from a name such as 'bloc-5'.

    'lrud-N': cross (left/right/up/down, N cells), size (2N+1)²; 'lrud-1' = 3x3
    'diag-N': both diagonals, size (2N+1)²
    'bloc-N': full N x N block (N odd); 'bloc-5' = 5x5
    'disc-N': disk of radius N; 'disc-3' = 7x7
    """
    if "-" in arg:
        stencil_type, val = arg.split("-", 1)
        N = int(val)
    else:
        stencil_type, N = arg, None

    if stencil_type == "lrud":
        size = 2 * N + 1
        stencil = np.zeros((size, size), dtype=int)
        center = N
        # Cells up to N steps up/down/left/right
        stencil[center - N:center + N + 1, center] = 1
        stencil[center, center - N:center + N + 1] = 1
        stencil[center, center] = 0
        return stencil

    elif stencil_type == "diag":
        size = 2 * N + 1
        stencil = np.zeros((size, size), dtype=int)
        center = N
        for i in range(-N, N+1):
            stencil[center + i, center + i] = 1
            stencil[center + i, center - i] = 1
        stencil[center, center] = 0
        return stencil

    elif stencil_type == "bloc":
        if N % 2 == 0:
            raise ValueError("bloc-N requires odd N")
        stencil = np.ones((N, N), dtype=int)
        stencil[N//2, N//2] = 0
        return stencil

    elif stencil_type == "disc":
        stencil = disk(N)
        stencil[stencil.shape[0]//2, stencil.shape[1]//2] = 0
        return stencil

    else:
        raise ValueError(f"Unknown stencil type: {arg}")

@jit(nopython=True)
def heightmap_3d_iou_directNUMBA(H1, H2, resolution):
    """Volumetric IoU of two height maps: sum of the cell-wise minimum over sum of the cell-wise maximum."""
    if H1.shape != H2.shape:
        raise ValueError("Heightmaps must have the same shape")
    
    if np.any(H1 < 0) or np.any(H2 < 0):
        raise ValueError("Heightmaps must have non-negative values")

    intersection = np.minimum(H1, H2)
    union = np.maximum(H1, H2)
    
    voxel_area = resolution * resolution  # cancels out; kept for clarity
    iou = np.sum(intersection) * voxel_area / (np.sum(union) * voxel_area)
    return iou

def compute_clearance_volume(elevation_data, binary_mask, cell_area):
    """
    Calculates the total volume of material residing inside the binary mask region.
    """
    # Keep only material above 0, multiply by the mask to isolate the clearance zone
    material_in_zone = np.maximum(elevation_data, 0.0) * binary_mask
    return np.sum(material_in_zone) * cell_area

def compute_error_metrics(
        mapA, 
        mapB, 
        resolution=0.001, 
        requested=None, # None means all!
        ):
    
    """
    Compare two height maps of the same shape.

    Metrics (`requested`, default all): 'iou' (volumetric IoU), 'l1' (sum of absolute differences),
    'diff_max_height' (difference of the maximum heights), 'max_abs_diff' (largest cell-wise difference).
    Returns (mapA - mapB, None, {metric: value}). Invalid inputs (NaNs, negative values, different
    shapes, empty maps) give -1 for every metric.
    """
    diff = mapA - mapB
    req = set(requested) if requested is not None else {'l1', 'iou', 'max_abs_diff', 'diff_max_height'}
    metrics = {k: -1.0 for k in req}
    
    # Input checks
    if np.isnan(mapA).any() or np.isnan(mapB).any():
        print(f"Input maps contain NaNs. Returning default metrics.")
        return diff, None, metrics

    if np.any(mapA < 0) or np.any(mapB < 0):
        print(f"Input maps contain negative values. Returning default metrics.")
        return diff, None, metrics

    if mapA.shape != mapB.shape:
        print(f"Input maps have different shapes. Returning default metrics.")
        return diff, None, metrics

    if np.sum(mapA) == 0 or np.sum(mapB) == 0:
        print(f"Input maps have zero sum. Returning default metrics.")
        return diff, None, metrics

    # Metrics
    if 'iou' in req:
        metrics['iou'] = heightmap_3d_iou_directNUMBA(mapA, mapB, resolution)

    if 'l1' in req:
        metrics['l1'] = np.nansum(np.abs(mapA - mapB))


    if 'diff_max_height' in req:
        max_h_a = np.max(mapA)
        max_h_b = np.max(mapB)
        metrics['diff_max_height'] = np.abs(max_h_a - max_h_b)

    if 'max_abs_diff' in req:
        metrics['max_abs_diff'] = np.max(np.abs(diff))
    
    return diff, None, metrics

@jit(nopython=True)
def erodeHeightMapNUMBA(
        height_map,
        paths,
        neighbors,      # precomputed kernel LUT
        max_neighbors,
        height_unit=0.005,
        relative_height_tolerance=0.9,
        N_iter_max=500,
        random_neighbor_order=True,
        hu_divisor=1.0,
        save_history=False,
        random_seed=None):
    """
    HBM erosion: relax a height map until no slope exceeds the angle of repose.

    `height_unit` = resolution * tan(angle of repose) is the largest stable height difference
    between two adjacent cells (scaled by the distance for diagonal neighbours). In each iteration,
    the cells are visited in the order given by `paths[n]`; whenever a cell is higher than a
    stencil neighbour by more than this step (plus `relative_height_tolerance * height_unit`),
    one height unit moves from the cell to the neighbour. Volume is conserved. Stops when no cell
    changes or after `N_iter_max` iterations.

    Returns (height_map_stack, N_iter_actual); the result is height_map_stack[:, :, N_iter_actual]
    (with `save_history`, the stack also holds the intermediate maps).
    """
    np.random.seed(random_seed)

    absolute_height_tolerance = relative_height_tolerance * height_unit
    
    map_copy = height_map.copy()
    map_rows, map_cols = map_copy.shape
    
    # Full stack allocated so callers can index the result by N_iter_actual
    height_map_stack = np.empty((map_rows, map_cols, N_iter_max + 1), dtype=height_map.dtype)
    height_map_stack[:, :, 0] = height_map.copy()

    N_iter_actual = 0
    hu_div = height_unit / hu_divisor
    
    # Neighbour visiting order: an index buffer, and a doubled copy so any start offset can be read without wrap-around
    idx_buffer = np.arange(max_neighbors, dtype=np.int32)
    double_buffer = np.empty(max_neighbors * 2, dtype=np.int32)
    
    for n in range(N_iter_max):
        N_iter_actual += 1
        
        map_copy_normalized = map_copy
        if save_history:
            height_map_stack[:, :, n + 1] = map_copy_normalized

        num_cells_updated = 0
        path = paths[n]

        # Reset index buffer
        for m in range(max_neighbors):
            idx_buffer[m] = m

        # Shuffle the neighbour order once per iteration
        if random_neighbor_order:
            for m in range(max_neighbors - 1, 0, -1):
                swap = np.random.randint(0, m + 1)
                temp = idx_buffer[m]
                idx_buffer[m] = idx_buffer[swap]
                idx_buffer[swap] = temp
                
        for m in range(max_neighbors):
            double_buffer[m] = idx_buffer[m]
            double_buffer[m + max_neighbors] = idx_buffer[m]

        # Visit every cell; all cells share the shuffled neighbour order
        for k in range(path.shape[0]):
            i = path[k, 0]
            j = path[k, 1]

            local_neighbors = neighbors[i, j]
            
            # Cell-dependent start index into the neighbour order, so neighbours are not always tried in the same sequence
            offset = (i * 73 + j * 83) % max_neighbors

            for m in range(max_neighbors):
                curr_idx = double_buffer[m + offset]
                
                ni = local_neighbors[curr_idx, 0]
                nj = local_neighbors[curr_idx, 1]

                if ni < 0:
                    continue  # no neighbour (map border)

                # Allowed height step scales with the distance to the neighbour (diagonals: sqrt(2))
                dx = ni - i
                dy = nj - j
                step_len = math.sqrt(dx * dx + dy * dy)

                height_unit_effective = height_unit * step_len
                height_diff = map_copy[i, j] - map_copy[ni, nj]
                
                if height_diff > height_unit_effective + absolute_height_tolerance:
                    map_copy[i, j] -= hu_div
                    map_copy[ni, nj] += hu_div
                    num_cells_updated += 1

        if num_cells_updated == 0:
            break

    if not save_history:
        height_map_stack[:, :, N_iter_actual] = map_copy

    return height_map_stack, N_iter_actual

def erodeHeightMapNUMBA_Safe(height_map, paths, neighbors, max_neighbors, height_unit, relative_height_tolerance, N_iter_max, random_neighbor_order, random_seed, save_history=False):
    """
    Call the erosion kernel above with inputs cast to the exact types it is compiled for.

    Avoids Numba recompilation and type errors. `random_seed=None` becomes 42.
    """
    # Arrays: contiguous, fixed dtypes
    h_map_safe = np.require(height_map, dtype=np.float32, requirements=['C'])
    paths_safe = np.require(paths, dtype=np.int64, requirements=['C'])
    neigh_safe = np.require(neighbors, dtype=np.int32, requirements=['C'])
    
    # Scalars: plain Python types
    max_n_safe = int(max_neighbors)         
    hu_safe = float(height_unit)            
    tol_safe = float(relative_height_tolerance)
    iters_safe = int(N_iter_max)
    rand_safe = bool(random_neighbor_order)
    random_seed_safe = int(random_seed) if random_seed is not None else 42
    save_hist_safe = bool(save_history)

    return erodeHeightMapNUMBA(
        h_map_safe, 
        paths_safe, 
        neigh_safe, 
        max_n_safe, 
        height_unit=hu_safe, 
        relative_height_tolerance=tol_safe, 
        N_iter_max=iters_safe, 
        random_neighbor_order=rand_safe, 
        random_seed=random_seed_safe,
        save_history=save_hist_safe
    )

def erodeHeightMapNUMBA_Warmup(random_seed):
    """
    Compile the Numba erosion once on a small dummy map, so later (timed) runs exclude compilation.

    Returns the time this first call took (s).
    """
    print("Warming up Numba JIT compiler...")
    
    # Small dummy pile: a cylinder in the middle of a 10 cm map
    warmup_map = elevationMap2D()
    warmup_map.setMetricConfig(map_width=0.1, resolution=0.005, label='warmup')
    
    warmup_map.overrideWithCircleElevation(
        const_elevation=0.05,
        xy_centers=[np.array([0.0, 0.0])],
        radius=0.02
    )
    
    rs, cs = warmup_map.elevation_data.shape
    warmup_iters = 5
    
    dummy_stencil = parse_erosion_stencil("lrud-1")
    neighbors, max_neighbors = make_kernel_LUT(rs, cs, dummy_stencil)
    paths = generate_erosion_update_path(rs, cs, N=warmup_iters, random=False, random_seed=random_seed)
    erosion_height_unit = warmup_map.computeHeightUnit(angle_of_repose_deg=30.0, resolution=0.005)
    
    time_for_jit = time.time()
    erodeHeightMapNUMBA_Safe(
        height_map=warmup_map.elevation_data, 
        paths=paths, 
        neighbors=neighbors, 
        max_neighbors=max_neighbors,
        height_unit=erosion_height_unit, 
        relative_height_tolerance=0.95, 
        N_iter_max=warmup_iters, 
        random_neighbor_order=True,
        random_seed=random_seed
    )
    time_for_jit = time.time() - time_for_jit
    print(f"Numba JIT warm-up completed in {time_for_jit:.4f} seconds.")
    return time_for_jit

# ---- Height map ----

class samplerDrv2D():
    """Sample cells of a 2D grid, uniformly or weighted by a distribution (kept up to date by elevationMap2D.setData)."""
    def __init__(self):
        self.drv_2d = None
        self.drv_flattened = None
        self.rows = None
        self.cols = None

    def setDrv2D(self, drv_2d):
        self.drv_2d = drv_2d
        self.rows = self.drv_2d.shape[0]
        self.cols = self.drv_2d.shape[1]
        self.drv_flattened = self.drv_2d.flatten()

    def sampleDrvUniform(self):
        # sample row-col uniformly
        row_sample = int(np.random.rand() * self.rows)
        col_sample = int(np.random.rand() * self.cols)
        return [row_sample, col_sample]

    def sampleDrvWeighted(self):
        # sample row-col according to the underlying distribution
        sampled_index_flat = np.random.choice(
            a=self.drv_flattened.size, p=self.drv_flattened)
        [row_sample, col_sample] = np.unravel_index(
            sampled_index_flat, self.drv_2d.shape)
        return [row_sample, col_sample]

class elevationMap2D():
    """
    Square height map centered at the origin, with the metric (x, y) position of every cell in `XY_lut`.

    Also holds the HBM push model (geometric displacement and the full push prediction).
    """
    def __init__(self):
        # elevation data.
        self.elevation_data = None
        self.map_shape = None

        # metric parameters.
        self.map_width = None
        self.resolution = None
        self.label = None

        # metric mapping (same dim as elevation_data)
        self.XY_lut = None

        # Sampler.
        self.sampler = samplerDrv2D()

        # other parameters
        self.color_map_elevation = 'viridis'

    def setMetricConfig(self, map_width, resolution, label):
        """Set the map size (`map_width`, m) and cell size (`resolution`, m); creates `XY_lut` and an all-zero map."""
        self.map_width = map_width
        self.resolution = resolution
        self.label = label
        self.map_shape = (int(self.map_width/self.resolution), int(self.map_width/self.resolution))
        # Create XY lookup table.
        x_linspace = np.linspace(
            -self.map_width/2.0, 
            self.map_width/2.0, 
            self.map_shape[1])
        y_linspace = np.linspace(
            -self.map_width/2.0, 
            self.map_width/2.0, 
            self.map_shape[0])
        X_meshgrid, Y_meshgrid = np.meshgrid(x_linspace, y_linspace)
        self.XY_lut = np.dstack((X_meshgrid, Y_meshgrid)) # The alignment of XY <-> row/col sets the coordinate conventions.
        self.setData(np.zeros(self.map_shape))

    def setData(self, elevation_data):
        """Set the heights (shape must match the map); also updates `volume` and the sampler."""
        if elevation_data.shape != self.map_shape:
            raise ValueError("Elevation data shape does not match map shape!")
        self.elevation_data = elevation_data
        if np.sum(elevation_data) == 0.0:
            # Empty map: use ones for volume and sampler to avoid NaNs (the stored heights stay zero)
            elevation_data = np.ones(elevation_data.shape)
        self.volume = self.resolution*self.resolution*np.sum(elevation_data)
        self.sampler.setDrv2D(elevation_data/np.sum(elevation_data))

    def XYToRowCol(self, x, y, discrete=True):
        """Convert a metric (x, y) to (row, col); with `discrete`, rounded down and clipped to the map."""
        if discrete:
            row = int(self.map_shape[0]/2.0 - y/self.resolution)
            col = int(x/self.resolution + self.map_shape[1]/2.0)
            row = min(max(0, row), self.map_shape[0]-1)
            col = min(max(0, col), self.map_shape[1]-1)
            return [row, col]
        else:
            row = self.map_shape[0]/2.0 - y/self.resolution
            col = x/self.resolution + self.map_shape[1]/2.0
            return [row, col]

    def overrideWithConstantElevation(self, const_elevation):
        """Set every cell to `const_elevation`."""
        self.setData(const_elevation * np.ones(self.map_shape))

    def overrideWithCircleElevation(self, const_elevation, xy_centers, radius=1.0):
        """Set the heights to `const_elevation` inside one or more circles (centers `xy_centers`, radii `radius`) and 0 elsewhere."""
        x = self.XY_lut[:, :, 0]
        y = self.XY_lut[:, :, 1]
        elev_const = np.zeros(self.map_shape)
        
        # One value per circle
        if isinstance(const_elevation, (int, float)):
            const_elevation = [const_elevation] * len(xy_centers)
        if isinstance(radius, (int, float)):
            radius = [radius] * len(xy_centers)
        elif len(radius) != len(xy_centers):
            raise ValueError("Length of radius must match the number of xy_centers.")
        
        for xy_center, elevation, r in zip(xy_centers, const_elevation, radius):
            mask = (x-xy_center[0])**2 + (y-xy_center[1])**2 <= r**2
            elev_const[mask] = elevation
        
        self.setData(elev_const)

    def overrideWithRectangleElevation(self,
                const_elevation=None,
                xy_bottomleft=None,
                xy_topright=None):
        """
        Set the heights to `const_elevation` inside an axis-aligned rectangle and 0 elsewhere.

        Args:
            const_elevation (float): The constant elevation value.
            xy_bottomleft (tuple): The bottom-left corner of the rectangle (x, y).
            xy_topright (tuple): The top-right corner of the rectangle (x, y).
        """
        if const_elevation is None or xy_bottomleft is None or xy_topright is None:
            raise ValueError("All parameters (const_elevation, xy_bottomleft, xy_topright) must be provided.")

        x = self.XY_lut[:, :, 0]
        y = self.XY_lut[:, :, 1]

        mask = (x >= xy_bottomleft[0]) & (x <= xy_topright[0]) & \
               (y >= xy_bottomleft[1]) & (y <= xy_topright[1])

        elev_const = np.zeros(self.map_shape)
        elev_const[mask] = const_elevation

        self.setData(elev_const)


    @staticmethod
    def get_sweep_mask(XY_lut, grid_shape, xy_sweep_start, xy_sweep_end, spatula_width):
        """
        Compute a boolean mask of a sweep rectangle defined by start, end, and width.

        Parameters
        ----------
        XY_lut : ndarray of shape (rows, cols, 2)
            Lookup table giving (x,y) coordinates for each grid cell.
        grid_shape : tuple
            Shape of the grid (rows, cols).
        xy_sweep_start : array-like, shape (2,)
            XY coordinates of rectangle start.
        xy_sweep_end : array-like, shape (2,)
            XY coordinates of rectangle end.
        spatula_width : float
            Width of the rectangle (in XY units).

        Returns
        -------
        mask : ndarray of shape grid_shape, dtype=bool
            Boolean mask: True where grid cells lie inside the rectangle.
        """

        # Unit vectors along and across the push
        n_long = np.array(xy_sweep_end) - np.array(xy_sweep_start)
        length = np.linalg.norm(n_long)
        if length == 0:
            print("Warning: Sweep length is zero!")
            return np.zeros(grid_shape, dtype=bool)
        n_long /= length
        n_lat = np.array([n_long[1], -n_long[0]])
        half_width = spatula_width / 2.0

        vec = XY_lut - xy_sweep_start  # shape (rows, cols, 2)

        proj_long = vec[..., 0] * n_long[0] + vec[..., 1] * n_long[1]
        proj_lat  = vec[..., 0] * n_lat[0]  + vec[..., 1] * n_lat[1]

        mask = (proj_long >= 0) & (proj_long <= length) & (np.abs(proj_lat) <= half_width)

        return mask

    def computeHeightUnit(self, angle_of_repose_deg, resolution):
        """Height unit for the erosion: resolution * tan(angle of repose), i.e. the largest stable height step between adjacent cells (rounded to 1e-6 m)."""
        angle_of_repose_rad = np.deg2rad(angle_of_repose_deg)
        height_unit = resolution*np.tan(angle_of_repose_rad)
        height_unit = np.round(height_unit, 6)
        return height_unit

    @staticmethod
    def sweepModelDisplacement(height_map_src, 
                               xy_start=np.array([0.0, 0.0]), 
                               xy_end=np.array([0.1, 0.0]), 
                               sweep_width=0.05, 
                               dump_base_angle_deg=45.0,
                               sweep_shortening=False,):
        """
        Geometric displacement model of a push.

        All material inside the swept rectangle (from `xy_start` to `xy_end`, width `sweep_width`)
        is removed and placed as a wedge just beyond `xy_end`: highest at the end of the push and
        sloping down at `dump_base_angle_deg`, with the same volume as the removed material.
        `sweep_shortening` is kept for compatibility and has no effect.

        Returns a dict with the predicted heights ('elevmap_predicted'), the wedge ('elevmap_dump',
        'elev_dump_prism'), the displaced volume and the number of (occupied) cells in the swept area.
        """
        base_map_src = copy.deepcopy(height_map_src)

        assert sweep_width >= 2.0*base_map_src.resolution, "sweepModelDisplacement: Sweep width must be greater than 2.0*resolution!"

        elevmap_pred = copy.deepcopy(base_map_src)

        # Cells inside the swept rectangle
        sweep_mask = elevationMap2D.get_sweep_mask(
            base_map_src.XY_lut,
            grid_shape=height_map_src.elevation_data.shape, 
            xy_sweep_start=xy_start, 
            xy_sweep_end=xy_end, 
            spatula_width=sweep_width)

        sweep_distance_shortened = np.linalg.norm(xy_end - xy_start)

        # Remove the swept material
        elevmap_pred.elevation_data[sweep_mask] = 0.0

        swept_sum = base_map_src.elevation_data[sweep_mask].sum()
        swept_volume = swept_sum*base_map_src.resolution*base_map_src.resolution

        # Nothing to push: return the map unchanged
        if swept_sum <= 1e-6:
            print("Displacement model: Early termination since swept volume is negligible.")
            empty_map = np.zeros_like(base_map_src.elevation_data)
            return {
                'elevmap_predicted': base_map_src.elevation_data,
                'elevmap_dump': empty_map,
                'displaced_volume': 0.0,
                'num_cells_inside_sweep_patch': np.sum(sweep_mask),
                'num_occupied_cells_inside_sweep_patch': 0,
                'elev_dump_prism': empty_map,
                'displacement_map_hi_res': base_map_src.elevation_data,
                'sweep_distance_shortened': np.linalg.norm(xy_end - xy_start),
            }

        num_cells_inside_sweep_patch = np.sum(sweep_mask)
        dummy_helper = base_map_src.elevation_data[sweep_mask]

        num_occupied_cells_inside_sweep_patch = np.sum(dummy_helper > 0.0)
        
        # --- Wedge of displaced material in front of the push end ---
        elevmap_dump = copy.deepcopy(base_map_src)
        elevmap_dump.overrideWithConstantElevation(0.0)

        # Wedge length and height from its volume (= swept volume), width and angle
        dump_base_angle_rad = np.deg2rad(dump_base_angle_deg)
        dump_prism_pile_length = np.sqrt(2.0 * swept_volume / (np.tan(dump_base_angle_rad) * sweep_width))
        dump_prism_pile_height = dump_prism_pile_length * np.tan(dump_base_angle_rad)

        x = elevmap_dump.XY_lut[:, :, 0]
        y = elevmap_dump.XY_lut[:, :, 1]

        # Unit vectors along and across the push
        dx = xy_end[0] - xy_start[0]
        dy = xy_end[1] - xy_start[1]
        sweep_length = np.hypot(dx, dy)
        e_sweep = np.array([dx, dy]) / sweep_length
        e_lat   = np.array([-dy, dx]) / sweep_length

        # Coordinates relative to the push end: along the push (0 at the end) and across it
        x_rel_end = x - xy_end[0]
        y_rel_end = y - xy_end[1]
        s_from_end = x_rel_end * e_sweep[0] + y_rel_end * e_sweep[1]
        t = x_rel_end * e_lat[0] + y_rel_end * e_lat[1]

        # Wedge: linear from full height at the push end down to 0 at distance dump_prism_pile_length
        ramp_mask = (s_from_end >= 0.0) & (s_from_end <= dump_prism_pile_length)
        elev_dump_prism = np.zeros_like(s_from_end)
        elev_dump_prism[ramp_mask] = dump_prism_pile_height * (1.0 - s_from_end[ramp_mask] / dump_prism_pile_length)

        # Limit to the tool width
        elev_dump_prism[np.abs(t) > sweep_width / 2.0] = 0.0

        elev_dump_prism = np.clip(elev_dump_prism, 0.0, None)
        elev_dump_prism = np.round(elev_dump_prism, 5)
        mask_dump_prism = elev_dump_prism > 0.0

        # Scale the wedge to exactly the swept volume (after discretization)
        dump_sum = elev_dump_prism.sum()
        if dump_sum > 0.0:
            elev_dump_prism = elev_dump_prism / dump_sum * swept_sum

        # Add the wedge to the map
        elevmap_dump.elevation_data += elev_dump_prism
        elevmap_pred.elevation_data += elev_dump_prism     
        
        absolute_sum_difference = base_map_src.elevation_data.sum() - elevmap_pred.elevation_data.sum()        
        max_num_retries = 100
        num_retry = 0
        if absolute_sum_difference > 0.0:
            # Remaining volume difference: spread it in equal parts over the wedge cells
            # (if that does not converge, put the rest at the push end)
            number_of_units_to_add = elevmap_pred.elevation_data.shape[0]*elevmap_pred.elevation_data.shape[1]
            correction_unit = absolute_sum_difference/number_of_units_to_add
            cnt_added = 0          
            while cnt_added < number_of_units_to_add:
                for i in range(elevmap_pred.elevation_data.shape[0]):
                    for j in range(elevmap_pred.elevation_data.shape[1]):
                        if mask_dump_prism[i,j]:
                            if elevmap_pred.elevation_data[i,j] + correction_unit >= 0.0:
                                elevmap_pred.elevation_data[i,j] += correction_unit
                                cnt_added += 1
                                if cnt_added == number_of_units_to_add:
                                    break
                            if cnt_added == number_of_units_to_add:
                                break
                    if cnt_added == number_of_units_to_add:
                        break
                if cnt_added == number_of_units_to_add:
                    break
                num_retry += 1
                if num_retry > max_num_retries:
                    rc_end = base_map_src.XYToRowCol(xy_end[0], xy_end[1])
                    elevmap_pred.elevation_data[rc_end[0], rc_end[1]] += correction_unit*(number_of_units_to_add-cnt_added)
                    break
        if elevmap_pred.elevation_data.sum() == 0.0:
            assert False, "The predicted profile has zero volume!"
        
        elev_map_pred_ds = elevmap_pred.elevation_data
        output_args = {
            'elevmap_predicted': elev_map_pred_ds,
            'elevmap_dump': elevmap_dump.elevation_data,
            'displaced_volume': swept_volume,
            'num_cells_inside_sweep_patch': num_cells_inside_sweep_patch,
            'num_occupied_cells_inside_sweep_patch': num_occupied_cells_inside_sweep_patch,
            'elev_dump_prism': elev_dump_prism,
            'displacement_map_hi_res': elevmap_pred.elevation_data,
            'sweep_distance_shortened': sweep_distance_shortened,
        }
        return output_args

    @staticmethod
    def sweepModelDisplacementAndErosion(height_map_src, 
                                         xy_sweep_start=None, 
                                         xy_sweep_end=None, 
                                         sweep_width=None, 
                                         angle_of_repose_deg=None,
                                         dump_base_angle_deg=None, 
                                         relative_height_tolerance=0.1, 
                                         N_iter_max=None,
                                         only_erosion=False,
                                         sweep_shortening=False,
                                         erosion_random_neighbor_order=False,
                                         erosion_stencil=None,
                                         random_seed=None,
                                         save_history=False):
        """
        HBM prediction of one push: geometric displacement, then erosion.

        Three cases:
          - `N_iter_max == 0`: displacement only (variants without erosion)
          - `only_erosion`: erosion only, no push (e.g. the pile settling after the bucket lift)
          - otherwise: displacement followed by erosion (variants with erosion)
        `dump_base_angle_deg` sets the wedge angle of the displacement; `angle_of_repose_deg` and
        `erosion_stencil` set the erosion.

        Returns a dict with the prediction ('elev_map_predicted', an elevationMap2D), the erosion
        stack and number of iterations ('map_stack', 'N_iter_actual'), the displaced volume and
        the runtimes ('time_for_setup', 'time_for_push', 'time_for_erosion').
        """
        height_unit = height_map_src.computeHeightUnit(angle_of_repose_deg, height_map_src.resolution)
        rs, cs = height_map_src.elevation_data.shape

        # Cache of update paths and neighbour tables, per map size, iteration count and stencil
        if not hasattr(elevationMap2D, '_sim_cache'):
            elevationMap2D._sim_cache = {}

        cache_key = (rs, cs, N_iter_max, str(erosion_stencil))

        time_setup_start = time.time()
        
        if cache_key in elevationMap2D._sim_cache:
            cached_data = elevationMap2D._sim_cache[cache_key]
            random_lookup_paths = cached_data['paths']
            neighbors = cached_data['neighbors']
            max_neighbors = cached_data['max_neighbors']
        else:
            # Fixed visiting order (random=False)
            random_lookup_paths = generate_erosion_update_path(rs, cs, N_iter_max, random=False, random_seed=random_seed)
            neighbors, max_neighbors = make_kernel_LUT(rs, cs, erosion_stencil)
            
            elevationMap2D._sim_cache[cache_key] = {
                'paths': random_lookup_paths,
                'neighbors': neighbors,
                'max_neighbors': max_neighbors
            }
        
        time_for_setup = time.time() - time_setup_start

        time_for_push = 0.0
        time_for_erosion = 0.0

        if N_iter_max == 0:
            # Case 1: displacement only
            time_for_push = time.time()
            displacement_result = elevationMap2D.sweepModelDisplacement(
                height_map_src=height_map_src,
                xy_start=xy_sweep_start, 
                xy_end=xy_sweep_end,
                sweep_width=sweep_width,
                dump_base_angle_deg=dump_base_angle_deg,
                sweep_shortening=sweep_shortening)
            time_for_push = time.time() - time_for_push
            elev_prd_map = copy.deepcopy(height_map_src)
            elev_prd_map.setData(displacement_result['elevmap_predicted'])
            output_args = {
                'elev_map_predicted': elev_prd_map,
                'map_stack': np.zeros((height_map_src.elevation_data.shape[0], height_map_src.elevation_data.shape[1], 1)),
                'dump_map': displacement_result['elevmap_dump'],
                'N_iter_actual': 0,
                'displaced_volume': displacement_result['displaced_volume'],
                'displacement_map_hi_res': displacement_result['displacement_map_hi_res'],
                'elevmap_dump': displacement_result['elevmap_dump'],
                'sweep_distance_shortened': displacement_result['sweep_distance_shortened'],
                'time_for_setup': time_for_setup,
                'time_for_push': time_for_push,
                'time_for_erosion': time_for_erosion,
            }
            return output_args
        if only_erosion:
            # Case 2: erosion only
            time_for_erosion = time.time()
            map_stack, N_iter_actual = erodeHeightMapNUMBA_Safe(
                height_map_src.elevation_data, 
                paths=random_lookup_paths,
                neighbors=neighbors, max_neighbors=max_neighbors,
                height_unit = height_unit, 
                relative_height_tolerance=relative_height_tolerance,
                N_iter_max=N_iter_max,
                random_neighbor_order=erosion_random_neighbor_order, random_seed=random_seed, save_history=save_history)
            time_for_erosion = time.time() - time_for_erosion
            # Keep only the iterations that were run
            map_stack = map_stack[:, :, :N_iter_actual+1]
            elev_prd_map = copy.deepcopy(height_map_src)
            elev_prd_map.setData(map_stack[:, :, -1])
            output_args = {
                'elev_map_predicted': elev_prd_map,
                'map_stack': map_stack,
                'dump_map': np.zeros(height_map_src.elevation_data.shape),
                'N_iter_actual': N_iter_actual,
                'displaced_volume': 0.0,
                'num_cells_inside_sweep_patch': 0,
                'num_occupied_cells_inside_sweep_patch': 0,
                'sweep_distance_shortened': 0.0,
                'time_for_setup': time_for_setup,
                'time_for_push': time_for_push,
                'time_for_erosion': time_for_erosion,
            }
            return output_args
        else:
            # Case 3: displacement, then erosion
            time_for_push = time.time()
            displacement_result = elevationMap2D.sweepModelDisplacement(
                height_map_src=height_map_src,
                xy_start=xy_sweep_start, 
                xy_end=xy_sweep_end,
                sweep_width=sweep_width,
                dump_base_angle_deg=dump_base_angle_deg,
                sweep_shortening=sweep_shortening)
            time_for_push = time.time() - time_for_push
            time_for_erosion = time.time()
            map_stack, N_iter_actual = erodeHeightMapNUMBA_Safe(
                displacement_result['elevmap_predicted'], 
                paths=random_lookup_paths,
                neighbors=neighbors, max_neighbors=max_neighbors,
                height_unit = height_unit, 
                relative_height_tolerance=relative_height_tolerance,
                N_iter_max=N_iter_max,
                random_neighbor_order=erosion_random_neighbor_order, random_seed=random_seed, save_history=save_history)
            time_for_erosion = time.time() - time_for_erosion
            # Keep only the iterations that were run
            map_stack = map_stack[:, :, :N_iter_actual+1]
            elev_prd_map = copy.deepcopy(height_map_src)
            elev_prd_map.setData(map_stack[:, :, -1])
            output_args = {
                'elev_map_predicted': elev_prd_map,
                'map_stack': map_stack,
                'dump_map': displacement_result['elevmap_dump'],
                'N_iter_actual': N_iter_actual,
                'displaced_volume': displacement_result['displaced_volume'],
                'num_cells_inside_sweep_patch': displacement_result['num_cells_inside_sweep_patch'],
                'num_occupied_cells_inside_sweep_patch': displacement_result['num_occupied_cells_inside_sweep_patch'],
                'elevmap_dump': displacement_result['elevmap_dump'],
                'time_for_setup': time_for_setup,
                'time_for_push': time_for_push,
                'time_for_erosion': time_for_erosion,
            }
            return output_args