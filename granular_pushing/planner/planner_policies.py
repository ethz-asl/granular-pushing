from abc import ABC, abstractmethod
import numpy as np
from hbm_model import elevationMap2D, parse_erosion_stencil, compute_clearance_volume
import time

def get_planner(config: dict):
    """Factory to initialize the selected planner with consistent nested config."""
    # 1. Extract Top-Level Blocks
    hbm_defaults = config["hbm"]
    planner_cfg = config["planner"]
    p_type = planner_cfg["planner_type"]
    mapping_cfg = config["mapping"]
    
    # 2. Extract Shared reach constraints
    bounds_cfg = {
        "start_bounds": planner_cfg["start_bounds"],
        "end_bounds": planner_cfg["end_bounds"]
    }

    if p_type == "random":
        algo_cfg = planner_cfg["random"]
        return RandomPlanner(algo_cfg, bounds_cfg)
    
    elif p_type == "clearance":
        algo_cfg = planner_cfg["clearance"]
        
        return ClearancePlanner(algo_cfg, bounds_cfg, hbm_defaults, mapping_cfg)

    else:
        raise ValueError(f"Planner type '{p_type}' not supported.")

class BasePlanner(ABC):
    def __init__(self, algo_block, bounds_block):
        self.params = algo_block
        # Bounds are now passed from the shared planner level
        self.start_bounds = np.array(bounds_block["start_bounds"])
        self.end_bounds = np.array(bounds_block["end_bounds"])

    @abstractmethod
    def plan(self, current_hmap: elevationMap2D, target_hmap: elevationMap2D) -> dict:
        pass

class RandomPlanner(BasePlanner):
    def plan(self, current_hmap: elevationMap2D, target_hmap: elevationMap2D):
        s_x = np.random.uniform(*self.start_bounds[0])
        s_y = np.random.uniform(*self.start_bounds[1])
        e_x = np.random.uniform(*self.end_bounds[0])
        e_y = np.random.uniform(*self.end_bounds[1])
        return {"PRE": np.array([s_x, s_y]), "POST": np.array([e_x, e_y])}

    
class ClearancePlanner(BasePlanner):
    '''
    Plans pushes that clear material out of a target region (a binary mask, e.g. a circle).
    Candidates: radial pushes around a centre point (`radial_config`); pushes shorter than
    `min_sweep_len` are discarded. 'exhaustive': predicts every candidate with the HBM and picks
    the one that leaves the least volume in the region. 'random_blind': picks a random candidate.
    '''

    def __init__(self, algo_block, bounds_block, merged_model_params, mapping_cfg):
        super().__init__(algo_block, bounds_block)
        
        # Physics & Model
        self.model_params = merged_model_params
        self.erosion_stencil = parse_erosion_stencil(self.model_params["erosion_stencil"])
        
        # Options: 
        self.sweep_patterns = algo_block.get('sweep_patterns', None) 
        self.min_sweep_len = float(algo_block.get('min_sweep_len', 0.01))

        # Radial sweeps: centre and radius sets
        self.radial_config = algo_block.get('radial_config', {})

        # Selection strategy: 'exhaustive' (default) or 'random_blind'
        self.selection_strategy = algo_block.get('selection_strategy', 'exhaustive').lower()

    def _generate_candidates(self, hmap) -> tuple:
        print(f"Generating candidates for patterns: {self.sweep_patterns}")
        candidates = []
        colors = [] # Track colors in parallel
        
        # --- Generate Sweeps ---
        for pattern_name in self.sweep_patterns:
            pattern_key = pattern_name.lower() 
            
            # --- RADIAL SWEEPS ---
            if 'radial' in pattern_key:
                if not self.radial_config:
                    print("[WARNING] Radial pattern requested but 'radial_config' is missing.")
                    continue
                
                # FORCE FLOATS to prevent ruamel.yaml crashes
                cx = float(self.radial_config['xy_center'][0])
                cy = float(self.radial_config['xy_center'][1])
                radius_sets = self.radial_config['radius_sets']
                sweep_width = float(self.model_params['sweep_width'])
                
                # Find the absolute maximum radius to compute outer circumference
                max_radius = max([max(float(r_set[0]), float(r_set[1])) for r_set in radius_sets])
                
                # Circumference = 2 * pi * r
                circumference = 2 * np.pi * max_radius
                
                # Number of subdivisions (ceil to ensure no gaps at the outer edge)
                num_sweeps = int(np.ceil(circumference / sweep_width))
                
                # Generate angles (endpoint=False prevents overlapping the 360/0 degree sweep)
                angles = np.linspace(0, 2 * np.pi, num_sweeps, endpoint=False)
                
                for angle in angles:
                    cos_a = np.cos(angle)
                    sin_a = np.sin(angle)
                    
                    for r_set in radius_sets:
                        r_start = float(r_set[0])
                        r_end = float(r_set[1])
                        color = str(r_set[2]) if len(r_set) > 2 else 'black' # Extract color safely
                        
                        start_x = cx + r_start * cos_a
                        start_y = cy + r_start * sin_a
                        end_x   = cx + r_end * cos_a
                        end_y   = cy + r_end * sin_a
                        
                        candidates.append(np.array([start_x, start_y, end_x, end_y]))
                        colors.append(color)

        cand_arr = np.array(candidates) if len(candidates) > 0 else np.empty((0, 4))
        return cand_arr, colors

    def _filter_sweeps(self, candidates: np.ndarray, colors: list) -> tuple:
        print(f"Filtering {len(candidates)} candidates based on minimum sweep length: {self.min_sweep_len}")
        """Discard sweeps that are too short."""
        if len(candidates) == 0:
            return candidates, colors

        lengths = np.linalg.norm(candidates[:, 2:] - candidates[:, :2], axis=1)
        valid_mask = lengths >= self.min_sweep_len
        
        valid_candidates = candidates[valid_mask]
        valid_colors = [c for i, c in enumerate(colors) if valid_mask[i]]
        
        num_discarded = np.sum(~valid_mask)
        print(f"Discarded {num_discarded} candidates. {len(valid_candidates)} remain.")
        
        return valid_candidates, valid_colors

    def _predict(self, current_hmap, sweep_vec):
        """Simulate the sweep physics."""
        p = self.model_params

        time_before = time.time()
        data = current_hmap.sweepModelDisplacementAndErosion(
            current_hmap, 
            xy_sweep_start=sweep_vec[:2], 
            xy_sweep_end=sweep_vec[2:], 
            sweep_width=p['sweep_width'], 
            angle_of_repose_deg=p['aor'],
            dump_base_angle_deg=p['dump_base_angle_deg'], 
            relative_height_tolerance=0.95, 
            N_iter_max=p['erosion_max_iters'],
            erosion_stencil=self.erosion_stencil,
        )
        time_for_prediction = time.time() - time_before
        return data['elev_map_predicted'], time_for_prediction

    def plan(self, current_hmap, target_mask, cell_area) -> dict:
        candidates_info = []

        print(f"Starting ClearancePlanner planning with strategy: {self.selection_strategy}...")        
        
        # --- 1. Target Processing ---
        # No longer needed! target_mask is already a perfect binary mask passed from run_planner.
        
        # --- 2. Generate & Filter ---
        candidates, colors = self._generate_candidates(current_hmap)
        candidates, colors = self._filter_sweeps(candidates, colors)
        
        if len(candidates) == 0:
            print("[WARNING] No valid sweeps generated.")
            return {"PRE": [0,0], "POST": [0,0], "candidates_info": []}

        if self.selection_strategy == 'random_blind':
            random_idx = np.random.choice(len(candidates))
            best_sweep = candidates[random_idx]
            print(f"Randomly selected sweep. Bypassing exhaustive predictions.")
            return {"PRE": best_sweep[:2], "POST": best_sweep[2:], "candidates_info": []}

        elif self.selection_strategy == 'exhaustive':
            # --- 3. Exhaustive Search ---
            best_sweep = None
            min_vol_remaining = float('inf')

            time_for_predictions = []
            
            for idx, sweep in enumerate(candidates):
                sweep_color = colors[idx]
                
                # Predict
                pred_map, time_for_prediction = self._predict(current_hmap, sweep)
                time_for_predictions.append(time_for_prediction)

                # SCORE: Calculate absolute volume REMAINING in the danger zone using unified function
                vol_in_zone = compute_clearance_volume(pred_map.elevation_data, target_mask, cell_area)
                
                if vol_in_zone < min_vol_remaining:
                    min_vol_remaining = vol_in_zone
                    best_sweep = sweep

                # Store candidate info
                candidates_info.append({
                    'sweep': sweep,
                    'color': sweep_color,  
                    'predicted_map': pred_map,
                    'vol_in_zone': vol_in_zone,
                    'time_for_prediction': time_for_prediction
                })

            print(f"Best Sweep Found. Remaining Vol: {min_vol_remaining:.5f} m^3")
            return {"PRE": best_sweep[:2], "POST": best_sweep[2:], "candidates_info": candidates_info}
        else:
            raise ValueError(f"Unknown selection strategy: {self.selection_strategy}")