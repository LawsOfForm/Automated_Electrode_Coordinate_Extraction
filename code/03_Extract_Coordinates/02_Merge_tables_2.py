import pandas as pd
import numpy as np
from pathlib import Path
import os
import pickle
import glob
import re
import copy
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import sys
from datetime import datetime

# define most import path variables
script_directory = Path(__file__).parent.resolve()
root = script_directory.parent.parent.resolve()
tables = os.path.join(script_directory ,'Tables')

# Create logfile path and setup logging
logfile_path = os.path.join(script_directory, 'coordinate_correction.log')

# Redirect print output to logfile
class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w')
        
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        
    def flush(self):
        self.terminal.flush()
        self.log.flush()
        
    def close(self):
        self.log.close()

# Initialize logger
logger = Logger(logfile_path)
sys.stdout = logger

# Write header with timestamp
print(f"Coordinate Correction Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)

# Load the data
df = pd.read_csv(os.path.join(tables,'electrode_positions_72tr_mni.csv'))

def euclidean_distance(coord1, coord2):
    return np.sqrt(np.sum((np.array(coord1) - np.array(coord2))**2))

def verify_anode_configuration(electrode_coords, tolerance_factor=1.5):
    """
    Verify that the anode is surrounded by 3 cathodes in a circular fashion.
    The distances from anode to all cathodes should be roughly equal,
    while distances between cathodes should be larger.
    
    Returns:
    - is_valid: Boolean indicating if configuration is valid
    - anode_idx: Index of the electrode that appears to be the anode
    - confidence: Confidence score (0-1) of the anode identification
    """
    electrodes = list(electrode_coords.keys())
    
    # Filter out electrodes with None coordinates
    valid_electrodes = []
    valid_coords = []
    
    for electrode in electrodes:
        # Handle both nested and flat dictionary structures
        coords_data = electrode_coords[electrode]
        if isinstance(coords_data, dict) and electrode in coords_data:
            # Nested structure: {'anode': {'anode': [x, y, z]}}
            coords = coords_data[electrode]
        else:
            # Flat structure: {'anode': [x, y, z]}
            coords = coords_data
            
        # Check if all coordinates are valid (not None and numeric)
        if (isinstance(coords, (list, np.ndarray)) and len(coords) >= 3 and
            coords[0] is not None and coords[1] is not None and coords[2] is not None and
            not np.isnan(coords[0]) and not np.isnan(coords[1]) and not np.isnan(coords[2])):
            valid_electrodes.append(electrode)
            valid_coords.append(coords)
    
    if len(valid_electrodes) < 4:
        print(f"Warning: Only {len(valid_electrodes)} valid electrodes found, need 4 for verification")
        return False, None, 0.0
    
    coords_array = np.array(valid_coords)
    
    try:
        # Calculate distance matrix
        dist_matrix = cdist(coords_array, coords_array)
    except Exception as e:
        print(f"Error calculating distance matrix: {e}")
        return False, None, 0.0
    
    # For each electrode as potential anode, calculate configuration score
    best_anode_idx = None
    best_score = -1
    
    for i in range(len(valid_electrodes)):
        # Distances from potential anode to others
        anode_dists = [dist_matrix[i, j] for j in range(len(valid_electrodes)) if j != i]
        
        if len(anode_dists) < 3:
            continue
            
        # Sort distances and take the 3 smallest (potential cathodes)
        cathode_dists = sorted(anode_dists)[:3]
        
        # Calculate metrics
        mean_cathode_dist = np.mean(cathode_dists)
        std_cathode_dist = np.std(cathode_dists)
        
        # Distance uniformity score (higher is better)
        if mean_cathode_dist > 0:
            uniformity_score = 1 - (std_cathode_dist / mean_cathode_dist)
            uniformity_score = max(0, min(1, uniformity_score))  # Clamp between 0 and 1
        else:
            uniformity_score = 0
            
        # Calculate distances between the potential cathodes
        other_indices = [j for j in range(len(valid_electrodes)) if j != i]
        cathode_indices = [other_indices[k] for k in np.argsort(anode_dists)[:3]]
        
        cathode_pair_dists = []
        for idx1 in range(len(cathode_indices)):
            for idx2 in range(idx1 + 1, len(cathode_indices)):
                cathode_pair_dists.append(dist_matrix[cathode_indices[idx1], cathode_indices[idx2]])
        
        if len(cathode_pair_dists) > 0 and mean_cathode_dist > 0:
            mean_cathode_pair_dist = np.mean(cathode_pair_dists)
            # Ratio should be > 1 (cathode-cathode distances > anode-cathode distances)
            distance_ratio = mean_cathode_pair_dist / mean_cathode_dist
            distance_ratio = min(distance_ratio, 3)  # Cap at 3 to avoid extreme values
        else:
            distance_ratio = 0
            
        # Combined score (emphasize uniformity more)
        configuration_score = (uniformity_score * 0.7) + (min(distance_ratio / 3, 1) * 0.3)
        
        if configuration_score > best_score:
            best_score = configuration_score
            best_anode_idx = i
    
    is_valid = best_score > 0.3  # Threshold for validity
    
    # Map back to original electrode names
    if best_anode_idx is not None:
        detected_anode_name = valid_electrodes[best_anode_idx]
        # Find index in original electrodes list
        original_anode_idx = electrodes.index(detected_anode_name) if detected_anode_name in electrodes else None
    else:
        original_anode_idx = None
    
    return is_valid, original_anode_idx, best_score

def construct_baseline_coordinate_table(tables):
    root_table = '/media/MeMoSLAP_Mesh2/PDF_Report_Generation'
    pickle_filelist = glob.glob(os.path.join(root_table, 'sham','02-ANALYSIS', '**', '*.pkl'), recursive=True)
    dict_data = {}
    for file in pickle_filelist:
        folder_str = os.path.basename(os.path.dirname(file))

        Exp = folder_str.split('_')[0]
        targe_condition = folder_str.split('_')[1]
        sub = ['sub-' + folder_str.split('_')[2]][0]

        dict_templ_Exp = {
            'P1': '1',
            'P2': '2',
            'P3': '3',
            'P4': '4',
            'P5': '5',
            'P6': '6',
            'P7': '7',
            'P8': '8'
        }

        with open(file, 'rb') as f:data = pickle.load(f)
        #print(data)
        key = list(data[2].keys())[0]
        
        dict_data[f'{Exp}_{targe_condition}_{sub}']={'anode':list(data[1]), 'cathode1':list(data[2][key][0]), 'cathode2':list(data[2][key][1]), 'cathode3':list(data[2][key][2])} 
    
    df_template = pd.DataFrame.from_dict(dict_data, orient='index')
    df_template=df_template.reset_index()
    df_template[['exp', 'condition', 'subject']] = df_template['index'].str.split('_', expand=True)
    df_template['session'] = 'ses-baseline'
    df_template['run'] = 'run-baseline'
    df_template.to_csv(os.path.join(tables,'baseline_coordinate_table.csv'), index=False)
    
    return df_template

def parse_coordinates(coord_str):
    # Find all numbers (including negative) in the string
    numbers = re.findall(r'-?\d+\.\d+', coord_str)
    return [float(num) for num in numbers]

def wide_to_long(df, electrodes):
    """Convert wide format dataframe to long format"""
    df = df.melt(
        id_vars = [col for col in df.columns if col not in electrodes],
        value_vars = electrodes,
        var_name = 'electrode',
        value_name = 'coordinates'
    )

    try:
        df['coordinates'] = df['coordinates'].apply(parse_coordinates)
    except:
        print('Error parsing coordinates but if the code is running it should not be a problem')

    df_coordinates = df.copy()

    dimensions = ['X','Y','Z']
    
    df[dimensions]=df['coordinates'].apply(pd.Series)
    
    df=df.drop(columns=['coordinates'], axis=1)

    df = df.melt(
        id_vars = [col for col in df.columns if col not in dimensions],
        value_vars = dimensions,
        var_name = 'dimension',
        value_name = 'coordinates'
    )

    return df, df_coordinates

def is_dataframe_in_long_format(df, required_columns=['subject', 'session', 'run', 'electrode', 'dimension', 'coordinates']):
    """Check if dataframe is already in long format"""
    return all(col in df.columns for col in required_columns)

def prepare_dataframe_for_processing(df, electrodes=['anode', 'cathode1', 'cathode2', 'cathode3']):
    """
    Prepare dataframe for coordinate correction processing.
    Handles both wide and long format dataframes.
    """
    if is_dataframe_in_long_format(df):
        print("DataFrame is already in long format, using as-is")
        df_coordinates = df.copy()
        
        # If coordinates are strings, parse them
        if df_coordinates['coordinates'].dtype == 'object':
            try:
                df_coordinates['coordinates'] = df_coordinates['coordinates'].apply(parse_coordinates)
            except:
                print('Note: Could not parse coordinates, assuming they are already in numeric format')
        
        return df, df_coordinates
    else:
        print("DataFrame is in wide format, converting to long format")
        return wide_to_long(df, electrodes)

def create_coords_df(df_2, electrode):
    """
    Extract coordinates from DataFrame, handling both formats:
    - Format 1: Coordinates split by dimension (X, Y, Z in separate rows)
    - Format 2: Coordinates as list in single row
    """
    df_copy = copy.deepcopy(df_2)
    
    # Check if we have dimension format (Format 1)
    if 'dimension' in df_copy.columns and not df_copy[df_copy['dimension'].isin(['X', 'Y', 'Z'])].empty:
        # Format 1: Coordinates split by dimension
        try:
            x_coord = df_copy[df_copy['dimension']=='X']['coordinates'].values
            y_coord = df_copy[df_copy['dimension']=='Y']['coordinates'].values
            z_coord = df_copy[df_copy['dimension']=='Z']['coordinates'].values
            
            coords = [
                x_coord[0] if len(x_coord) > 0 else None,
                y_coord[0] if len(y_coord) > 0 else None,
                z_coord[0] if len(z_coord) > 0 else None
            ]
        except:
            coords = [None, None, None]
    else:
        # Format 2: Coordinates as list in single row
        try:
            coords_list = df_copy['coordinates'].values
            if len(coords_list) > 0:
                coord_array = coords_list[0]
                if isinstance(coord_array, (list, np.ndarray)) and len(coord_array) >= 3:
                    coords = [coord_array[0], coord_array[1], coord_array[2]]
                else:
                    coords = [None, None, None]
            else:
                coords = [None, None, None]
        except:
            coords = [None, None, None]
    
    return {electrode: coords}

def extract_coordinates(coord_dict, electrode):
    """Extract coordinates from either nested or flat dictionary structure"""
    if electrode in coord_dict:
        coords_data = coord_dict[electrode]
        if isinstance(coords_data, dict) and electrode in coords_data:
            # Nested structure: {'anode': {'anode': [x, y, z]}}
            return coords_data[electrode]
        else:
            # Flat structure: {'anode': [x, y, z]}
            return coords_data
    return [None, None, None]

def check_and_correct_coordinates(df_input, tables, df_ground_truth_wide, electrodes=['anode', 'cathode1', 'cathode2', 'cathode3']):
    """
    Main function to check and correct coordinates.
    Handles both wide and long format input dataframes.
    """
    
    # Prepare input data (convert to long format if needed)
    df, df_coordinates = prepare_dataframe_for_processing(df_input, electrodes)
    
    # Prepare ground truth data (always convert from wide to long)
    ground_truth_df, ground_truth_df_coordinates = wide_to_long(df_ground_truth_wide, electrodes)

    # Process subject/session/run combinations
    df = _process_coordinate_correction(df, df_coordinates, ground_truth_df, ground_truth_df_coordinates, electrodes, tables)
    
    return df, ground_truth_df

def _process_coordinate_correction(df, df_coordinates, ground_truth_df, ground_truth_df_coordinates, electrodes, tables):
    """Internal function to process coordinate correction"""
    
    # join both df to get combined coordinates
    ground_truth_df_coordinates = ground_truth_df_coordinates.drop(columns=['index','exp','condition'], axis=1)
    df_all = pd.concat([df_coordinates, ground_truth_df_coordinates], axis=0, ignore_index=True)
    df['condition'] = 'unknown'
    
    # create euclidian norm for all electrodes
    df_all['euclidian_norm'] = df_all['coordinates'].apply(lambda x: np.linalg.norm(x) if isinstance(x, (list, np.ndarray)) and len(x) >= 3 else np.nan)
    df_all.to_csv(os.path.join(tables,'all_coordinates_list_table.csv'), index=False)
    
    # Get unique combinations
    subjects = df_all['subject'].unique()
    sessions = df_all['session'].unique() 
    runs = df_all['run'].unique()

    # Loop over each subject, session, and run
    for subject in subjects:
        for session in sessions:
            for run in runs:
                _process_single_subject_session_run(subject, session, run, df_all, ground_truth_df, df, electrodes)

    return df

def _process_single_subject_session_run(subject, session, run, df_all, ground_truth_df, df, electrodes):
    """Process a single subject-session-run combination"""
    
    # create df for each subject
    sub_session_run_df = df_all[(df_all['subject'] == subject) & (df_all['session'] == session) & (df_all['run'] == run)].copy(deep=True)
    sub_ground_truth_target_df = ground_truth_df[(ground_truth_df['subject'] == subject) & (ground_truth_df['condition']=='target')].copy(deep=True)
    sub_ground_truth_control_df = ground_truth_df[(ground_truth_df['subject'] == subject) & (ground_truth_df['condition']=='control')].copy(deep=True)
    
    # Check if we have data for all required DataFrames
    if (len(sub_session_run_df) > 0 and len(sub_ground_truth_target_df) > 0 and len(sub_ground_truth_control_df) > 0):
        
        full_auto_coords = {}
        ground_truth_coords_target = {}
        ground_truth_coords_control = {}

        for electrode in electrodes:
            # create df for each electrode
            sub_session_run_electrode_df = sub_session_run_df[(sub_session_run_df['electrode'] == electrode)].copy(deep=True)
            sub_electrode_ground_truth_target_df = sub_ground_truth_target_df[(sub_ground_truth_target_df['electrode'] == electrode)].copy(deep=True)
            sub_electrode_ground_truth_control_df = sub_ground_truth_control_df[(sub_ground_truth_control_df['electrode'] == electrode)].copy(deep=True)

            # Only process if we have data
            if len(sub_session_run_electrode_df) > 0:
                full_auto_coords[electrode] = create_coords_df(sub_session_run_electrode_df, electrode)
            
            if len(sub_electrode_ground_truth_target_df) > 0:
                ground_truth_coords_target[electrode] = create_coords_df(sub_electrode_ground_truth_target_df, electrode)
            
            if len(sub_electrode_ground_truth_control_df) > 0:
                ground_truth_coords_control[electrode] = create_coords_df(sub_electrode_ground_truth_control_df, electrode)

        # Check if we have all required coordinates
        if (len(full_auto_coords) == len(electrodes) and 
            len(ground_truth_coords_target) == len(electrodes) and 
            len(ground_truth_coords_control) == len(electrodes)):
            
            # Perform the actual coordinate correction
            _perform_coordinate_correction(subject, session, run, full_auto_coords, ground_truth_coords_target, 
                                         ground_truth_coords_control, electrodes, df)
        else:
            print(f"Warning: Missing coordinate data for subject {subject}, session {session}, run {run}")
    else:
        print(f"Warning: No data for subject {subject}, session {session}, run {run}")

def _perform_coordinate_correction(subject, session, run, full_auto_coords, ground_truth_coords_target, 
                                 ground_truth_coords_control, electrodes, df):
    """Perform the actual coordinate correction logic"""
    
    # VERIFY ANODE CONFIGURATION BEFORE CORRECTION
    print(f"\n--- Anode Configuration Verification for {subject}, {session}, {run} ---")
    is_valid, detected_anode_idx, confidence = verify_anode_configuration(full_auto_coords)
    
    electrodes_list = list(full_auto_coords.keys())
    if detected_anode_idx is not None and detected_anode_idx < len(electrodes_list):
        detected_anode = electrodes_list[detected_anode_idx]
        print(f"Detected anode: {detected_anode} (confidence: {confidence:.3f})")
        
        if detected_anode != 'anode':
            print(f"WARNING: Anode mismatch! Expected 'anode', but configuration suggests '{detected_anode}'")
        else:
            print("Anode configuration appears correct")
    else:
        print("WARNING: Could not confidently identify anode from configuration")
    
    # Calculate distance matrices for both conditions
    distance_matrix_target = np.zeros((len(electrodes), len(electrodes)))
    distance_matrix_control = np.zeros((len(electrodes), len(electrodes)))
    
    for i, gt_electrode in enumerate(electrodes):
        for j, auto_electrode in enumerate(electrodes):
            # Extract coordinates using the helper function
            gt_coords = extract_coordinates(ground_truth_coords_target, gt_electrode)
            auto_coords = extract_coordinates(full_auto_coords, auto_electrode)
            
            if (None not in gt_coords and None not in auto_coords and
                not any(np.isnan(coord) for coord in gt_coords) and
                not any(np.isnan(coord) for coord in auto_coords)):
                distance_matrix_target[i, j] = euclidean_distance(gt_coords, auto_coords)
            else:
                distance_matrix_target[i, j] = float('inf')
            
            gt_coords_control = extract_coordinates(ground_truth_coords_control, gt_electrode)
            if (None not in gt_coords_control and None not in auto_coords and
                not any(np.isnan(coord) for coord in gt_coords_control) and
                not any(np.isnan(coord) for coord in auto_coords)):
                distance_matrix_control[i, j] = euclidean_distance(gt_coords_control, auto_coords)
            else:
                distance_matrix_control[i, j] = float('inf')
    
    # Calculate total distance for each condition
    total_distance_target = np.sum(distance_matrix_target[distance_matrix_target != float('inf')])
    total_distance_control = np.sum(distance_matrix_control[distance_matrix_control != float('inf')])
    
    # Choose the condition with smaller total distance
    if total_distance_target <= total_distance_control:
        distance_matrix = distance_matrix_target
        condition = 'target'
        # CORRECTED: Use .loc to avoid chained indexing
        mask = (df['subject'] == subject) & (df['session'] == session) & (df['run'] == run)
        df.loc[mask, 'condition'] = condition
        ground_truth_coords = ground_truth_coords_target
    else:
        distance_matrix = distance_matrix_control
        condition = 'control'
        ground_truth_coords = ground_truth_coords_control
        # CORRECTED: Use .loc to avoid chained indexing
        mask = (df['subject'] == subject) & (df['session'] == session) & (df['run'] == run)
        df.loc[mask, 'condition'] = condition
    
    print(f"Using {condition} condition for subject {subject}, session {session}, run {run}")
    print(f"Total distance - target: {total_distance_target:.2f}, control: {total_distance_control:.2f}")
    
    # Use Hungarian algorithm for optimal assignment (also called Munkres algorithm) 
    # Replace inf with a large number for the assignment
    distance_matrix_fixed = np.where(distance_matrix == float('inf'), 1e9, distance_matrix)
    row_ind, col_ind = linear_sum_assignment(distance_matrix_fixed)
    
    correct_mapping = {}
    total_optimized_distance = 0
    
    for i, j in zip(row_ind, col_ind):
        gt_electrode = electrodes[i]
        auto_electrode = electrodes[j]
        distance = distance_matrix[i, j] if distance_matrix[i, j] != float('inf') else 0
        total_optimized_distance += distance
        
        if gt_electrode != auto_electrode:
            print(f"Optimal assignment: {gt_electrode} -> {auto_electrode} (distance: {distance:.2f})")
            correct_mapping[gt_electrode] = auto_electrode
    
    print(f"Total optimized distance: {total_optimized_distance:.2f}")
    
    # Calculate what the total distance would be with identity mapping (no correction)
    identity_distance = 0
    for i in range(len(electrodes)):
        if distance_matrix[i, i] != float('inf'):
            identity_distance += distance_matrix[i, i]
    print(f"Identity mapping distance: {identity_distance:.2f}")
    
    # Only apply corrections if the optimized mapping is better
    if total_optimized_distance < identity_distance:
        print(f"Applying corrections - improvement: {identity_distance - total_optimized_distance:.2f}")
        
        # Create corrected coordinates for verification - use flat structure
        corrected_coords = {}
        for electrode in electrodes:
            if electrode in correct_mapping:
                # Extract coordinates from the mapped electrode
                corrected_coords[electrode] = extract_coordinates(full_auto_coords, correct_mapping[electrode])
            else:
                corrected_coords[electrode] = extract_coordinates(full_auto_coords, electrode)
        
        # VERIFY ANODE CONFIGURATION AFTER CORRECTION
        print("--- Verifying anode configuration after correction ---")
        is_valid_corrected, detected_anode_idx_corrected, confidence_corrected = verify_anode_configuration(corrected_coords)
        
        if detected_anode_idx_corrected is not None and detected_anode_idx_corrected < len(electrodes_list):
            detected_anode_corrected = electrodes_list[detected_anode_idx_corrected]
            print(f"Detected anode after correction: {detected_anode_corrected} (confidence: {confidence_corrected:.3f})")
            
            if detected_anode_corrected != 'anode':
                print(f"WARNING: Anode mismatch after correction! Expected 'anode', but configuration suggests '{detected_anode_corrected}'")
            else:
                print("Anode configuration correct after correction")
        else:
            print("WARNING: Could not confidently identify anode after correction")
        
        # Apply the corrections to the DataFrame
        for gt_electrode, auto_electrode in correct_mapping.items():
            correct_coords = extract_coordinates(full_auto_coords, auto_electrode)
            for num, dimension in enumerate(['X', 'Y', 'Z']):
                mask = (df['subject'] == subject) & (df['session'] == session) & (df['run'] == run) & (df['electrode'] == gt_electrode) & (df['dimension'] == dimension) & (df['condition'] == condition)
                if correct_coords[num] is not None and not np.isnan(correct_coords[num]):
                    df.loc[mask, 'coordinates'] = correct_coords[num]
    else:
        print("No correction needed - identity mapping is optimal")

def save_results(corrected_df, baseline_df, tables, suffix=""):
    """Save the corrected results with optional suffix"""
    
    print("Saving corrected data...")
    corrected_df = corrected_df.drop_duplicates()
    corrected_df.to_csv(os.path.join(tables, f'corrected_electrode_positions_no_baseline_long_RU{suffix}.csv'), index=False)

    if 'rater' in corrected_df.columns:
        column_names = ['subject', 'session', 'run', 'electrode', 'condition', 'rater','method']
        corrected_df_long = corrected_df.melt(id_vars=[column_names], var_name='dimension', value_name='coordinates')
    else:
        column_names = ['subject', 'session', 'run', 'electrode', 'condition','coordinates']
        corrected_df_long = corrected_df.melt(id_vars=[column_names], var_name='dimension', value_name='coordinates')
    
    corrected_df_long.to_csv(os.path.join(tables, f'corrected_electrode_positions_no_baseline_long_RU{suffix}.csv'), index=False)
    corrected_df_wide = corrected_df.pivot(index=[column_names], columns='dimension', values='coordinates')
    corrected_df_wide.reset_index().to_csv(os.path.join(tables, f'corrected_electrode_positions_no_baseline_wide_RU{suffix}.csv'), index=False)

    baseline_df = baseline_df.drop(columns=['index','exp'], axis=1)
    df_concat = pd.concat([corrected_df, baseline_df], axis=0, ignore_index=True)

    # Remove duplicates
    df_concat = _remove_duplicates(df_concat)
    
    # Create wide format and save
    df_concat_wide = df_concat.pivot(
        index=['subject', 'session', 'run', 'electrode', 'condition','rater','method'], 
        columns='dimension', 
        values='coordinates'
    )
    
    df_concat_wide['euclidian_norm'] = np.linalg.norm(df_concat_wide[['X', 'Y', 'Z']], axis=1)
    df_concat_wide.reset_index().to_csv(os.path.join(tables, f'corrected_electrode_positions_with_baseline_wide_RU{suffix}.csv'), index=False)

def _remove_duplicates(df_concat):
    """Remove duplicates from the concatenated dataframe"""
    
    # 1. First, check for exact duplicates
    print("Exact duplicates (all columns same):")
    exact_dups = df_concat[df_concat.duplicated(keep=False)]
    print(f"Exact duplicates: {len(exact_dups)}")

    # 2. Remove exact duplicates
    df_concat = df_concat.drop_duplicates()

    # 3. Check for pivot index duplicates (the real problem)
    pivot_columns = ['subject', 'session', 'run', 'electrode', 'condition', 'dimension']
    print(f"\nChecking for duplicates in pivot index columns: {pivot_columns}")

    # Find rows that have duplicate index combinations
    index_duplicates = df_concat[df_concat.duplicated(subset=pivot_columns, keep=False)]
    print(f"Pivot index duplicates: {len(index_duplicates)} rows")

    if len(index_duplicates) > 0:
        print("\nDuplicate index combinations found:")
        print(index_duplicates.sort_values(pivot_columns))
        
        # 4. Decide how to handle the duplicates
        # Option A: Keep the first occurrence
        df_concat = df_concat.drop_duplicates(subset=pivot_columns, keep='first')
        
        print(f"After removing pivot duplicates: {len(df_concat)} rows")

    return df_concat

def process_experiment_data(df_experiment, df_ground_truth_wide, tables, experiment_name=""):
    """
    Process experiment data with coordinate correction.
    
    Parameters:
    - df_experiment: DataFrame with experiment data (can be wide or long format)
    - df_ground_truth_wide: DataFrame with ground truth data (wide format)
    - tables: Path to tables directory
    - experiment_name: Optional name for the experiment (used in output files)
    """
    
    print(f"\n{'='*60}")
    print(f"Processing experiment data: {experiment_name}")
    print(f"{'='*60}")
    
    # Apply the correction
    corrected_df, baseline_df = check_and_correct_coordinates(df_experiment, tables, df_ground_truth_wide)
    
    # Save results with experiment suffix
    suffix = f"_{experiment_name}" if experiment_name else "_experiment"
    save_results(corrected_df, baseline_df, tables, suffix)
    
    return corrected_df

# Main execution block
if __name__ == "__main__":
    try:
        # Apply the correction for main experiment
        print("Starting coordinate correction process...")
        df_ground_truth_wide = construct_baseline_coordinate_table(tables)
        corrected_df, baseline_df = check_and_correct_coordinates(df, tables, df_ground_truth_wide)

        # Save the corrected data for main experiment
        save_results(corrected_df, baseline_df, tables)
        
        print("Coordinate correction completed successfully!")
        
    except Exception as e:
        print(f"Error during coordinate correction: {str(e)}")
        import traceback
        traceback.print_exc()

    finally:
        # Close the logger and restore stdout
        print(f"Process automated electrode coordinate extraction and correction completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.close()
        sys.stdout = logger.terminal

    ## start process for other experiments - NOW MUCH EASIER TO INTEGRATE

    # If you have a dataframe with baseline electrode coordinates you can merge it with the electrode_positions.csv file
    df_other_experiment = pd.read_csv(os.path.join(tables,'DF_2Methods_2Raters_All_Coord.csv'))

    df_other_experiment = df_other_experiment.rename(columns={
        'Subject': 'subject',
        'Session': 'session', 
        'run': 'run',
        'Experiment': 'experiment',
        'Electrode': 'electrode',
        'Rater':'rater',
        'Method':'method',
        'Dimension': 'dimension',
        'Coordinates': 'coordinates',
        'date': 'date'
    })

    df_other_experiment['run'] = df_other_experiment['run'].str.replace('baseline','run-baseline')
    df_other_experiment['session'] = df_other_experiment['session'].str.replace('ses-0','ses-baseline') 
    df_other_experiment['electrode'] = df_other_experiment['electrode'].str.replace('A', 'a').str.replace('C', 'c')

    # Now simply call the processing function
    corrected_other_experiment = process_experiment_data(
        df_other_experiment, 
        df_ground_truth_wide, 
        tables, 
        experiment_name="other_experiment"
    )