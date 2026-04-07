import pandas as pd
import numpy as np
from pathlib import Path
import os
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

# define most import path variables
script_directory = Path(__file__).parent.resolve()
root = script_directory.parent.parent.resolve()
tables = os.path.join(root,'code','Extract_Coordinates','Tables')
# Load the data
df = pd.read_csv(os.path.join(tables,'combined_electrode_positions.csv'))

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
        coords = electrode_coords[electrode]
            
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

def check_and_correct_coordinates(df):
    df=df[df['Rater']!='Sophie']
    electrodes = ['Anode', 'Cathode1', 'Cathode2', 'Cathode3']
    sessions = df['Session'].unique()
    runs = df['run'].unique()
    subjects = df['Subject'].unique()

    for subject in subjects:
        for session in sessions:
            for run in runs:
                sub_session_run_df = df[(df['Subject'] == subject) & (df['Session'] == session) & (df['run'] == run)]
                
                half_auto_coords = {}
                full_auto_coords = {}

                if not any(sub_session_run_df['Method']=='full-automated'):
                    print('no full automated data for ',subject, session, run)
                    continue
                
                if any(sub_session_run_df['Method']=='full-automated') & any(sub_session_run_df['Method']=='half-automated'):

                    for electrode in electrodes:
                        half_auto = sub_session_run_df[(sub_session_run_df['Electrode'] == electrode) & 
                                                (sub_session_run_df['Method'] == 'half-automated')]
                        full_automated = sub_session_run_df[(sub_session_run_df['Electrode'] == electrode) & 
                                                (sub_session_run_df['Method'] == 'full-automated')]

                        half_auto_coords[electrode] = [half_auto[half_auto['Dimension']=='X']['Coordinates'].values[0], 
                                                    half_auto[half_auto['Dimension']=='Y']['Coordinates'].values[0], 
                                                    half_auto[half_auto['Dimension']=='Z']['Coordinates'].values[0]]
                        full_auto_coords[electrode] = [full_automated[full_automated['Dimension']=='X']['Coordinates'].values[0], 
                                                    full_automated[full_automated['Dimension']=='Y']['Coordinates'].values[0], 
                                                    full_automated[full_automated['Dimension']=='Z']['Coordinates'].values[0]]
                    
                    # VERIFY ANODE CONFIGURATION BEFORE CORRECTION
                    print(f"\n--- Anode Configuration Verification for {subject}, {session}, {run} ---")
                    is_valid, detected_anode_idx, confidence = verify_anode_configuration(full_auto_coords)
                    
                    electrodes_list = list(full_auto_coords.keys())
                    if detected_anode_idx is not None and detected_anode_idx < len(electrodes_list):
                        detected_anode = electrodes_list[detected_anode_idx]
                        print(f"Detected anode: {detected_anode} (confidence: {confidence:.3f})")
                        
                        if detected_anode != 'Anode':
                            print(f"WARNING: Anode mismatch! Expected 'Anode', but configuration suggests '{detected_anode}'")
                        else:
                            print("Anode configuration appears correct")
                    else:
                        print("WARNING: Could not confidently identify anode from configuration")
                    
                    # Calculate distance matrix using Hungarian algorithm instead of greedy approach
                    distance_matrix = np.zeros((len(electrodes), len(electrodes)))
                    
                    for i, half_electrode in enumerate(electrodes):
                        for j, full_electrode in enumerate(electrodes):
                            half_coords = half_auto_coords[half_electrode]
                            full_coords = full_auto_coords[full_electrode]
                            
                            if (None not in half_coords and None not in full_coords and
                                not any(np.isnan(coord) for coord in half_coords) and
                                not any(np.isnan(coord) for coord in full_coords)):
                                distance_matrix[i, j] = euclidean_distance(half_coords, full_coords)
                            else:
                                distance_matrix[i, j] = float('inf')
                    
                    # Use Hungarian algorithm for optimal assignment
                    distance_matrix_fixed = np.where(distance_matrix == float('inf'), 1e9, distance_matrix)
                    row_ind, col_ind = linear_sum_assignment(distance_matrix_fixed)
                    
                    correct_mapping = {}
                    total_optimized_distance = 0
                    
                    for i, j in zip(row_ind, col_ind):
                        half_electrode = electrodes[i]
                        full_electrode = electrodes[j]
                        distance = distance_matrix[i, j] if distance_matrix[i, j] != float('inf') else 0
                        total_optimized_distance += distance
                        
                        if half_electrode != full_electrode:
                            print(f"Optimal assignment: {half_electrode} -> {full_electrode} (distance: {distance:.2f})")
                            correct_mapping[half_electrode] = full_electrode
                    
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
                        
                        # Create corrected coordinates for verification
                        corrected_coords = {}
                        for electrode in electrodes:
                            if electrode in correct_mapping:
                                corrected_coords[electrode] = full_auto_coords[correct_mapping[electrode]]
                            else:
                                corrected_coords[electrode] = full_auto_coords[electrode]
                        
                        # VERIFY ANODE CONFIGURATION AFTER CORRECTION
                        print("--- Verifying anode configuration after correction ---")
                        is_valid_corrected, detected_anode_idx_corrected, confidence_corrected = verify_anode_configuration(corrected_coords)
                        
                        if detected_anode_idx_corrected is not None and detected_anode_idx_corrected < len(electrodes_list):
                            detected_anode_corrected = electrodes_list[detected_anode_idx_corrected]
                            print(f"Detected anode after correction: {detected_anode_corrected} (confidence: {confidence_corrected:.3f})")
                            
                            if detected_anode_corrected != 'Anode':
                                print(f"WARNING: Anode mismatch after correction! Expected 'Anode', but configuration suggests '{detected_anode_corrected}'")
                            else:
                                print("Anode configuration correct after correction")
                        else:
                            print("WARNING: Could not confidently identify anode after correction")
                        
                        # Apply the corrections to the DataFrame
                        for half_electrode, full_electrode in correct_mapping.items():
                            correct_coords = full_auto_coords[full_electrode]
                            for num, dimension in enumerate(['X', 'Y', 'Z']):
                                mask = (df['Subject'] == subject) & (df['Session'] == session) & (df['run'] == run) & (df['Electrode'] == half_electrode) & (df['Method'] == 'full-automated') & (df['Dimension'] == dimension)
                                df.loc[mask, 'Coordinates'] = correct_coords[num]
                    else:
                        print("No correction needed - identity mapping is optimal")

    return df

# Apply the correction
corrected_df = check_and_correct_coordinates(df)

# Save the corrected data
corrected_df.to_csv(os.path.join(tables,'corrected_electrode_positions.csv'), index=False)