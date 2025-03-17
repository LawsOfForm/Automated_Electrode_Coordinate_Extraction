import pandas as pd
import numpy as np
from pathlib import Path
import os

# define most import path variables
script_directory = Path(__file__).parent.resolve()
root = script_directory.parent.parent.resolve()
tables = os.path.join(root,'code','Extract_Coordinates','Tables')
# Load the data
df = pd.read_csv(os.path.join(tables,'combined_electrode_positions.csv'))

def euclidean_distance(coord1, coord2):
    return np.sqrt(np.sum((np.array(coord1) - np.array(coord2))**2))

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
                    dict_electrode_distances = {}
                    for num, electrode in enumerate(electrodes):
                        dict_electrode_distances[electrode]= {electrode: euclidean_distance(half_auto_coords[electrodes[num]], full_auto_coords[electrode]) 
                                    for electrode in electrodes}

                    #generate template in which anode is paired with anode etc
                    correct_mapping = {}
                    used_electrodes = set()  # Track electrodes that have already been mapped

                    for electrode_distance in dict_electrode_distances.keys():
                        # Filter out electrodes that have already been used
                        available_electrodes = {k: v for k, v in dict_electrode_distances[electrode_distance].items() if k not in used_electrodes}
                        if not available_electrodes:
                            continue  # Skip if no available electrodes left
                        closest_match = min(available_electrodes, key=available_electrodes.get)
                        if closest_match != electrode_distance:
                            print(f"Potential mix-up detected in {session}, {run}:")
                            print(f"{electrode_distance} coordinates seem to match {closest_match}")
                            correct_mapping[electrode_distance] = closest_match
                            used_electrodes.add(closest_match)  # Mark this electrode as used
                        dict_electrode_distances[electrode_distance].pop(closest_match)

                    # Correct the coordinates if mix-ups are detected
                    for electrode, correct_electrode in correct_mapping.items():
                        if electrode != correct_electrode:
                            for num,dimension in enumerate(['X', 'Y', 'Z']):
                                mask = (df['Subject'] == subject) & (df['Session'] == session) & (df['run'] == run) & (df['Electrode'] == electrode) & (df['Method'] == 'full-automated') & (df['Dimension'] == dimension)
                                df.loc[mask, 'Coordinates'] = full_auto_coords[correct_electrode][num]

    return df

# Apply the correction
corrected_df = check_and_correct_coordinates(df)

# Save the corrected data
corrected_df.to_csv(os.path.join(tables,'corrected_electrode_positions.csv'), index=False)