import pandas as pd
import ast
import numpy as np
from pathlib import Path
import os
import ast

# define most import path variables
script_directory = Path(__file__).parent.resolve()
root = script_directory.parent.parent.resolve()

path_Tables = os.path.join(root, 'code','Extract_Coordinates','Tables')

# Read the CSV files
df = pd.read_csv(os.path.join(path_Tables,'electrode_positions.csv'))

#If you have a dataframe with baseline electrode coordinates you can merge it with the electrode_positions.csv file
df_baseline = pd.read_csv(os.path.join(path_Tables,'DF_2Methods_2Raters_All_Coord.csv'))

# Process each electrode
electrodes = ['anode_mni', 'cathode1_mni', 'cathode2_mni', 'cathode3_mni']
long_data = []

for electrode in electrodes:
    coord_df = df[['subject', 'session', 'run', electrode]].copy()
    coord_df['Electrode'] = electrode.replace('_mni', '').capitalize()
    # Extract the coordinates from the string format to a numpy array
    coord_df[electrode] = coord_df[electrode].apply(lambda x: np.array(x.strip('[]').split(), dtype=float))
    coord_df[['X', 'Y', 'Z']] = pd.DataFrame(coord_df[electrode].tolist(), index=coord_df.index)
    
    coord_df = coord_df.melt(id_vars=['subject', 'session', 'run', 'Electrode'],
                             value_vars=['X', 'Y', 'Z'],
                             var_name='Dimension', value_name='coordinate')
    long_data.append(coord_df)

# Combine all processed data
result_df = pd.concat(long_data, ignore_index=True)

# Add additional columns
result_df['Experiment'] = 'YourExperimentName'  # Replace with actual experiment name
result_df['Rater'] = 'Network'
result_df['Method'] = 'full-automated'

# Reorder columns
column_order = ['Experiment', 'subject', 'session', 'run', 'Rater', 'Method', 'Electrode', 'Dimension', 'coordinate']
df_results = result_df[column_order]

# Save to CSV
#result_df.to_csv('electrode_positions_long_format.csv', index=False)

# Assuming df_results is your transformed electrode_positions.csv data
# and df_original is your DF_2Methods_2Raters_All_Coord.csv data

# Rename columns in df_results to match df_original
df_results = df_results.rename(columns={
    'subject': 'Subject',
    'session': 'Session',
    'run': 'run',
    'Electrode': 'Electrode',
    'Dimension': 'Dimension',
    'coordinate': 'Coordinates'
})

# Ensure 'Experiment', 'Rater', and 'Method' columns exist in df_results
if 'Experiment' not in df_results.columns:
    df_results['Experiment'] = 'YourExperimentName'  # Replace with actual experiment name
if 'Rater' not in df_results.columns:
    df_results['Rater'] = 'Network'
if 'Method' not in df_results.columns:
    df_results['Method'] = 'full-automated'

# Reorder columns to match df_original
column_order = ['Experiment', 'Subject', 'Session', 'run', 'Rater', 'Method', 'Electrode', 'Dimension', 'Coordinates']
df_results = df_results[column_order]

# Merge df_results with df_original
df_combined = pd.concat([df_A, df_results], ignore_index=True)

# Sort the combined dataframe
df_combined = df_combined.sort_values(['Experiment', 'Subject', 'Session', 'run', 'Rater', 'Method', 'Electrode', 'Dimension', 'Coordinates'])

# Save the combined dataframe to a new CSV file
df_combined.to_csv('combined_electrode_positions.csv', index=False)

print("Combined CSV file has been created: combined_electrode_positions.csv")


