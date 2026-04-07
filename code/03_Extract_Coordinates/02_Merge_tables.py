import pandas as pd
import ast
import numpy as np
from pathlib import Path
import os
import ast
import pickle
import glob

# define most import path variables
script_directory = Path(__file__).parent.resolve()
root = script_directory.parent.parent.resolve()

path_Tables = os.path.join(root, 'code','Extract_Coordinates','Tables')

def construct_baseline_coordinate_table(tables):
    root_table = '/media/MeMoSLAP_Mesh2/PDF_Report_Generation'
    pickle_filelist = glob.glob(os.path.join(path_Tables, '**','02-ANALYSIS', '**', '*.pkl'), recursive=True)
    dict_data = {}
    for file in pickle_filelist:
        folder_str = os.path.basename(os.path.dirname(file))
        #print(folder_str)
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

# Read the CSV files
df_ground_truth_wide = construct_baseline_coordinate_table(path_Tables)
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
df_combined = pd.concat([df_baseline, df_results], ignore_index=True)

# Sort the combined dataframe
df_combined = df_combined.sort_values(['Experiment', 'Subject', 'Session', 'run', 'Rater', 'Method', 'Electrode', 'Dimension', 'Coordinates'])

# Save the combined dataframe to a new CSV file
df_combined.to_csv(os.path.join(path_Tables,'combined_electrode_positions.csv'), index=False)

print("Combined CSV file has been created: combined_electrode_positions.csv")


