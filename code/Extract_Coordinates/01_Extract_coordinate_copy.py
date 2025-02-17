import os
import glob
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.ndimage import center_of_mass, label, sum
from scipy.spatial.distance import euclidean
#from scipy.ndimage import measurements
from nilearn import image
from itertools import permutations
from pathlib import Path

# define most import path variables
script_directory = Path(__file__).parent.resolve()
root = script_directory.parent.parent.resolve()

#root_images = '/media/Data03/Thesis/Hering/derivatives/automated_electrode_extraction'

root_images = '/media/MeMoSLAP_Subjects/derivatives/automated_electrode_extraction'
Table_path = os.path.join(root,'code','Extract_Coordinates','Tables')


def find_nifti_files(base_path):
    pattern = os.path.join(base_path, "sub-*", "unzipped", "*inference.nii.gz")
    #pattern = os.path.join(base_path, "sub-*", "electrode_extraction","ses*","run*", "*cut_pads_inference.nii.gz")
    #pattern = os.path.join(base_path, "sub-*", "electrode_extraction","ses*","run*", "petra_inference_2.nii.gz")
    return glob.glob(pattern)

def load_nifti(file_path):
    return nib.load(file_path).get_fdata()

def voxel_to_mni(voxel_coords, affine):
    mni_coords = nib.affines.apply_affine(affine, voxel_coords)
    return mni_coords

def find_electrode_clusters(image):
    labeled_array, num_features = label(image > 0)
    
    clusters = {}
    
    if num_features > 4:
        # Calculate sizes of all clusters
        #cluster_sizes = measurements.sum(image > 0, labeled_array, index=range(1, num_features + 1))
        cluster_sizes = sum(image > 0, labeled_array, index=range(1, num_features + 1))
        
        # Calculate average cluster size
        average_cluster_size = np.mean(cluster_sizes)
        
        # Threshold for cluster size (20% of average)
        size_threshold = 0.2 * average_cluster_size
        
        for i in range(1, num_features + 1):
            if cluster_sizes[i-1] >= size_threshold:
                cluster_mask = labeled_array == i
                cluster_coords = np.array(np.where(cluster_mask)).T
                cluster_com = center_of_mass(cluster_mask)
                clusters[i] = {
                    'coords': cluster_coords,
                    'center_of_mass': cluster_com,
                    'size': cluster_sizes[i-1]
                }
    else:
        for i in range(1, num_features + 1):
            cluster_mask = labeled_array == i
            cluster_coords = np.array(np.where(cluster_mask)).T
            cluster_com = center_of_mass(cluster_mask)
            clusters[i] = {
                'coords': cluster_coords,
                'center_of_mass': cluster_com
            }
    
    return clusters

def is_valid_configuration(clusters,counters, subject, session, run):

    if len(clusters) != 4:
        if len(clusters)==3:
            counters['three_mask_detected'] +=1
            counters['three_mask_detected_sub'].append(f'{subject}_{session}_{run}')
        elif len(clusters)==2:
            counters['two_mask_detected'] +=1
            counters['two_mask_detected_sub'].append(f'{subject}_{session}_{run}')
        elif len(clusters)==1:
            counters['one_mask_detected'] +=1
            counters['one_mask_detected_sub'].append(f'{subject}_{session}_{run}')
        elif len(clusters)==0:
            counters['no_mask_detected'] +=1
            counters['no_mask_detected_sub'].append(f'{subject}_{session}_{run}')
        else:
            counters['more_then_four_mask_detected'] +=1
            counters['more_then_four_mask_detected_sub'].append(f'{subject}_{session}_{run}')
        return False, counters
    
    centers = [cluster['center_of_mass'] for cluster in clusters.values()]
    
    # Check all possible permutations of centers
    for perm in permutations(centers):
        center = perm[0]
        satellites = perm[1:]
        
        distances = [euclidean(center, satellite) for satellite in satellites]
        
        # Check if all distances are within the valid range
        if all(5 <= d <= 70 for d in distances):
            return True, counters
    
    return False, counters

def process_nifti_files(base_path):
    nifti_files = find_nifti_files(base_path)
    results = []
    counters = {
        'total_images': 0,
        'valid_configurations': 0,
        'invalid_configurations': 0,
        'three_mask_detected': 0,
        'two_mask_detected': 0,
        'one_mask_detected': 0,
        'no_mask_detected': 0,
        'more_then_four_mask_detected': 0,
        'three_mask_detected_sub': [],
        'two_mask_detected_sub': [],
        'one_mask_detected_sub': [],
        'no_mask_detected_sub': [],
        'more_then_four_mask_detected_sub': []
    }
    for file_path in nifti_files:
        counters['total_images'] += 1
        parts = file_path.split(os.sep)
        subject = parts[-5]
        session = parts[-3]
        run = parts[-2]

        # Load the image using nibabel to get the affine matrix
        nii_img = nib.load(file_path)
        affine = nii_img.affine
        image = nii_img.get_fdata()

        clusters = find_electrode_clusters(image)

        is_valid, counters = is_valid_configuration(clusters, counters, subject, session, run)

        if is_valid:
            counters['valid_configurations'] += 1
            cluster_list = list(clusters.values())

            # Transform voxel coordinates to MNI coordinates
            anode_mni = voxel_to_mni(cluster_list[0]['center_of_mass'], affine)
            cathode1_mni = voxel_to_mni(cluster_list[1]['center_of_mass'], affine)
            cathode2_mni = voxel_to_mni(cluster_list[2]['center_of_mass'], affine)
            cathode3_mni = voxel_to_mni(cluster_list[3]['center_of_mass'], affine)

            results.append({
                'subject': subject,
                'session': session,
                'run': run,
                'anode_mni': anode_mni,
                'cathode1_mni': cathode1_mni,
                'cathode2_mni': cathode2_mni,
                'cathode3_mni': cathode3_mni
            })
        else:
            counters['invalid_configurations'] += 1

    for key, value in counters.items():
        print(f"{key.replace('_', ' ').capitalize()}: {value}")


    return pd.DataFrame(results)

# Usage

df = process_nifti_files(base_path=root_images)

df.to_csv(os.path.join(Table_path,'electrode_positions.csv'), index=False)
