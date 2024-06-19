# Electrode extraction by hand: "Pancake Method"
(written for Ubuntu)

0. Choose a folder in which you like to clone the github repository
   
- in this folder open a terminal and clone the repository
  
```bash
git clone https://github.com/LawsOfForm/Automated_Electrode_Coordinate_Extraction.git
```

- if the command "git" is not working install github on your computer  
  
```bash  
    sudo apt-get install git
```

- change directory into the Automated_Electrode_Coordinate_Extraction folder

```bash
cd  Automated_Electrode_Coordinate_Extraction
```

- change to the clean branch of the repository with the command

```bash
git checkout cleaned
```

- no you should be in the correct branch and repositories should be up to date

## Tabel of Contents

1. [Matlab](#matlab)
2. [Python](#python) 2.1. [Adjustments](#adjustments)

## Matlab

1. open Matlab 2019 </br> ... by typing `matlab_2019` into the terminal and
   pressing `Enter`

   or

   ... by using matlab icon in the application launcher:

   1. open the application launcher:

      ![Application Launcher](./sop/images/applications.png)

   2. click on the matlab icon:

      ![Matlab 2019a Icon](./sop/images/matlab2019a.png)

   The scripts will only work with Matlab 2019 and not with other Matlab
   versions installed on this machine. If you are not sure which version is open
   just type `version` in the Command Window (pink box in the image below) and
   press `Enter`

2. use the script `Coregistrate_and_Create_mask.m ` in the `01_preprocess` folder to coregistrate the PETRA images onto the first T1 baseline

   - for clearer instructions read the README.md in the folder.

3. Get electrode coordinates -> use the script in the `02_electrode_extraction_by_hand` folder
   `hand_labeling_memoslap_no_mid_point.m`

4. Adjust the first lines of the matlab script

   ```matlab
   sub_to_analyse = 'sub-010'; %insert here
   session = 1; % and here
   run = 1; % and here
   ```

5. Adjust also the path of `electrode_dir` to the path created in point 2 with script `Coregistrate_and_Create_mask.m `

   -(something like `**/derivative/automated_electrode_extraction`)

   These are the only lines you have to adjust.

6. Run the script by pressing `Run` (green box) or `F5`. This will run
   the script and after a while, the pancake view of a structural brain scan will
   pop up. Here every electrode is extracted with 6 points near the rim of the
   electrode. Be careful to place the points as close to the rim of the
   electrode as possible, but still **on** the electrode not beside the
   electrode, as the computation of the electrode centre otherwise will be
   imprecise. The first electrode you mark has always to be the anode, i.e., the
   electrode in the middle of the montage. The other electrodes do not have to
   be extracted in a specific order. </br> </br> If you misplace any point,
   close the window and rerun the script. This will also delete all
   previously made markings. When you marked all electrodes (i.e., made 6
   markings on all 4 electrodes) just close the window with the pancake view.
   The script will continue to run and output some coordinates in the command
   window (pink box).

7. Control the outputs of the script in the derivatives directory (
   [**/derivatives/automated_electrode_extraction](**/derivatives/automated_electrode_extraction))
   </br></br> According to the subject info you provided the Matlab script
   directories are created with the subject-id, session, and run. The bottom
   directory should be populated with the files shown below in the file tree
   (i.e., 30 mricoords files, finalmask.nii.gz, handextracted_electrode_pos.csv,
   ...). If you also processed the other MRI data of the subjects for all
   processed sessions and runs the same files should be available.

8. coordinates will be written into `handextracted_electrode_pos.csv`

```file-tree
sub-010/
┗ electrode_extraction/
┃ ┣ ses-1/
┃ ┃ ┣ run-01/
┃ ┃ ┃ ┣ finalmask.nii.gz
┃ ┃ ┃ ┣ handextracted_electrode_pos.csv
┃ ┃ ┃ ┣ idqs.nii.gz
┃ ┃ ┃ ┣ layers.nii.gz
┃ ┃ ┃ ┣ layers_binarized.nii.gz
┃ ┃ ┃ ┣ maskvq.nii.gz
┃ ┃ ┃ ┣ mricoords_1.mat
┃ ┃ ┃ ┣ mricoords_10.mat
┃ ┃ ┃ ┣ mricoords_11.mat
┃ ┃ ┃ ┣ mricoords_12.mat
┃ ┃ ┃ ┣ mricoords_13.mat
┃ ┃ ┃ ┣ mricoords_14.mat
┃ ┃ ┃ ┣ mricoords_15.mat
┃ ┃ ┃ ┣ mricoords_16.mat
┃ ┃ ┃ ┣ mricoords_17.mat
┃ ┃ ┃ ┣ mricoords_18.mat
┃ ┃ ┃ ┣ mricoords_19.mat
┃ ┃ ┃ ┣ mricoords_2.mat
┃ ┃ ┃ ┣ mricoords_20.mat
┃ ┃ ┃ ┣ mricoords_21.mat
┃ ┃ ┃ ┣ mricoords_22.mat
┃ ┃ ┃ ┣ mricoords_23.mat
┃ ┃ ┃ ┣ mricoords_24.mat
┃ ┃ ┃ ┣ mricoords_25.mat
┃ ┃ ┃ ┣ mricoords_26.mat
┃ ┃ ┃ ┣ mricoords_27.mat
┃ ┃ ┃ ┣ mricoords_28.mat
┃ ┃ ┃ ┣ mricoords_29.mat
┃ ┃ ┃ ┣ mricoords_3.mat
┃ ┃ ┃ ┣ mricoords_30.mat
┃ ┃ ┃ ┣ mricoords_4.mat
┃ ┃ ┃ ┣ mricoords_5.mat
┃ ┃ ┃ ┣ mricoords_6.mat
┃ ┃ ┃ ┣ mricoords_7.mat
┃ ┃ ┃ ┣ mricoords_8.mat
┃ ┃ ┃ ┣ mricoords_9.mat
┃ ┃ ┃ ┣ petra_.nii.gz
┃ ┃ ┃ ┣ petra_masked.nii.gz
┃ ┃ ┃ ┗ rgbs.png
┃ ┃ ┗ run-02/
...
┃ ┣ ses-2/
┃ ┃ ┗ run-01/
...
┃ ┣ ses-3/
┃ ┃ ┗ run-01/
...
┃ ┗ ses-4/
┃   ┗ run-01/
...
```
