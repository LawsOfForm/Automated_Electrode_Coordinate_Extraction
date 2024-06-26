# Electrode extraction by hand: "Pancake Method"

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

2. When matlab is open select the correct path to your working directory, i.e.,
   the path to your Thesis directory (either
   [/media/Data03/Thesis/Dabelstein/code/electrode_extraction_by_hand](/media/Data03/Thesis/Dabelstein/code/electrode_extraction_by_hand)
   or
   [/media/Data03/Thesis/Hering/code/electrode_extraction_by_hand](/media/Data03/Thesis/Hering/code/electrode_extraction_by_hand)).
   This path should shown in the matlab window (red box image below)

   ![Matlab after path adjustment](./sop/images/matlab_adjust_path.png)

3. If you correctly adjusted the path, you should see the
   `hand_labeling_memoslap_no_mid_point.m` script on the left side under
   **Current Folder** (blue box). Open this script by double clicking it.

4. Adjust the first lines of the matlab script

   ```matlab
   sub_to_analyse = 'sub-010'; %insert here
   session = 1; % and here
   run = 1; % and here
   ```

   These are the only lines you have to adjust.

5. Run the script by pressing `Run` (green box) or `F5`. This will run
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

   ![pancake view](/media/MeinzerShare/05_Zwischenablagen/pancake_view.png)

6. Control the outputs of the script in the derivatives directory (either
   [/media/Data03/Thesis/Dabelstein/derivatives/automated_electrode_extraction](/media/Data03/Thesis/Dabelstein/derivatives/automated_electrode_extraction)
   or
   [/media/Data03/Thesis/Hering/derivatives/automated_electrode_extraction](/media/Data03/Thesis/Hering/derivatives/automated_electrode_extraction))
   </br></br> According to the subject info you provided the Matlab script
   directories are created with the subject-id, session, and run. The bottom
   directory should be populated with the files shown below in the file tree
   (i.e., 30 mricoords files, finalmask.nii.gz, handextracted_electrode_pos.csv,
   ...). If you also processed the other MRI data of the subjects for all
   processed sessions and runs the same files should be available.

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

## Python

The Python scripts in the cylinder-roi directory (either
[/media/Data03/Thesis/Dabelstein/code/cylinder_roi](/media/Data03/Thesis/Dabelstein/code/cylinder_roi)
or
[/media/Data03/Thesis/Hering/code/cylinder_roi](/media/Data03/Thesis/Hering/code/cylinder_roi))
allow you to check how precise your electrode placement was.

The main reason to run these commands is to get the centre of the extracted
electrode and to get a visual report to control the electrode placement.

1. Open a terminal in the cylinder-roi directory (right-click anywhere in the
   directory and select `Open in terminal`)

2. The packages to run the Python scripts are installed into a virtual
   environment. If this environment is not active, the scripts will throw a
   "ModuleNotFound" error. Activate the virtual environment by typing the
   following command in the terminal and execute the command by pressing
   **Enter**.

   ```bash
   conda activate electrode_extraction
   ```

3. run the `create_cylinder.py` script. This script will a electrode shaped
   region of interest using the coordinates extracted with the matlab script.
   You can run the script by writing the following command in the terminal and
   pressing **Enter**:

   ```bash
   python create_cylinder.py
   ```

   This script will also create a text file called `mid.txt`. This textfile
   centres of the electrodes in the format (electrodes x dimensions), i.e., the
   first row contains the x,y,z coordinates of the first electrode, the second
   row contains the x,y,z coordinates of the second electrode and so on. The
   coordinate dimensions are separated by commas, thus this file format is
   called comma-separated values (CSV).

4. After this command, run the `mask_separation.py` script. This should create a
   single nifti image for every electrode extracted. Depending on how hard it is
   to separate the electrodes the script will take a long time to run for a
   single participant (see [adjustments](#Adjustments)).

   ```bash
   python mask_separation.py
   ```

5. Finally, run `report.py`:

   ```bash
   python report.py
   ```

   This script will create a `reports` directory in you code directory (either
   [/media/Data03/Thesis/Dabelstein/code/cylinder_roi](/media/Data03/Thesis/Dabelstein/code/cylinder_roi)
   or
   [/media/Data03/Thesis/Hering/code/cylinder_roi](/media/Data03/Thesis/Hering/code/cylinder_roi)).
   The script will create an image for every processed subject with the naming
   scheme `sub-<id>_ses-<num>_run-<num>.png`. This image shows the subject's
   structural image overlayed with the segmented electrodes created by the
   `mask_separation.py` script. This allows us to check the precision of the
   electrode extraction, i.e., if the electrode, which can be seen in the
   structural image, is enclosed by an extracted electrode and if the real and the
   extracted electrode have similar tilt.

   Run this step to control if your electrode extraction with the Matlab script
   was precise and repeat the extraction, and all the following steps if
   necessary.

### Adjustments

Some adjustments can be made to make the Python scripts run faster. To make
these adjustments in the scripts, just open the script file by double-clicking
it and change the parts described below.

#### `create_cylinder.py`

Python code can be commented with `#` at the start of the line. Commented code
is not executed and normally this is used to write explanations for the code.

The script `create_cylinder.py` can be adjusted to only run if the electrodes
are newly extracted by opening the script with a text editor and deleting the `#`
signs in front of these lines:

before:

```python
# if op.exists(cylinder_mask_path) and op.exists(cylinder_mask_plus_plug):
#    continue
```

after:

```python
if op.exists(cylinder_mask_path) and op.exists(cylinder_mask_plus_plug):
    continue
```

This checks in the output directory `cylinder_ROI.nii.gz` and
`cylinder_plus_plug_ROI.nii.gz` exists. The script is not run for subjects, if
these files exist. This will make the execution of the script much faster.

#### `mask_separation.py`

`mask_separation.py` has a similar code section that is commented out, i.e., a
section that checks if the product of the script is already present and does not
rerun the code if it is. You can activate this feature by removing the `#` at
the following lines:

before:

```python
# if glob(op.join(sub_dir, "*mask_*.nii.gz")):
#     continue
```

after:

```python
if glob(op.join(sub_dir, "*mask_*.nii.gz")):
    continue
```

#### `report.py`

The same goes for the `report.py` script.

before:

```python
# if op.isfile(report):
#     continue
```

after:

```python
if op.isfile(report):
    continue
```
