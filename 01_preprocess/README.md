# Coregistration script

- The script **Coregistrate_and_Create_mask.m** only works if your data are in **BIDS** format and have similar naming filters

## important change in the scriopt

- in the scirpt you have to change the variable `path` to your **BIDS source folder**
- also change the `coregistration_path` to a new folder on your Linux computer like **/derivatives/automated_electrode_extraction**
- the script will create unzipped coregistrate files in this directory to further process it with SPM
- line 44-46 provides the seach string for the *T1* and *PETRA* files, change if your naming scheme is different then the given
- line 285-305 change the path to your `spm12/tpm/TPM` path
- line 326-330, change the naming scheme according to line 44

## general use

- if a new subject is finished with all 4 sessions, run the script
- if the script run on subjects with onlys 3 sessions and session 4 is available delete that complete subject folder in **/derivatives/automated_electrode_extraction/sub-XXX** and run the script again

## how to use

- go to path of the Coregistrate.m script file in **/media/MeMoSLAP_Subjects/code/Coregistration_script**
- open an terminal

```bash
matlab
```

- open the script file in matlab and press the green play button (tab: Editor)
- everythin else will work automatically
  