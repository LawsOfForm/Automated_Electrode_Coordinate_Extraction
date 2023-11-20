# Coregistration script

- take input from the SUBJECTS_XNAT folder
- creates new folder on the Linux5 /MeMoSLAP computer **/media/MeMoSLAP_Subjects/derivatives/coregistration_coordinates**
- created unzipped files in this folder
- runs batch automatically

## general use

- if a new subject is finished with all 4 sessions, run the script
- if the script run on subjects with onlys 3 sessions and session 4 is available delete that complete subject folder in **/media/MeMoSLAP_Subjects/derivatives/coregistration_coordinates/sub-XXX** and run the script again

## how to use

- go to path of the Coregistrate.m script file in **/media/MeMoSLAP_Subjects/code/Coregistration_script**
- open an terminal

```bash
matlab
```

- open the script file in matlab and press the green play button (tab: Editor)
- everythin else will work automatically