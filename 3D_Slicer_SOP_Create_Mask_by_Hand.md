# Create Masks by Hand

- we need a software called [3DSlicer](https://www.slicer.org/)

## open 3D Slicer in terminal

```bash
Slicer
```

## Data
### path

- */media/MeMoSLAP_Subjects/derivatives/automated_electrode_extraction*

### files
- __USE ONLY REALIGNED IMAGES__ 

- *rsub-0xx_ses-x_acq-petra_run-0x_PDw.nii*

## Process of segmenting electrodes

### overall procedure

- 1) create a segment
- 2) Get optimal visual of electrode
    - Choose one electrode
    - Align the electrod in the middle
    - Crosshair of AS and LS should perfectly be plane to electrode surface
    - To achieve that rotate the SPR and AS images via "Transforms" module

- 3) Add circular segment over gel and electrode plug   
    - Choose another electrode and add segement 
    - Export segmented electrodes

- Save between steps to avoid data loss when the program crashes
- save project data in folder named like *sub-0xx_ses-x_run-x*
- check everytime IF THE FOLDERNAME AND PATH IS CORRECT, otherwise data will be overwriten and lost

### Step by step
### Adjust View 

- change View in the menu bar
*View -> Layout -> Four-Up*
    - you should see now a 2x2 window in wich on the 
    - upper left: SPR (Horizontal) plane is displayed in red
    - upper right: 3D image of the head is shown (feature must be activated)
    - bottom left: AS (Coronal) plane in green
    - bottom right: LS (Saggital) plane in yellow
- Link the slice views by clicking on the 2 rings in each window
- Rings have to appear closed
- Zomm in and Out: STRG + mouse wheel

### Switching slider menu on the left of the image

- Adding segments, grouping segments and images etc. can be done with __MODULES__. 
- All available modules are in listed in the slide menu next to "Modules:" Icon on the top left next to "Data", "DCN" and "SAVE" Icon
- This is called in the text the **Modules: slide bar**
- Choose the module you need like *Segment Editor* or *Transforms* or *Views* 
- On the left of the 3D images you will no find the **module menu**
- In further instruction we will refer to **Modules: slide bar** and **module menu** to not get confused

### 1) Add Segments 

#### **Modules: slide bar**

- Choose on the slide bar *Segmentation Editor* to open the editor menu 

####  **module menu**

- Go to "Add" and add 4 segments

### 2) Apply Transforms

#### **Modules: slide bar**

- Choose on the slide bar *Transforms* to open the menu opportunities
- On the menu bar on the left are now all options for image transformations we like to apply

####  **module menu**

- Go to *Active Transform:* and choose the the slide bar *Linear transforms*
- Before you start the transformations go to *Apply transform* 
- Select the "Segment" and the "loaded image" to process transformation on both simultanously
- Use the Translation and Rotation bars to put the electrode in the middle and to align them to the crosshair of one dimension
- example:
    - the crosshair in each image has a vertical and horizontal line
    - in the slice the electrode gel forms also a line
    - the line of the gel shoul be parallel with the horizontal line in exact 2 images
    - the SPR image and the AS image (see image descripitions above the slide bars of the images
    - if the lines are parallel in both images in the LS image the gel shoul appear nearly as perfect circle
    - scroll to the outmost point from which a clear circular shape is detectable
    - start segmenting the electrode with a painter that is circular (details in the next step)
    
#### tips

- if you place the mous on one point of the imag an then press shift for a wile the crosshair of all images will be centered to that point
- so if you target a new electrode and you found it. Place the mouse on it and press shift for a while
 
### 3) Segment the electrode

#### **Modules: slide bar**

- Choose on the slide bar *Segmentation Editor* to open the editor menu 

####  **module menu**

- Click on the segment you want to create (exmaple named Segment_1)
- Click on the paint icon to get a circular painter
- Increase or decrease the radius of the painter: shift+mouse whell
- The radius should perfectly match the radius of the electrode or gel
- If the circular painter is exctly matching the boundaries of the gel, click to segment the first slide
- use the mouse wheel in the LS window to get to the next slide
- Use the *Show 3D* button 
- If the gel and the electrode are fully covered go to the next electrode

- at the end of all segmentations use the smooting option to smooth the segments
- and the *Fill between slices* to fill between the segments

- _Export_ Segments as *.nii file
- click on the little sidebar next to the green error (in the same line in which you Add and Remove segments)
- click *Export to files*
- make sure to choose the correct folder and click *Export*

#### tips

- if the electrode is deformed or the gel is spread to more tissue then normally, decrease the radius of the painter to get those areas too and use the erase to to erase uncorrect segmentet areas

- if the painter is not drawing sometimes in the menu below the point "Editable area:" is not set to everywhere an therefore the painter can not draw everywhere





