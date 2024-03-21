# Automated_Electrode_Coordinate_Extraction

- `electrode_extraction_from_network` includes scripts that are used to segment original petra images `inference` and create masks.
- images are in "/media/MeMoSLAP_Subjects/derivatives/automated_electrode_extraction/sub-**/electrode_extraction"

1. Use `cut_pads` skript to cut pads from the petra images (change the dilation if needed, good default values 6,7,8)
2. Use `inference` to choose dilated petra, create inference image of electrode mask `petra_cut_pad_*_inference.nii` 
3. Use `mask_separation` to separate the electrode mask from the single inference mask
4. Use `report` to create for each mask an image