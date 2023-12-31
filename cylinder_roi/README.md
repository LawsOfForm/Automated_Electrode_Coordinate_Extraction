`paths.py` and all scripts in the `util` directory have some helperfunctions and variables to deal with the location of files given our local computer (i.e., `Linux-Meiner`), and functions that are used in the scripts, respectively.

1. `cylinder_roi.py`: Create a cylindrical roi given the [extracted electrode locations](../electrode_extraction_by_hand). For every electrode three coordinates have to be extracted and one of them has to be centre of the electrode. These coordinates are used to compute a vector that is perpendicular to the plane that all three electrodes include (i.e., the cross-product of the vectors spanned by the three coordinates or the normal vector). A binary cylindrical mask is constructed with the centre of the bottom at the electrode centre and afterwards rotated to face into the same direction as the normal vector of the plane. So a cylinder is constructed containing the electrode.
2. `SimpleITK_refinement.py`: the cylindrical roi is used to delineate an area in which a edge detection algorithm is used to detect the edges of the electrode which are used afterwards to compute the convex hull image (i.e., smallest convex polygon that surround all white pixels in the input image).
3. `mask_seperation.py`: create a file for all four electrode masks.
4. `report.py` creates a report directory in which for every participant, session, and run screenshots from the electrode placement are created. This provides a good visual check if the electrode extraction by hand was successful and precise enough.


