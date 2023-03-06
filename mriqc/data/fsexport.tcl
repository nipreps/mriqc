# Sample script for setting up and taking screen shots. Scripting
# reference is available at:

# https://surfer.nmr.mgh.harvard.edu/fswiki/TkMeditGuide/TkMeditReference/TkMeditScripting

# You can set the cursor or view center with the SetCursor command.



# Alternatively you can set the slice number (in volume index
# coordinates). This will not change the in-plane center.

# Use SetZoomLevel to zoom in and out. 1 is normal, >1 is zoomed in,
# and 0-1 is zoomed out.

# SetZoomLevel level 
SetZoomLevel 2
# SetZoomCenter 0 0 0

# SetOrientation orientation 
# orientation:
# 0     coronal
# 1     horizontal
# 2     sagittal
SetOrientation 0

# This command turns on and off various display flags.
# SetDisplayFlag flag value 
# flag:
# 1     Aux Volume - set to 1 to show aux volume
# 2     Anatomical Volume - set to 0 to hide main and aux volume
# 3     Cursor
# 4     Main Surface
# 5     Original Surface
# 6     Pial Surface
# 7     Interpolate Surface Vertices
# 8     Surface Vertices
# 9     Control Points
# 10    Selection
# 11    Functional Overlay
# 12    Functional Color Scale Bar
# 13    Mask to Functional Overlay
# 14    Histogram Percent Change
# 15    Segmentation Volume Overlay
# 16    Aux Segmentation Volume
# 17    Segmentation Label Volume Count
# 18    DTI Overlay
# 20    Focus Frame
# 21    Undoable Voxels
# 22    Axes
# 23    Maximum Intensity Projection
# 24    Head Points
# 25    Verbose GCA DumpSetDisplayFlag 

# SetCursor coordinateSpace x y z 
# coordinateSpace:
# 0     volume index
# 1     RAS
# 2     Talairach
SetCursor 0 128 128 128

# Turn cursor display off.
SetDisplayFlag 3 0

# Turn the axes on.
SetDisplayFlag 22 1

# Use this command to go to multiple views. This will copy the current
# view settings from the current view, so all the above commands will
# apply to all new views.
# SetDisplayConfig numberOfColumns numberOfRows linkPolicy 
# linkPolicy:
# 0     none
# 1     linked cursors
# 2     linked slice changes
# SetDisplayConfig 2 2 1

# Use the RedrawScreen command to force a redraw after you have a view
# set up, beofre taking a picture.
# RedrawScreen

# This command will save the actual screenshot.
# SaveTiff fileName
# SaveTIFF $::env(FS_OUTPUT_PATH)/screenshot.tif


# Use tcl loops to change orientations and take multiple
# screenshots. This will set the view to a single view, and take three
# screenshots, one of each orientation.
# SetDisplayConfig 1 1 0
# foreach orientation {0 1 2} label {cor horiz sag} {

#     SetOrientation $orientation
#     RedrawScreen
#     SaveTIFF $::env(FS_OUTPUT_PATH)/screenshot-$label.tif
# }

# Or take pictures of multiple slices for a movie. This goes through
# slices 0-255 and takes a shot at each one.
for { set slice 30 } { $slice < 226 } { incr slice } {
    SetZoomCenter 160 80 $slice
    SetSlice $slice
    RedrawScreen
    SaveTIFF $::env(FS_OUTPUT_PATH)[format "/screenshot-%03d.tif" $slice]
}

QuitMedit
