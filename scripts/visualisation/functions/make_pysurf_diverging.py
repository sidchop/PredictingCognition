#!/usr/bin/python
import argparse

parser = argparse.ArgumentParser(description='Output pysurfer cortical image for schaefer400n17')
parser.add_argument('-r','--rhpath', help='path to rh vector you want to plot (must beingwith 0 for medial wall',required=True)
parser.add_argument('-l','--lhpath', help='path to lh vector you want to plot (must beingwith 0 for medial wall',required=True)
parser.add_argument('-m','--min', help='minimum value',required=True)
parser.add_argument('-a','--max', help='maximum value',required=True)
parser.add_argument('-c','--colourmap', help='maximum value',required=True, default = "Reds")
parser.add_argument('-s','--surf', help='inflated, pial or white',required=True)
parser.add_argument('-k','--lhcol', help='vector of colors for lh',required=False)
parser.add_argument('-g','--rhcol', help='vector of colors for rh',required=False)

args = parser.parse_args()


#rhpath="~/Dropbox/Sid/R_files/STAGES_difussion/output/figures/temp/temp_degree_rh.txt"
#lhpath="~/Dropbox/Sid/R_files/STAGES_difussion/output/figures/temp/temp_degree_lh.txt"
#mim=0
#max=100
#colourmapc="viridis"
#surf="inflated"

import os
import numpy as np
import nibabel as nib
from surfer import Brain 
#print(__doc__)
hemis = "rh lh"
for h in hemis.split():
    subject_id = "fsaverage"
    hemi = h
    surf = args.surf #pick pial, inflated or white
    
    """
    Bring up the visualization.
    """
    brain = Brain(subject_id, hemi, surf,background="white", cortex = "grey")
    
    """
    Read in the automatic parcellation of sulci and gyri.
    """
    aparc_file = os.path.join('/Users/sidchopra/Dropbox/Sid/R_files/STAGES_fmri/data/fsaverage/label', hemi + '.Schaefer2018_400Parcels_17Networks_order.annot')
    labels, ctab, names = nib.freesurfer.read_annot(aparc_file)
    
    """
    Vector of scalar data corresponding to a value for each region in
    the parcellation.
    
    """
    
    if hemi == 'rh':
        roi_data = np.loadtxt(args.rhpath)
       
    if hemi == 'lh':
        roi_data = np.loadtxt(args.lhpath)
        
       
      
    """
    Make a vector containing the data point at each vertex.
    """
    vtx_data = roi_data[labels]

    """
    Handle vertices that are not defined in the annotation.
    """
    vtx_data[labels == -1] = 0
    vtx_data[labels == 0] = 0
    
    """
    Display these values on the brain. Use a sequential colormap (assuming
    these data move from low to high values), and add an alpha channel so the
    underlying anatomy is visible.
    """
    brain.add_data(vtx_data, float(args.min), float(args.max), colormap=args.colourmap, transparent = True, colorbar = False, mid = 0) #add  mid for 0 ( mid = 0)middle diverging colourmaps and remove thresh (thresh=0)
    
    #brain.add_contour_overlay(vtx_data, line_width=3, hemi='rh', colormap = [(1,1,1)])
    
    #change view
    brain.save_imageset(os.path.join('/Users/sidchopra/Dropbox/Sid/python_files/metamatching/scripts/visualisation/temp/'+ hemi), ['med', 'lat'], 'jpg')


#brain.show_view('lateral')
#brain.show_view('m')
#brain.show_view('rostral')
#brain.show_view('caudal')
#brain.show_view('ve')
#brain.show_view('frontal')
#brain.show_view('par')
#brain.show_view('dor')
