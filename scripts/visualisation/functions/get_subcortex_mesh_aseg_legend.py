#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 21:37:52 2020

@author: sidchopra
"""
import numpy as np
import pyvista as pv
import matplotlib
import sys
import os

path_repo = '/Users/sidchopra/Dropbox/Sid/python_files/PredictingCognition/'
sys.path.append(path_repo)

rh_degree = list(range(0,10))
mesh2 = pv.read(os.path.join(path_repo, 'data/atlas/subcortex_rh.vtk'))

cells = mesh2.active_scalars
unique_elements, counts_elements = np.unique(cells, return_counts=True)
mesh2Smooth = mesh2.smooth(n_iter=600)

scalarsDegree2 = np.repeat(rh_degree, counts_elements, axis=0)



cmap = np.array([[228,137,68],
                 [34,82,233], 
                 [125,238,239], 
                 [221,69,223],
                 [132,28,193], 
                 [249,226,80],
                 [212,53,80],
                 [49,113,39],
                 [144,213,71],
                 [225,225,225]])
cmap = cmap/255

my_colormap = matplotlib.colors.ListedColormap(cmap)

mesh2Smooth.plot(scalars=scalarsDegree2, cmap=my_colormap, off_screen=True,
                  background="White", 
                  parallel_projection=True, 
                  clim = [0,9],
                  cpos=[-1, 0, 0.4],
                  show_scalar_bar=False, 
                  screenshot=os.path.join(path_repo,'output/figures/vector_files/aseg_subcortex_pyvista_uncroped.png'))
        
    


