#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 21:37:52 2020

@author: sidchopra
"""
import argparse
import numpy as np
import pyvista as pv

parser = argparse.ArgumentParser(description='Output subcortical image for aseg file')
parser.add_argument('-n','--numvec', help='path to 1:20 values for aseg (brain step is repeated for each hemi)',required=True)
parser.add_argument('-m','--min', help='minimum value',required=True)
parser.add_argument('-a','--max', help='maximum value',required=True)
parser.add_argument('-c','--colourmap', help='colour map',required=True)
parser.add_argument('-o','--output', help='output folder (usually temp dir)',required=True)

args = parser.parse_args()

#def get_subcortex_mesh_tian2(numvec = range(1,32), min=1,max=32,colourmap="viridis",outputfolder="~/Dropbox/Sid/R_files/STAGES_difussion/output/figures/temp/"):




#from numpy import inf
#import matplotlib.pyplot as plt

#max_degree = np.array([0,63, 63, 58, 58,21, 21, 6, 6, 21, 21, 12, 12, 5, 5]) 
mesh = pv.read('/Users/sidchopra/Dropbox/Sid/python_files/PredictingCognition/data/atlas/subcortex_lh.vtk')
meshSmooth = mesh.smooth(n_iter=800)


##get scalars
cells = mesh.active_scalars
unique_elements, counts_elements = np.unique(cells, return_counts=True)

#read in degree file
degree = np.loadtxt(args.numvec)

##degree = np.genfromtxt(degreefile, delimiter=',')

lh_degree = degree[0:10]
scalarsDegree = np.repeat(lh_degree, counts_elements, axis=0)

meshSmooth.plot(scalars=scalarsDegree, cmap=args.colourmap, off_screen=True,
          background="White", 
          parallel_projection=True, 
          clim = [int(args.min),int(args.max)],
          cpos=[1, 0, 0.4] ,screenshot=args.output + str('/sub_temp_lh.png'))






rh_degree = degree[10:20]
mesh2 = pv.read('/Users/sidchopra/Dropbox/Sid/python_files/PredictingCognition/data/atlas/subcortex_rh.vtk')
cells = mesh2.active_scalars
unique_elements, counts_elements = np.unique(cells, return_counts=True)
mesh2Smooth = mesh2.smooth(n_iter=800)

scalarsDegree2 = np.repeat(rh_degree, counts_elements, axis=0)

mesh2Smooth.plot(scalars=scalarsDegree2, cmap=args.colourmap, off_screen=True,
                  background="White", 
                  parallel_projection=True, 
                  clim = [int(args.min),int(args.max)],
                  cpos=[-1, 0, 0.4],
                  show_scalar_bar=True, screenshot=args.output + str('/sub_temp_rh.png'))
        
    



## Make legend 
#mesh = pv.read('/Users/sidchopra/Dropbox/Sid/R_files/STAGES_fmri/output/subcortex/tian_lh.vtk')
#meshSmooth = mesh.smooth(n_iter=200)
#
#
###get scalars
#cells = mesh.active_scalars
#unique_elements, counts_elements = np.unique(cells, return_counts=True)
#
#    #read in degree file
#degreefile ='/Users/sidchopra/Dropbox/Sid/R_files/STAGES_fmri/output/subcortex/sub_degree' + str(i) + '.txt'
#degree = np.genfromtxt(degreefile, delimiter=',')
#lh_degree = degree[0:16]
#scalarsDegree = np.repeat(lh_degree, counts_elements, axis=0)
#
#
#meshSmooth.plot(scalars=scalarsDegree, cmap="Reds", 
#                background="White", 
#                parallel_projection=True, 
#                clim = [0,max_degree[i]],
#                cpos=[1.5, 0.5, -1], screenshot='/Users/sidchopra/Dropbox/Sid/R_files/STAGES_fmri/output/subcortex/imgs/sub_'+str(i)+'_l.png')
#
#
#mesh2 = pv.read('/Users/sidchopra/Dropbox/Sid/R_files/STAGES_fmri/output/subcortex/tian_rh.vtk')
#mesh2Smooth = mesh2.smooth(n_iter=200)
#rh_degree = degree[16:32]
#scalarsDegree2 = np.repeat(rh_degree, counts_elements, axis=0)
#
#
#mesh2Smooth.plot(scalars=scalarsDegree2, cmap="Reds", 
#                     background="White", 
#                     parallel_projection=True, 
#                     clim = [0,max_degree[i]],
#                     cpos=[-10, 4, -6],
#                     show_scalar_bar=True, screenshot='/Users/sidchopra/Dropbox/Sid/R_files/STAGES_fmri/output/subcortex/imgs/sub_'+str(i)+'_r.png')
#
