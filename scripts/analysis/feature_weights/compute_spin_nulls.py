#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 10:23:56 2021

@author: sidchopra
"""


from netneurotools import datasets, freesurfer, stats

annot = datasets.fetch_schaefer2018('fsaverage5')
annot = annot['400Parcels17Networks']


coords, hemi = freesurfer.find_parcel_centroids(lhannot=annot.lh,
                                                rhannot=annot.rh,
                                                version='fsaverage5',
                                                surf='sphere',
                                                method='surface')
spins = stats.gen_spinsamples(coords, hemi, method='vasa', seed=24021993, n_rotate=10000)
#nulls = brain[spins]
#write out spins

import numpy
a = numpy.asarray(spins)
numpy.savetxt("/Users/sidchopra/Dropbox/Sid/python_files/metamatching/output/schaefer400_17_vasa_10000spins.csv", a, delimiter=",")


