#!/usr/bin/env python3

# In[0]:
# initialization and random seed set
import os
import random
import numpy as np
import pandas as pd
import sys
#set repo path
path_repo = '/Users/sidchopra/Dropbox/Sid/python_files/PredictingCognition/'
path_v1 = os.path.join(path_repo, 'Meta_matching_models/v1.0')
sys.path.append(path_v1)
from scipy.stats import zscore
from CBIG_model_pytorch import stacking
from sklearn.metrics import r2_score
seed = 42
random.seed(seed)

# In[1]:
# load in FC data for each of the three clincial datasets
#HCPEP
fc_data_hcpep = np.loadtxt(os.path.join(path_repo, 'data/HCPEP/brain/FC_n163_GSR.txt'))

#TCP (aka PC)
fc_data_tcp = np.loadtxt(os.path.join(path_repo, 'data/TCP/brain/FC_mean_matched_n101_GSR.txt'))

#CNP (aka UCLA)
fc_data_cnp = np.loadtxt(os.path.join(path_repo, 'data/CNP/brain/FC_n256_matched_GSR_noSchz.txt'))


# z-score FC values
fc_data_hcpep = zscore(fc_data_hcpep, axis =1)
fc_data_tcp = zscore(fc_data_tcp, axis =1 )
fc_data_cnp= zscore(fc_data_cnp, axis =1)

# In[2]:
# load in global cognitve function scores
# HCP

cogpc_hcpep = pd.read_csv(os.path.join(path_repo, 'data/HCPEP/behaviour/pheno_cog_PC_n145.txt'), sep=" ")

cogpc_tcp = pd.read_csv(os.path.join(path_repo, 'data/TCP/behaviour/pheno_cog_PC_n101.txt'), sep=" ")

cogpc_cnp = pd.read_csv(os.path.join(path_repo, 'data/CNP/behaviour/pheno_cog_PC_n256_noSchz.txt'), sep=" ")
cogpc_cnp['Cog_PC1']  = cogpc_cnp ['Cog_PC1']*-1



# In[4]:
# Match and check fc and cog data subjects and ordering

#subselect HCP data
hcp_ep_subjlist = pd.read_csv(os.path.join(path_repo, 'data/HCPEP/behaviour/fmri_subj_id_order_n163.txt'), sep=" ")
inclu_subj_index = hcp_ep_subjlist["x"].isin(cogpc_hcpep["src_subject_id"])
#use index to select fc data
fc_data_hcpep = fc_data_hcpep[inclu_subj_index,:]
#convery to np and remove subj id
cogpc_hcpep = cogpc_hcpep.drop(['src_subject_id'], axis=1) 
cogpc_hcpep = cogpc_hcpep.to_numpy()

#convery to np and remove subj id
cogpc_cnp  = cogpc_cnp.drop(['V1'], axis=1) 
cogpc_cnp  = cogpc_cnp.to_numpy()

cogpc_tcp = cogpc_tcp.drop(['SUBJECT_ID', 'Group'], axis=1) 
cogpc_tcp = cogpc_tcp.to_numpy()



pheno_list = [cogpc_hcpep, cogpc_tcp, cogpc_cnp]
fc_data_list = [fc_data_hcpep, fc_data_tcp, fc_data_cnp]


names = ["hcpep", "tcp", "cnp"]

cor_out = np.empty((3, 3))
COD_out = np.empty((3, 3))

#observerd 
for i in range(3):
    for j in range(3):
        
        #skip diag (i.e. i==j)
        if i==j:
            cor_out[j,i] = 0
            COD_out [i,j] = 0
            continue
        # Assign train/test
        y_input = pheno_list[i]
        y_train = y_input
        
        x_input = fc_data_list[i]
        x_train = x_input
        
        # Assign train/test
        y_test=pheno_list[j]
        x_test=fc_data_list[j]
        
        for p in range(y_input.shape[1]):
            y_test_pred, _ , _ = stacking(x_train, x_test, y_train[:,0],  splits=5)
            cor_out[j,i] = np.corrcoef(y_test_pred, y_test[:,0])[0,1]
            COD_out[j,i] = r2_score(y_test[:,0], y_test_pred)
            obs_pred =  pd.DataFrame(np.c_[y_test_pred, y_test[:,0]])
            obs_pred.to_csv(os.path.join(path_repo, 'output/generalizability/krr_train_'+names[i]+'_test_'+ names[j] + '_noSchz.csv'), 
                                   sep=' ',
                                   index=False,
                                   header=False)
        
            print(j,i)

#Write out matrix

np.savetxt(os.path.join(path_repo, 'output/generalizability/krr_generalizability_matrix_noSchz.txt'), cor_out)
np.savetxt(os.path.join(path_repo, 'output/generalizability/krr_generalizability_matrix_COD_noSchz.txt'), COD_out)


