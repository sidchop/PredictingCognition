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


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr, zscore
from CBIG_model_pytorch import stacking
from sklearn.metrics import r2_score
seed = 42
random.seed(seed)

#assign args
ind=int(sys.argv[1]) # study sample 0=hcpep; 1 = TCP ;2 = CNP
regress_covars = eval(sys.argv[2]) #regress age,sex,fd from cog var 

# In[1]:
# load in FC data for each of the three clincial datasets

#HCPEP
fc_data_hcpep = np.loadtxt(os.path.join(path_repo, 'data/HCPEP/brain/FC_n163_GSR.txt'))

#TCP (aka PC)
fc_data_tcp = np.loadtxt(os.path.join(path_repo, 'data/TCP/brain/FC_mean_matched_n101_GSR.txt'))

#CNP (aka UCLA)
fc_data_cnp = np.loadtxt(os.path.join(path_repo, 'data/CNP/brain/FC_n256_matched_GSR.txt'))



# In[2]:
# load in global cognitve function scores

cogpc_hcpep = pd.read_csv(os.path.join(path_repo, 'data/HCPEP/behaviour/pheno_cog_PC_n145.txt'), sep=" ")

cogpc_tcp = pd.read_csv(os.path.join(path_repo, 'data/TCP/behaviour/pheno_cog_PC_n101.txt'), sep=" ")

cogpc_cnp = pd.read_csv(os.path.join(path_repo, 'data/CNP/behaviour/pheno_cog_PC_n256.txt'), sep=" ")


# In[3]:
# load in covars and clean
if regress_covars == True:
  
    #HCPEP
    covars_hcpep = pd.read_csv(os.path.join(path_repo, 'data/HCPEP/behaviour/covars_age_sex_fd.txt'), sep=" ")
    covars_hcpep.sex[covars_hcpep.sex == 'M'] = 1
    covars_hcpep.sex[covars_hcpep.sex == 'F'] = 0
    covars_hcpep.interview_age = covars_hcpep.interview_age / 12
    covars_hcpep = covars_hcpep.drop(['src_subject_id'], axis=1) 
    
    #TCP
    covars_tcp = pd.read_csv(os.path.join(path_repo, 'data/TCP/behaviour/covars_age_sex_fd.txt'), sep=" ")
    covars_tcp.Gender[covars_tcp.Gender == 'M'] = 1
    covars_tcp.Gender[covars_tcp.Gender == 'F'] = 0
    covars_tcp = covars_tcp.drop(['SUBJECT_ID'], axis=1) 
    
    #CNP
    covars_cnp = pd.read_csv(os.path.join(path_repo, 'data/CNP/behaviour/covars_age_sex_fd.txt'), sep=" ")
    covars_cnp.gender[covars_cnp.gender == 'M'] = 1
    covars_cnp.gender[covars_cnp.gender== 'F'] = 0
    covars_cnp = covars_cnp.drop(['V1'], axis=1) 
    covars_list = [covars_hcpep, covars_tcp, covars_cnp]

# In[4]:
# Match and check fc and cog data subjects and ordering

hcp_ep_subjlist = pd.read_csv(os.path.join(path_repo, 'data/HCPEP/behaviour/fmri_subj_id_order_n163.txt'), sep=" ")
hcp_ep_inclu_subj_index = hcp_ep_subjlist["x"].isin(cogpc_hcpep["src_subject_id"])
fc_data_hcpep = fc_data_hcpep[hcp_ep_inclu_subj_index,:]

# remove id and group cols. 
cogpc_hcpep = cogpc_hcpep.drop(['src_subject_id'], axis=1) 
cogpc_hcpep = cogpc_hcpep[['Cog_PC1']]
cogpc_tcp = cogpc_tcp.drop(['SUBJECT_ID', 'Group'], axis=1) 
cogpc_cnp = cogpc_cnp.drop(['V1'], axis=1) 

#convert to np array
cogpc_hcpep = cogpc_hcpep.to_numpy()
cogpc_tcp = cogpc_tcp.to_numpy()
cogpc_cnp = cogpc_cnp.to_numpy()

#store cog and fc and covars data into lists 
fc_list = [fc_data_hcpep , fc_data_tcp , fc_data_cnp]
cog_list = [cogpc_hcpep, cogpc_tcp , cogpc_cnp]

names = ['hcpep', 'tcp', 'cnp']

# In[4]:
# Model options
splits=100 #set up number of splits for outer loop
regress_covars = regress_covars


# In[5]:
#compute mm model with crossvalidation for datasets indicated by 'ind' OR  #for ind in range(len(fc_list)):    
    
kf_pearson_results = []
kf_best_param_results = []
 
x_input = fc_list[ind]
y_input = cog_list[ind]
 
#empty array to store predictions
pred_phenotypes = np.zeros((y_input.shape))
corr = np.zeros((y_input.shape[1],splits))
r2 = np.zeros((y_input.shape[1],splits))
best_param = np.zeros((y_input.shape[1],splits))


#z-score to match MM model input. 
fc_list = [zscore(fc_data_hcpep) , zscore(fc_data_tcp) , zscore(fc_data_cnp)]
    
#if covars are being regressed then cbind x/y and covars
if regress_covars == True:
        y_input = np.c_[y_input, covars_list[ind]]


for k in range(splits):
     x_train, x_test, y_train, y_test = train_test_split(x_input, y_input, train_size=0.7, random_state=k)
     
     if regress_covars == True:
        cog_pc = y_train[:,0]
        cog_pc = cog_pc.reshape(-1,1)
        covars = y_train[:,1:4]  # all covars
        model = LinearRegression().fit(covars,cog_pc)
        prediction = model.predict(covars)
        y_train = (cog_pc - prediction)
        betas = model.coef_
        y_test = y_test[:,0] - (model.intercept_ + y_test[:,1]*betas[0,0] +  y_test[:,2]*betas[0,1] +  y_test[:,3]*betas[0,2])
        y_test=y_test.reshape(-1,1)
        
     for s in range(y_train.shape[1]):
         y_test_pred, _ , best_param[s,k] = stacking(x_train, x_test, y_train[:,s],  splits=5)
         corr[s,k] = pearsonr(y_test_pred, y_test)[0]
         r2[s,k] = r2_score(y_test, y_test_pred)
              

     print(ind, k)

     kf_pearson_results = corr
     kf_COD_results = r2
     kf_best_param_results = best_param

varnames = ["PC1"]

kf_pearson_results = pd.concat([pd.DataFrame(varnames), pd.DataFrame(kf_pearson_results)], axis=1)
kf_COD_results = pd.concat([pd.DataFrame(varnames), pd.DataFrame(kf_COD_results)], axis=1)
kf_best_param_results = pd.concat([pd.DataFrame(varnames), pd.DataFrame(kf_best_param_results)], axis=1)

output_path = os.path.join(path_repo, 'output/accuracy/KRR/')

typename = ''
if regress_covars == True:
    typename='_ASFd' #(AgeSexFramewisedispacement)
    
kf_pearson_results.to_csv(str(output_path  + names[ind] + '_' + 'pearsonr_cogPC' + typename + '.txt'), 
                           sep=' ',
                           index=False,
                           header=False)

kf_COD_results.to_csv(str(output_path + names[ind] + '_' + 'COD_cogPC' + typename  + '.txt'), 
                           sep=' ',
                           index=False,
                           header=False)

kf_best_param_results.to_csv(str(output_path  + names[ind] + '_' + 'best_param_cogPC' + typename  + '.txt'), 
                           sep=' ',
                           index=False,
                           header=False)
  
