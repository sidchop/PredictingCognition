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

#change DNN model here (see https://github.com/ThomasYeoLab/Meta_matching_models)
path_v1 = os.path.join(path_repo, 'Meta_matching_models/v1.0')
path_model_weight = os.path.join(path_v1, 'CBIG_ukbb_dnn_Zscore.pkl_torch') 
sys.path.append(path_v1)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr, zscore
from CBIG_model_pytorch import stacking, covariance_rowwise, multi_task_dataset
from sklearn.metrics import explained_variance_score, r2_score
from torch.utils.data import DataLoader

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = torch.load(path_model_weight,   map_location=torch.device('cpu') )
net.to(device)
net.train(False)
seed = 42
random.seed(seed)
np.random.seed(seed)
batch_size = 16
gpu = 0
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)


# Assign args
ind=int(sys.argv[1]) # study sample 0=HCPEP; 1 = TCP ;2 = CNP
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
#Compute MM model with crossvalidation for dataset set by 'ind' 0=HCPEP; 1 = TCP ;2 = CNP (or for ind in range(len(fc_list)):    )
splits=100 #set up number of splits for outer loop


kf_pearson_results = []
kf_COD_results = []
kf_VarExp_results = []
kf_Haufe_results = []
kf_Haufe_results2 = []
kf_best_param_results = []

x_input = fc_list[ind]
y_input = cog_list[ind]



#empty array to store predictions
pred_phenotypes = np.zeros((y_input.shape))
corr = np.zeros((y_input.shape[1],splits))
best_param = np.zeros((y_input.shape[1],splits))
r2 = np.zeros((y_input.shape[1],splits))
var_exp = np.zeros((y_input.shape[1],splits))
cov = np.zeros((x_input.shape[1], y_input.shape[1]))
cov2 = np.zeros((67, y_input.shape[1])) #DNN MM features (67)

#zscore fc data for GSR-DNN, 
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

    y_dummy = np.zeros(y_train.shape)
    dset_train = multi_task_dataset(x_train, y_dummy, True)
    trainLoader = DataLoader(dset_train,batch_size=batch_size,shuffle=False,num_workers=0)
    
    y_dummy = np.zeros(y_test.shape)
    dset_test = multi_task_dataset(x_test, y_dummy, True)
    testLoader = DataLoader(dset_test,batch_size=batch_size,shuffle=False,num_workers=0)

    # Here we apply the DNN trained on 67 UK Biobank phenotypes to predict the 67 phenotypes on `x_train` and `x_test`. We will get the predicted 67 phenotypes on both  training subjects and test subjects.

    y_train_pred = np.zeros((0, 67))
    for (x, _) in trainLoader:
        x= x.to(device)
        outputs = net(x)
        y_train_pred = np.concatenate((y_train_pred, outputs.data.cpu().numpy()), axis=0)
        
    y_test_pred = np.zeros((0, 67))
    for (x, _) in testLoader:
        x= x.to(device)
        outputs = net(x)
        y_test_pred = np.concatenate((y_test_pred, outputs.data.cpu().numpy()), axis=0)
    
    # Stacking with KRR
    # Perform stacking with `y_train_pred`, `y_test_pred`, `y_train`, where we use the prediction of training subjects `y_train_pred` (input) and real data `y_train` (output) to train the stacking model, then we applied the model to `y_test_pred` to get final prediction of cogntion score on test subjects.

    y_test_final =  np.zeros(((y_test.shape[0]), y_test.shape[1]))
    y_train_haufe = np.zeros(((y_train.shape[0]), y_train.shape[1]))
    
    for s in range(y_train.shape[1]):
        y_test_final[:,s], _ , best_param[s,k] = stacking(y_train_pred, y_test_pred, y_train[:,s],  splits=5)
        y_train_haufe[:,s], _, _ = stacking(y_train_pred, y_train_pred, y_train[:,s], splits=5)                                                                        
    
    #  Compute H-tranformed feature weights for:
    #   brain features/edges
    cov = covariance_rowwise(x_train, y_train_haufe) + np.squeeze(cov)
    #   67 MM phenotypes from UKB
    cov2 = covariance_rowwise(y_train_pred, y_train_haufe) + np.squeeze(cov2)
    
    # Compute performance metrics
    for i in range(y_test_final.shape[1]):
        corr[i,k] = pearsonr(y_test[:, i], y_test_final[:, i])[0]
        r2[i,k] = r2_score(y_test[:, i], y_test_final[:, i])
        var_exp[i,k] = explained_variance_score(y_test[:, i], y_test_final[:, i])
       
    print(ind, k)
    
kf_pearson_results = corr
kf_COD_results = r2
kf_VarExp_results = var_exp
kf_Haufe_results = cov/splits
kf_Haufe_results2 = cov2/splits
kf_best_param_results = best_param

varnames = ["PC1"]

kf_pearson_results = pd.concat([pd.DataFrame(varnames), pd.DataFrame(kf_pearson_results)], axis=1)
kf_COD_results = pd.concat([pd.DataFrame(varnames), pd.DataFrame(kf_COD_results)], axis=1)
kf_VarExp_results = pd.concat([pd.DataFrame(varnames), pd.DataFrame(kf_VarExp_results)], axis=1)
kf_Haufe_results = pd.DataFrame(kf_Haufe_results, columns=varnames)
kf_Haufe_results2 = pd.DataFrame(kf_Haufe_results2, columns=varnames)
kf_best_param_results = pd.concat([pd.DataFrame(varnames), pd.DataFrame(kf_best_param_results)], axis=1)

output_path = os.path.join(path_repo, 'output/accuracy/MM/')

typename = ''
if regress_covars == True:
    typename='_ASFd' #(AgeSexFramewisedispacement)

    
kf_pearson_results.to_csv(str(output_path  + names[ind] + '_' + 'pearsonr_cogPC' + typename  + '.txt'), 
                           sep=' ',
                           index=False,
                           header=False)

kf_COD_results.to_csv(str(output_path + names[ind] + '_' + 'COD_cogPC' + typename  + '.txt'), 
                           sep=' ',
                           index=False,
                           header=False)

kf_VarExp_results.to_csv(str(output_path  + names[ind] + '_' + 'VarExp_cogPC' + typename  + '.txt'), 
                           sep=' ',
                           index=False,
                           header=False)

kf_Haufe_results.to_csv(str(output_path  + names[ind] + '_' + 'haufe_cogPC' + typename  + '.txt'), 
                           sep=' ',
                           index=False,
                           header=False)
kf_Haufe_results2.to_csv(str(output_path  + names[ind] + '_' + 'haufe67pheno_cogPC' + typename  + '.txt'), 
                           sep=' ',
                           index=False,
                           header=False)
kf_best_param_results.to_csv(str(output_path  + names[ind] + '_' + 'best_param_cogPC' + typename + '.txt'), 
                           sep=' ',
                           index=False,
                           header=False)
