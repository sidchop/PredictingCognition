source /Users/sidchopra/opt/anaconda3/bin/activate
conda init zsh
conda activate metamatching

path_repo=/Users/sidchopra/Dropbox/Sid/python_files/PredictingCognition/
# arg1 = sample (0=HCPEP; 1=TCP; 2=CNP)
# arg2 = regress covars (True/False)

# Meta-matching Models
# Primary models

for s in {0..2} ; do python ${path_repo}/scripts/analysis/accuracy/compute_MM_cognitionPC.py ${s} False ; done

# Covar models
for s in {0..2} ; do python ${path_repo}/scripts/analysis/accuracy/compute_MM_cognitionPC.py ${s} True  ; done

# KRR Models
# Primary models

for s in {0..2} ; do python ${path_repo}/scripts/analysis/accuracy/compute_KRR_cognitionPC.py ${s} False ; done

# Covar models
for s in {0..2} ; do python ${path_repo}/scripts/analysis/accuracy/compute_KRR_cognitionPC.py ${s} True  ; done
