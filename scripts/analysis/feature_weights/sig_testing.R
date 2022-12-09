#compute significant FDR corrected PNF blocks for each data set (fix link using Here::here())

#Observerd mean PNF values by network block
atlas_labs <- readxl::read_xlsx("/Users/sidchopra/Dropbox/Sid/python_files/metamatching/data/atlas/atlas_labels.xlsx", sheet = 1)

library(data.table)
hcp_ep <- fread("/Users/sidchopra/Dropbox/Sid/python_files/metamatching/output/MM_stacking/hcpep_haufe_cogPC_GSR.txt", sep = " ")
pc <- fread("/Users/sidchopra/Dropbox/Sid/python_files/metamatching/output/MM_stacking/tcp_haufe_cogPC_GSR.txt", sep = " ")
ucla <- fread("/Users/sidchopra/Dropbox/Sid/python_files/metamatching/output/MM_stacking/cnp_haufe_cogPC_GSR.txt", sep = " ")

obs_pnf_list <- cbind(hcp_ep, pc, ucla*-1)
colname <- c("hcp-ep", "pc", "ucla")
n=18 #number of networks

obs_pnf_mat <- matrix(nrow = n*(n+1)/2, ncol = length(colname))
obs_roi_mat_pos <-  obs_roi_mat_neg <- matrix(nrow = 419, ncol = length(colname))

source("~/Dropbox/Sid/R_files/functions/smooth_matrix.R")

for (s in 1:ncol(obs_pnf_list)) {
  mat <- matrix(nrow = 419, ncol = 419)
  mat[upper.tri(mat)] <- as.matrix(obs_pnf_list[, ..s])
  mat <- as.matrix(Matrix::forceSymmetric(mat, uplo = "U"))
  diag(mat) <- 0
  mat <- mat[atlas_labs$reorder, atlas_labs$reorder]
  mat_pos <-  mat_neg <- mat
  mat_pos[mat_pos < 0] <- 0
  mat_neg[mat_neg > 0] <- 0
  obs_roi_mat_pos[,s] <- rowMeans(mat_pos)
  obs_roi_mat_neg[,s] <- rowMeans(mat_neg)
  smooth_mat <- smooth_matrix(mat = mat, 
                              row.cluster = atlas_labs$Network,
                              col.cluster = atlas_labs$Network,
                              return.full.matrix = F)
  obs_pnf_mat[,s] <- smooth_mat[upper.tri(smooth_mat, diag = T)]
}



#For each dataset, convert each null pnf vec into mat, compute network block means and the row means (for ROI)
extraxt_null_pnf <- function(x) {
  mat <- matrix(nrow = 419, ncol = 419)
  mat[upper.tri(mat)] <- as.matrix(x)
  mat <- as.matrix(Matrix::forceSymmetric(mat, uplo = "U"))
  diag(mat) <- 0
  mat <- mat[atlas_labs$reorder, atlas_labs$reorder]
  mat_pos <-  mat_neg <- mat
  mat_pos[mat_pos < 0] <- 0
  mat_neg[mat_neg > 0] <- 0
  row_means_pos <- rowMeans(mat_pos)
  row_means_neg <- rowMeans(mat_neg)
  smooth_mat_pos <- smooth_matrix(mat = mat_pos, 
                                  row.cluster = atlas_labs$Network,
                                  col.cluster = atlas_labs$Network,
                                  return.full.matrix = F)
  smooth_mat_neg <- smooth_matrix(mat = mat_neg, 
                                  row.cluster = atlas_labs$Network,
                                  col.cluster = atlas_labs$Network,
                                  return.full.matrix = F)
  smooth_mat <- smooth_matrix(mat = mat, 
                              row.cluster = atlas_labs$Network,
                              col.cluster = atlas_labs$Network,
                              return.full.matrix = F)
  smooth_vec <- smooth_mat[upper.tri(smooth_mat, diag = T)]
  row_vec <- cbind(row_means_pos,row_means_neg)
  smooth_vec_pos_neg <- cbind(smooth_mat_pos[upper.tri(smooth_mat_pos, diag = T)], smooth_mat_neg[upper.tri(smooth_mat_neg, diag = T)])
  return(list( smooth_vec, row_vec, smooth_vec_pos_neg))
}

library(parallel)
library(pbapply)
cl <- makeCluster(8)
clusterExport(cl, c("extraxt_null_pnf", "atlas_labs", "smooth_matrix"))

#~40 mins per data set (convert to matrix and extract mean network values and rowmeans for network and roi nulls)
hcp_null_pnf <- fread("/Users/sidchopra/Dropbox/Sid/python_files/metamatching/output/MM_stacking/nulls/hcpep_haufe_cogPC_GSR_nulls.txt",
                      nThread = 8, select = c(1:2000))
hcp_null_pnf_out <- pbapply(hcp_null_pnf[,1:2000], 2, extraxt_null_pnf, simplify = T, cl = cl)
rm(hcp_null_pnf)
gc()

pc_null_pnf <-  fread("/Users/sidchopra/Dropbox/Sid/python_files/metamatching/output/MM_stacking/nulls/tcp_haufe_cogPC_GSR_nulls.txt", 
                      nThread = 8)
pc_null_pnf_out <- pbapply(pc_null_pnf[,1:2000], 2, extraxt_null_pnf, simplify = T, cl = cl)
rm(pc_null_pnf)
gc()

ucla_null_pnf <-  fread("/Users/sidchopra/Dropbox/Sid/python_files/metamatching/output/MM_stacking/nulls/cnp_haufe_cogPC_GSR_nulls.txt", 
                        nThread = 8, sep = " ")

ucla_null_pnf <- ucla_null_pnf*-1
ucla_null_pnf_out <- pbapply(ucla_null_pnf[,1:2000], 2, extraxt_null_pnf, simplify = T, cl = cl)
rm(ucla_null_pnf)
gc()

stopCluster(cl)
null_pnf_out_list <- list(sapply(hcp_null_pnf_out, "[[", 1),sapply(pc_null_pnf_out, "[[", 1),sapply(ucla_null_pnf_out, "[[", 1))

null_pos_neg_roi_out_list <- list(sapply(hcp_null_pnf_out, "[[", 2),sapply(pc_null_pnf_out, "[[", 2),sapply(ucla_null_pnf_out, "[[", 2))

null_pnf_pos_neg_out <- list(sapply(hcp_null_pnf_out, "[[", 3),sapply(pc_null_pnf_out, "[[", 3),sapply(ucla_null_pnf_out, "[[", 2))

null_pnf_out_list_pos_list <- list(null_pnf_pos_neg_out[[1]][1:171,],null_pnf_pos_neg_out[[2]][1:171,],null_pnf_pos_neg_out[[3]][1:171,])
null_pnf_out_list_neg_list <- list(null_pnf_pos_neg_out[[1]][172:342,],null_pnf_pos_neg_out[[2]][172:342,],null_pnf_pos_neg_out[[3]][172:342,])
null_pnf_out_list_posneg <- list(null_pnf_out_list_pos_list,null_pnf_out_list_neg_list )

saveRDS(object = null_pnf_out_list, 
        file = "/Users/sidchopra/Dropbox/Sid/python_files/metamatching/output/MM_stacking/nulls/17Network_mean_PNF_nulls.RDS", compress = T)
saveRDS(object = null_pos_neg_roi_out_list, 
        file = "/Users/sidchopra/Dropbox/Sid/python_files/metamatching/output/MM_stacking/nulls/ROI_pos_neg_mean_PNF_nulls.RDS", compress = T)
saveRDS(object = null_pnf_out_list_posneg, 
        file = "/Users/sidchopra/Dropbox/Sid/python_files/metamatching/output/MM_stacking/nulls/17Network_mean_PNF_posneg_nulls.RDS", compress = T)


null_pnf_out_list  <- readRDS("/Users/sidchopra/Dropbox/Sid/python_files/metamatching/output/MM_stacking/nulls/17Network_mean_PNF_nulls.RDS")

## sig test network level features 

#get pvals for each network in each dataset
is.sig.twotail <- function(obs_val=NULL, null_vec=NULL) {
  obs_val <- as.numeric(unlist(obs_val))
  null_vec <- as.numeric(unlist(null_vec))
  return(sum(abs(obs_val) < abs(null_vec)) / length(null_vec))
}

#compute fdr corrected pvals
p_mat_fdr <- matrix(nrow=nrow(null_pnf_out_list[[1]]), ncol=length(null_pnf_out_list))
p_mat <- matrix(nrow=nrow(null_pnf_out_list[[1]]), ncol=length(null_pnf_out_list))
for (l in 1:length(null_pnf_out_list)) {
  for (p in 1:nrow(null_pnf_out_list[[1]])) {
    p_mat[p,l] <- is.sig.twotail(obs_val = round(obs_pnf_mat[p,l],3), 
                                 null_vec = round(null_pnf_out_list[[l]][p,],3))
  }
  p_mat_fdr[,l] <- p.adjust(p_mat[,l], method = "fdr")
}

#convert into binary matricies of sig values. 
fdr_corrected_pvals <- list()
for (f in 1:3) {
  mat <- matrix(0, nrow = 18, ncol = 18)
  mat[upper.tri(mat, diag = T)] <- p_mat_fdr[,f]
  fdr_corrected_pvals[[f]] <- as.matrix(Matrix::forceSymmetric(mat))
}
saveRDS(object = fdr_corrected_pvals, file = "~/Dropbox/Sid/python_files/metamatching/output/MM_stacking/nulls/17Network_mean_PNF_FDR_pvals.RDS")

pvals <- list()
for (f in 1:3) {
  mat <- matrix(0, nrow = 18, ncol = 18)
  mat[upper.tri(mat, diag = T)] <- p_mat[,f]
  pvals[[f]] <- as.matrix(Matrix::forceSymmetric(mat))
}
saveRDS(object = fdr_corrected_pvals, file = "~/Dropbox/Sid/python_files/metamatching/output/MM_stacking/nulls/17Network_mean_PNF_uncor_pvals.RDS")


### sig test ROI level features
null_pnf_roi_posneg  <- readRDS("/Users/sidchopra/Dropbox/Sid/python_files/metamatching/output/MM_stacking/nulls/ROI_pos_neg_mean_PNF_nulls.RDS")

null_pnf_roi_pos <- list(sapply(null_pnf_roi_posneg[[1]], "[[", 1),sapply(null_pnf_roi_posneg[[2]], "[[", 1),sapply(null_pnf_roi_posneg[[3]], "[[", 1))

null_pnf_roi_neg <- list(sapply(null_pnf_roi_posneg[[1]], "[[", 2),sapply(null_pnf_roi_posneg[[2]], "[[", 2),sapply(null_pnf_roi_posneg[[3]], "[[", 2))

null_pnf_roi_posneg <- list(null_pnf_roi_pos, null_pnf_roi_neg)

## sig test network level features 
#get pvals for each network in each dataset
is.sig.twotail <- function(obs_val=NULL, null_vec=NULL) {
  obs_val <- as.numeric(unlist(obs_val))
  null_vec <- as.numeric(unlist(null_vec))
  return(sum(abs(obs_val) < abs(null_vec)) / length(null_vec))
}

obs_roi_mat <- list(obs_roi_mat_pos, obs_roi_mat_neg)

#compute fdr corrected pvals pos and neg sep
p_mat_fdr <-  p_mat <-  list(matrix(nrow=nrow(null_pnf_roi_posneg[[1]][[1]]), ncol=length(null_pnf_roi_posneg[[1]])),
                             matrix(nrow=nrow(null_pnf_roi_posneg[[1]][[1]]), ncol=length(null_pnf_roi_posneg[[1]])))
for (s in 1:length(null_pnf_roi_posneg)) {
  for (l in 1:length(null_pnf_roi_posneg[[1]])) {
    for (p in 1:nrow(null_pnf_roi_posneg[[1]][[1]])) {
      p_mat[[s]][p,l] <- is.sig.twotail(obs_val = round(obs_roi_mat[[s]][p,l],3), 
                                        null_vec = round(null_pnf_roi_posneg[[s]][[l]][p,],3))
    }
    p_mat_fdr[[s]][,l] <- p.adjust(p_mat[[s]][,l], method = "fdr")
  }
}
saveRDS(object = p_mat_fdr, file = "~/Dropbox/Sid/python_files/metamatching/output/MM_stacking/nulls/ROI_pos_neg_mean_PNF_Pvals.RDS")
saveRDS(object = p_mat, file = "~/Dropbox/Sid/python_files/metamatching/output/MM_stacking/nulls/ROI_pos_neg_mean_PNF_uncor_Pvals.RDS")




#compute fdr corrected pvals pos and neg togeather

obs_roi_mat <- (obs_roi_mat_pos + obs_roi_mat_neg)/2
p_mat_fdr <-  p_mat <-  matrix(nrow=nrow(null_pnf_roi_posneg[[1]][[1]]), ncol=length(null_pnf_roi_posneg[[1]]))


for (l in 1:ncol(obs_roi_mat)) {
  for (p in 1:nrow(obs_roi_mat)) {
    p_mat[p,l] <- is.sig.twotail(obs_val = round(obs_roi_mat[p,l],3), 
                                      null_vec = round(rowMeans(cbind(as.numeric(null_pnf_roi_posneg[[1]][[l]][p,]),as.numeric(null_pnf_roi_posneg[[2]][[l]][p,]))),3) )
  }
  p_mat_fdr[,l] <- p.adjust(p_mat[,l], method = "fdr")
}


saveRDS(object = p_mat_fdr, file = "~/Dropbox/Sid/python_files/metamatching/output/MM_stacking/nulls/ROI_total_mean_PNF_Pvals.RDS")
saveRDS(object = p_mat, file = "~/Dropbox/Sid/python_files/metamatching/output/MM_stacking/nulls/ROI_total_mean_PNF_uncor_Pvals.RDS")




## NBS attempt ==== 
##OBS  max comp size
#source("~/Dropbox/Sid/R_files/functions/get_components.R")
#primary_thresh = 0.05
#max_comp_vec = NULL
#for (s in 1:ncol(obs_pnf_list)) {
#  mat <- matrix(nrow = 419, ncol = 419)
#  mat[upper.tri(mat)] <- scale(as.matrix(obs_pnf_list[, ..s]))
#  mat <- as.matrix(Matrix::forceSymmetric(mat, uplo = "U"))
#  diag(mat) <- 0
#  mat <- mat[atlas_labs$reorder, atlas_labs$reorder]
#  p_mat =   (1-pnorm(abs(mat)))*2
#  p_mat[p_mat>primary_thresh] <- 0
#  p_mat[p_mat!=0] <- 1
#  max_comp_vec[s] <- get_components(p_mat,return_max = T, return_comp = F)
#}
#
##null comps
#extraxt_null_comp <- function(x, primary_thresh=0.001) {
#  mat <- matrix(nrow = 419, ncol = 419)
#  mat[upper.tri(mat)] <- scale(as.matrix(x))
#  mat <- as.matrix(Matrix::forceSymmetric(mat, uplo = "U"))
#  diag(mat) <- 0
#  mat <- mat[atlas_labs$reorder, atlas_labs$reorder]
#  mat <- mat[atlas_labs$reorder, atlas_labs$reorder]
#  p_mat =   (1-pnorm(abs(mat)))*2
#  p_mat[p_mat>primary_thresh] <- 0
#  p_mat[p_mat!=0] <- 1
#  max_comp <- get_components(p_mat,return_max = T, return_comp = F)
#  return(max_comp)
#}
#
#library(parallel)
#library(pbapply)
#cl <- makeCluster(8)
#clusterExport(cl, c("extraxt_null_comp", "atlas_labs", "get_components"))
#
##~5 mins per data set (convert to matrix and extract max comp)
#pc_null_comp <- pbapply(pc_null_pnf, 2,
#                        extraxt_null_comp, 
#                        primary_thresh=0.05,
#                        simplify = T, cl = cl)
#stopCluster(cl)
#
#sum(max_comp_vec[2] < pc_null_comp)/2000

