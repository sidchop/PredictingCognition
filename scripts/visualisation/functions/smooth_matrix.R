#functions computes the mean within cluster of a matrix based on row and col member ship
#return.full.matrix == T returns r x r mat, where n is the number of ROIS, and 
# return.full.matrix == F returns n x n mat, where n is the number of networks/clusters

smooth_matrix <- function(mat=NULL, 
                          row.cluster=NULL, 
                          col.cluster=NULL, 
                          return.full.matrix = T,
                          diagFill=TRUE) {
  mask <- matrix(nrow = length(row.cluster), ncol = length(col.cluster))
  for(i in 1:length(row.cluster)) {
    for(j in i:length(col.cluster)) {
      mask[i,j] <- paste(row.cluster[i],col.cluster[j])
    }
  }
  mask[upper.tri(mask, diag = T)] <- as.numeric(as.factor(mask[upper.tri(mask, diag = T)]))
  if (diagFill == TRUE) {
  diag(mask) <- NA
  }
  mask <- apply(mask, 1, as.numeric)
  mask <- as.matrix(Matrix::forceSymmetric(mask, uplo = "L"))
  numlist<- na.omit(unique(c(mask)))
  sumary_matrix <- matrix(nrow = length(unique(row.cluster)), ncol = length(unique(col.cluster)))
  sumary_matrix <- sumary_matrix[upper.tri(sumary_matrix, diag = T)]

  for(s in 1:length(numlist)) {
    temp_mask <- mask
  #  temp_mask[temp_mask==numlist[s]] <- numlist[s]
    temp_mask[temp_mask!=numlist[s]] <- NA
    temp_mask[!is.na(temp_mask)] <- 1
    temp_mat <- mat*temp_mask
    mean <- mean(temp_mat, na.rm = T)
    if(is.na(mean)) {mean <- 0}
    mask[mask==numlist[s]] <- mean
    sumary_matrix[s] <- mean
  }
 if (diagFill == TRUE) { # fill diagonal with adjacent values for visualization purposes. 
  diag_fill <- c(mask[1,2],diag(mask[-1,]))
  diag(mask) <- diag_fill
  warning('Diagnoal filled with adjacent values. Set diagFill=False to diasble this behavior')
 }
  
  if(return.full.matrix == F){
    sumary_matrix2 <- matrix(nrow = length(unique(row.cluster)), ncol = length(unique(col.cluster)))
    sumary_matrix2[lower.tri(sumary_matrix2, diag = T)] <- sumary_matrix
    sumary_matrix2 <- as.matrix(Matrix::forceSymmetric(sumary_matrix2, uplo = "L"))
    return(sumary_matrix2)
  } else {
    return(mask)
  } 
}


