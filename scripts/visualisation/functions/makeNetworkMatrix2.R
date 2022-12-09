#create network matrix2
makeNetworkMatrix2 <- function(matrix1,
                               matrix2,
                               min1=NULL,
                               max1=NULL,
                               min2=NULL,
                               max2=NULL,
                               pal2 = "Reds",
                               pal1 = "Blues",
                               cex=15,
                               title="",
                             colours = c("#9C755FFF", "#F28E2BFF", "#E15759FF", "#76B7B2FF", 
                                         "#59A14FFF",  "#EDC948FF", "#B07AA1FF", "#5575b3FF", 
                                         "#496499FF", "#4E79A7FF"),
                             layout=c(area(1,1, 5,5), area(1,2,5,5))
) {
  library(reshape2)
  library(scales)
  library(gtable)
  library(patchwork)
  

  
  # Get lower triangle of the correlation matrix
  get_upper_tri<-function(cormat){
    cormat[upper.tri(cormat)] <- NA
    return(cormat)
  }
  # Get upper triangle of the correlation matrix
  get_lower_tri <- function(cormat){
    cormat[lower.tri(cormat)]<- NA
    return(cormat)
  }

  
  lower_tri <- get_lower_tri(matrix1) 
  
  melted_cormat <- melt(lower_tri, na.rm = TRUE)
  melted_cormat[melted_cormat==0] <- NA #sent zero values as NA so they come out grey
  #head(melted_cormat)
  
  
  ggheatmap <- ggplot(data = melted_cormat, aes(x=Var2, y=Var1, fill=value)) + 
    geom_tile(color = "white", lwd=1.2)  +
    theme(axis.text.x = element_text(angle = 45, vjust = 1, 
                                     size = cex, hjust = 1, colour = colours), 
          axis.text.y = element_text(vjust = 1, 
                                     size = cex, hjust = 1, colour = colours),
          axis.title.x = element_blank(),
          axis.title.y = element_blank(),
          panel.border = element_blank(),
          axis.ticks = element_blank(),
          legend.text=element_text(size=14),
          legend.title = element_text(size=15),
          legend.position = c(1.25, 0.55),
          panel.grid.major = element_blank(), 
          panel.grid.minor = element_blank(),
          panel.background = element_rect(fill = "transparent",colour = NA),
          plot.background = element_rect(fill = "transparent",colour = NA)) +
    coord_fixed() + scale_y_discrete(position = "right") + 
    scale_fill_distiller(palette = pal1, direction = -1, na.value = "#eeeeee", limits=c(min1, max1))+
  #  scale_fill_gradientn(colours=pal,
  #                       limits=c(min1, max1), na.value  = "#eeeeee") + 
    
    labs(fill = "Negative PNF") + 
    guides(fill = guide_colourbar(nbin = 100))
    

  
  upper_tri <- get_upper_tri(matrix2) 
  melted_cormat <- melt(upper_tri, na.rm = TRUE) #set zero values as NA so they come out grey/"#eeeeee"
  melted_cormat[melted_cormat==0] <- NA 
  
  
  ggheatmap2 <- ggplot(data = melted_cormat, aes(x=Var2, y=Var1, fill=value)) + 
    geom_tile(color = "white", lwd = 1.2)  +
    theme(axis.text.x = element_text(angle = 45, vjust = 1, 
                                     size = cex, hjust = 0, colour = colours), 
          axis.text.y = element_text(vjust = 1, 
                                     size = cex, hjust = 1, colour = colours),
          axis.title.x = element_blank(),
          axis.title.y = element_blank(),
          panel.border = element_blank(),
          axis.ticks = element_blank(),
          legend.position = c(-0.25, 0.57),
          legend.text=element_text(size=14),
          legend.title = element_text(size=15),
          panel.grid.major = element_blank(), 
          panel.grid.minor = element_blank(),
          panel.background = element_rect(fill = "transparent",colour = NA),
          plot.background = element_rect(fill = "transparent",colour = NA)) +
    coord_fixed() + scale_y_discrete(position = "left") +
    scale_x_discrete(position = "top") +
    scale_fill_distiller(palette = pal2, direction = +1, na.value = "#eeeeee",  limits=c(min2, max2)) +
  #  scale_fill_gradientn(colours=pal,
  #                       limits=c(min2, max2), na.value = "#eeeeee") +
    labs(fill = "Positive PNF") +
guides(fill = guide_colourbar(nbin = 100))
  

  ggheatmap <- ggheatmap +
    theme(
      rect = element_rect(fill = "transparent") # all rectangles
    )

  ggheatmap2 <- ggheatmap2 +
    theme(
      rect = element_rect(fill = "transparent") # all rectangles
    )
  
  layout <- layout

  ggheatmap_fin <- ggheatmap2 +  ggheatmap +  plot_layout(design = layout)
  
  #Add title 
  ggheatmap_fin <- ggheatmap_fin + ggtitle(title)   
  
  return( ggheatmap_fin)
  
  
}

