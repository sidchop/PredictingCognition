make_custom_legend <- function(breaks=c(0,50,100), 
                               hcl_palette = "Reds",
                               reverse = FALSE, 
                               title = NULL,
                               labels = breaks,
                               height=4, 
                               width=6,
                               direction = "horizontal",
                               fontsize=10) {
  library(ComplexHeatmap)
  library(circlize)
  col_fun= colorRamp2(breaks =breaks,
                      hcl_palette = hcl_palette, 
                      reverse = reverse,  space='RGB')
  color_bar = Legend(col_fun=col_fun, 
                     title = title, 
                     at=breaks, 
                     labels=labels,
                     labels_gp = gpar(col = "black",fontsize = fontsize),
                     grid_height = unit(height, "mm"), 
                     grid_width = unit(width, "mm"),
                     direction = direction) 
  png("temp.png")
  grid.newpage()
  draw(color_bar)
  dev.off()
  img <- magick::image_read("temp.png")
  img <-  magick::image_trim(img)
  file.remove("temp.png")
  return(img)
}





