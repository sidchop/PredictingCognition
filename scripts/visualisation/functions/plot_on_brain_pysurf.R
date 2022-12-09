plot_on_brain_pysurf <- function(plot_vector = NULL,
                                 temppath=NULL,
                                 min=NULL,
                                 max=NULL,
                                 colourscale='Reds', 
                                 jpeg = FALSE,
                                 clean_files = TRUE,
                                 surf = "inflated",
                                 no_subcortex = F,
                                 medial_wall_val = -1,
                                 diverging=F, 
                                 add.cb = F) {
  #setwd("/Users/sidchopra/Dropbox/Sid/python_files/metamatching/")
  #rhfile = '/Users/sidchopra/Dropbox/Sid/R_files/STAGES_difussion/output/figures/temp/temp_degree_rh.txt'  # remember to add a zero to top
  #lhfile =  '/Users/sidchopra/Dropbox/Sid/R_files/STAGES_difussion/output/figures/temp/temp_degree_lh.txt'
  #temppath = '~/Dropbox/Sid/R_files/STAGES_difussion/output/figures/temp/'
  #min = 0
  #max = 100
  #colourscale = "viridis"
  ###Options above
  
  #trouble shooting
  # make sure x11/xquartz is closed
  library(cowplot)
  library(ggplot2)
  library(reticulate)
  library(magick)
  library(RColorBrewer)
  library(rstudioapi)
  
  if(is.null(min)) {
    min <- min(plot_vector)
  }
  
  if(is.null(max)) {
    max <- max(plot_vector)
  }
  
  #write out degree vector for cortex
  plot_vector <- c(plot_vector)
  if(length(plot_vector)!=420) {
    stop("Plot vector should be 420 long (419 rois, brain stem repeated)")
  }
  
  #if(is.null(temppath)) {
  #  temppath <- here::here()
  #}
  
  
  #check if colourscale is a pal or a vector of colors
  if(length(colourscale)!=1) {
    colourscale <- c("#D3D3D3",colourscale) 
    colourscale_rgba <- t(col2rgb(colourscale))
    lh_colorvec_rgba <-  colourscale_rgba[1:201,]
    rh_colorvec_rgba <- colourscale_rgba[c(1,202:401),]
    sub_plotvec <- colourscale[402:421]
    
    #add grey for medial wall
    write.table(lh_colorvec_rgba, paste0(temppath, "temp_colour_lh.txt"), quote = F, row.names = F, col.names = F)
    write.table(rh_colorvec_rgba, paste0(temppath, "temp_colour_rh.txt"),quote = F, row.names = F, col.names = F)
    colourscale <- "Reds"
  }
  
  #split plot vector into lh_cortex,rh_cortex and subcortex
  lh_plotvec <- plot_vector[1:200]
  rh_plotvec <- plot_vector[201:400]
  sub_plotvec <- plot_vector[401:420]
  
  #add  -1 to cortical files with for medial wall 
  message(paste0("Adding ",medial_wall_val," to start of cortical lh and rh vectors for medial wall"))
  lh_plotvec <- c(medial_wall_val,lh_plotvec) 
  rh_plotvec <- c(medial_wall_val,rh_plotvec) 
  write.table(lh_plotvec, paste0(temppath, "temp_degree_lh.txt"), quote = F, row.names = F, col.names = F)
  write.table(rh_plotvec, paste0(temppath, "temp_degree_rh.txt"),quote = F, row.names = F, col.names = F)
  
  pyactivate <- "source activate /opt/anaconda3/envs/pysurfer \n"
  
  if(diverging==F) {
    pyexecute <- paste0("python /Users/sidchopra/Dropbox/Sid/python_files/PredictingCognition/scripts/visualisation/functions/make_pysurf.py -r ",
                        paste0(temppath, "temp_degree_rh.txt"),
                        " -l ", paste0(temppath, "temp_degree_lh.txt"),
                        " -m ", min,
                        " -a ", max,
                        " -c ", colourscale, 
                        " -s ", surf, 
                        " -k ", paste0(temppath, "temp_colour_lh.txt"),
                        " -g ", paste0(temppath, "temp_colour_rh.txt"), "\n")
  }
  
  if(diverging==T) {
    pyexecute <- paste0("python /Users/sidchopra/Dropbox/Sid/python_files/PredictingCognition/scripts/visualisation/functions/make_pysurf_diverging.py -r ",
                        paste0(temppath, "temp_degree_rh.txt"),
                        " -l ", paste0(temppath, "temp_degree_lh.txt"),
                        " -m ", min,
                        " -a ", max,
                        " -c ", colourscale, 
                        " -s ", surf, 
                        " -k ", paste0(temppath, "temp_colour_lh.txt"),
                        " -g ", paste0(temppath, "temp_colour_rh.txt"), "\n")
  }
  myTerm <- terminalCreate()
  
  terminalSend(myTerm, pyactivate)
  
  terminalSend(myTerm, pyexecute)
  
   repeat{
    Sys.sleep(1)
    if(rstudioapi::terminalBusy(myTerm) == FALSE){
      print("Surfaces made. ")
      rstudioapi::terminalKill(myTerm)
      break
    }
  }
  
  #trim and remove background
  
  list_c <- list.files(path = temppath, 
                       pattern = 'h_', full.names = T) #get list of file names you want
  
  for (x in list_c) {
    pic  <- image_read(x)
    #tpic <- image_transparent(pic, 'white')
    tpic_c <- image_trim(pic)
    image_write(tpic_c, path = x, format = "png") # new file is t_$file_name
  }
  
  if (no_subcortex == F) {
    ### Addin subcortex
    #use pyvista to make a png with subcortical surface mesh
    #source_python("/Users/sidchopra/Dropbox/Sid/R_files/STAGES_difussion/scripts/functions/get_subcortex_mesh_tian2.py")
    #get_subcortex_mesh_tian2(numvec = sub_plotvec, min=min,max=max,colourmap=colourscale,outputfolder=temppath)
    
    write.table(sub_plotvec, paste0(temppath, "sub_degree.txt"), quote = F, row.names = F, col.names = F)
    
    pyactivate <- "source activate  /Users/sidchopra/opt/anaconda3/envs/pyvista \n"
    
    pyexecute <- paste0("python /Users/sidchopra/Dropbox/Sid/python_files/PredictingCognition/scripts/visualisation/functions/get_subcortex_mesh_aseg.py -n ",
                        paste0(temppath, "sub_degree.txt"),
                        " -m ", min,
                        " -a ", max,
                        " -c ", colourscale, 
                        " -o ", temppath, " \n")
    
    myTerm <- terminalCreate()
    Sys.sleep(2)
    terminalSend(myTerm, pyactivate)
    Sys.sleep(2)
    terminalSend(myTerm, pyexecute)
    Sys.sleep(2)
    repeat{
      Sys.sleep(1)
      if(rstudioapi::terminalBusy(myTerm) == FALSE){
        print("Subcortex made.")
        rstudioapi::terminalKill(myTerm)
        break
      }
    }
    
    
    #Trim , flip and remove background from subcortex pngs
    list_s <- list.files(path = temppath,
                         pattern = 'sub_temp*', full.names = T) #get list of file names you want
    
    
    
    #make bg transpatent
    for (x in 1:length(list_s)) {
      pic  <- image_read(list_s[x])
      tpic_c  <- image_transparent(pic , 'white')
      tpic_c <- image_crop(tpic_c, "810x700+200")
      tpic_c <- image_trim(tpic_c)
      image_write(tpic_c, path = list_s[x], format = "png") # new file is t_$file_name
    }
    
    
    #add subcortex to medial images
    lm <- image_read(list_c[2])
    ls <- image_read(list_s[1]) #backwards on purpose, because the images have been fliped
    
    
    ls <- image_background(ls,"none")
    image_write(image_rotate(ls,-16), path = paste0(temppath, "rot_temp.png"))
    ls <- image_read( paste0(temppath, "rot_temp.png"))
    white <- image_blank(1350, 550, "white")
    lm <- image_composite(white, lm, offset = "+550")
    lm_s <- image_composite(lm, image_scale(ls,'500'), offset = "-180")
    lm_s <- image_trim(lm_s)
    
    rm <- image_read(list_c[4])
    rs <- image_read(list_s[2]) #backwards on purpose, because the images have been fliped
    
    rs <- image_background(rs,"none")
    image_write(image_rotate(rs,16), path = paste0(temppath, "rot_temp.png"))
    rs <- image_read( paste0(temppath, "rot_temp.png"))
    white <- image_blank(1350, 550, "white")
    rm <- image_composite(white, rm)
    rm_s <- image_composite(rm, image_scale(rs,'500'), offset = "+680")
    rm_s <- image_trim(rm_s)
    
    #combine 
    ll <-  image_trim(image_read(list_c[1]))
    rl <- image_trim(image_read(list_c[3]))
    
    rm_lm <- image_append(c(lm_s, rm_s))
    white <-  image_blank(image_info(rm_lm)$width, 
                          image_info(rm_lm)$height*2, "white")
    
    rm_lm <- image_composite(white,  rm_lm, offset = "+0+528")
    
    rl_ll <- image_append(c(ll, rl))
    
    all <- image_composite(rm_lm, rl_ll, offset = "+380")
    
  } else {
    list_c <- list.files(path = temppath,
                         pattern = 'surf_*', full.names = T)
    lm <- image_read(list_c[2])
    rm <- image_read(list_c[4])
    ll <-  image_read(list_c[1])
    rl <- image_read(list_c[3])
    
    
    rl_ll <- image_append(c(rl, ll))
    rm_lm <- image_append(c(rm, lm))
    all <- image_append(c(rl_ll, rm_lm), stack = T)
  }
  
  #addcolour bar
  if ( add.cb == T) {
  mat = matrix(rnorm(400,1,1), ncol=20)
  #
  ##change colour bar here
  ##grad = viridis::viridis(n=100)
  ##grad = RColorBrewer::brewer.pal(9,"coolwarm")
  ##grad = pals::coolwarm(n=100)
  grad =  pals::brewer.reds(100)
  #
  
  Cairo::CairoPNG(paste0(temppath, "temp_cb.png"), height = 5, width = 5, res = 300)
  print({lattice::levelplot(mat, col.regions=grad, colorkey = list(at=seq(min,max,length.out=100)))})
  dev.off()
  
  cb <- image_read(paste0(temppath,"temp_cb.png"))
  cb <- image_chop(cb, "1300x150")
  cb <- image_trim(cb)
  cb <- image_transparent(cb, 'white')
  
  final <- image_composite(image_scale(all, '1500'),image_scale(cb,'60'), offset = "+1410+0")
  } else {
    final  <- all
  }
  
  if(jpeg==T){ image_write(final, paste0(temppath, "final_output.jpeg"), quality = 100) }
  
  if(clean_files==T) {
    remove <- list.files(temppath, pattern = "*_lat*", full.names = T)
    file.remove(remove)
    remove <- list.files(temppath, pattern = "*_med*", full.names = T)
    file.remove(remove)
    remove <- list.files(temppath, pattern = "*surf*", full.names = T)
    file.remove(remove)
    remove <- list.files(temppath, pattern = "*degree*", full.names = T)
    file.remove(remove)
    remove <- list.files(temppath, pattern = "*temp*", full.names = T)
    file.remove(remove)
  }
  #
  return(final)
}
