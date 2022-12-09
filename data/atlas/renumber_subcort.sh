#make lh .nii for .vkt conversion
x=1
for s in 401 402 403 404 405 406 407 408 409 410 ; do \
	fslmaths  19_subcortex.nii.gz -thr $s -uthr $s -bin  -mul $x roi_${x}.nii.gz ; \
	echo $x ; (( x+=1 )) ; \
done

#erode brain stem and cerebellum for visualisation 
fslmaths roi_6.nii.gz -ero roi_6.nii.gz
#fslmaths roi_1.nii.gz -kernel box 3 -ero roi_1.nii.gz

fslmaths roi_1.nii.gz -add roi_2.nii.gz  -add roi_3.nii.gz  -add roi_4.nii.gz  \
-add roi_5.nii.gz  -add roi_6.nii.gz  -add roi_7.nii.gz  -add roi_8.nii.gz \
 -add roi_9.nii.gz  -add roi_10.nii.gz  lh_subcortex_renum.nii \

 rm roi_*

 #make lh .nii for .vkt conversion
x=1
for s in 411 412 413 414 415 416 417 418 419 406 ; do \
	fslmaths  19_subcortex.nii.gz -thr $s -uthr $s -bin  -mul $x roi_${x}.nii.gz ; \
	echo $x ; (( x+=1 )) ; \
done
fslmaths roi_10.nii.gz -ero roi_10.nii.gz
#fslmaths roi_1.nii.gz -kernel box 3 -ero roi_1.nii.gz

fslmaths roi_1.nii.gz -add roi_2.nii.gz  -add roi_3.nii.gz  -add roi_4.nii.gz  \
-add roi_5.nii.gz  -add roi_6.nii.gz  -add roi_7.nii.gz  -add roi_8.nii.gz \
 -add roi_9.nii.gz  -add roi_10.nii.gz  rh_subcortex_renum.nii \

 rm roi_*
