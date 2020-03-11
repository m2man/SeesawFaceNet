# ======= NOTE ====== #
# Run this file to create entire new facebank database
# All ID images should be organised as following structure
# facebank/
#   + ID_1/
#       - ID_1_1.jpg (or png, ...)
#       - ID_1_2.jpg
#   + ID_2/
#       - ID_2_1.jpg
# 1 ID can have 1 or many images (average of all images)
# IMPORTANT: Each image should only contain 1 face in the image
# ================== #

from seesaw import Seesaw_Recognise

facebank_folders = 'facebank/' # directory to the folder containing IDs folders
save_facebank_dir = facebank_folders # directory to save the embedded vector of IDs --> usually same as facebank_folders
run_detect = True # Set True if the ID images are not face-cropped images, False if it already face-cropped image
tta = True # Set True if want to flip images and calculate average vector of original and flipped images

seesaw_model = Seesaw_Recognise(pretrained_path='pretrained_model/DW_SeesawFaceNetv2.pth',
                                save_facebank_path='', # this is to load current facebank --> but if want to create entire new facebank, leave it ''
                                device='cpu')

temp1, temp2 = seesaw_model.create_facebank(facebank_path = facebank_folders, 
                                            save_facebank_path = save_facebank_dir, 
                                            run_detect=run_detect, tta=tta)

# Now there should be 2 extra file in save_facebank_dir (facebanks.pth and names.npy)