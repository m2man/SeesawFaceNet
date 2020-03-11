# ======= NOTE ====== #
# Run this file to create update facebank database
# The pipeline: Read current facebank --> read new IDs --> update to new facebank
# All new ID images should be organised as following structure
# new_IDs/
#   + ID_1/
#       - ID_1_1.jpg (or png, ...)
#       - ID_1_2.jpg
#   + ID_2/
#       - ID_2_1.jpg
# 1 ID can have 1 or many images (average of all images)
# IMPORTANT: Each image should only contain 1 face in the image
# ================== #

from seesaw import Seesaw_Recognise

new_facebank_folders = 'new_IDs/' # directory to the folder containing new IDs folders
current_facebank_dir = 'facebank/' # where the current facebank.pth and names.npy located
save_facebank_dir = 'facebank/' # directory to save the embedded vector of IDs
run_detect = True # Set True if the ID images are not face-cropped images, False if it already face-cropped image
tta = True # Set True if want to flip images and calculate average vector of original and flipped images

seesaw_model = Seesaw_Recognise(pretrained_path='pretrained_model/DW_SeesawFaceNetv2.pth',
                                save_facebank_path=current_facebank_dir, # This is to load current facebank
                                device='cpu')

print('Updating facebank ... ')
t1, t2 = seesaw_model.update_facebank(new_facebank_path=new_facebank_folders,
                                      save_facebank_path=save_facebank_dir,
                                      run_detect=run_detect, tta=tta)
print('Finished ')
