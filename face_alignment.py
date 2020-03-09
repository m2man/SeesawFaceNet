import os
os.chdir('/Users/duynguyen/DuyNguyen/Gitkraken/SeesawFaceNet')
import cv2
from PIL import Image
from mtcnn import MTCNN_Alignment
import numpy as np

mtcnn_alignment = MTCNN_Alignment()
print('mtcnn_alignment loaded')

folder_path = '/Users/duynguyen/DuyNguyen/Gitkraken/Pytorch_Retinaface/crop_face_from_video/duy'
save_path = 'aligned_imgs'
list_imgs = os.listdir(folder_path)

for img_name in list_imgs:
    img_path = folder_path + '/' + img_name
    image = Image.open(img_path)
    face = mtcnn_alignment.align(image)
    cv2.imwrite(save_path+'/'+img_name[:-4]+'_aligned.png', np.asarray(face))