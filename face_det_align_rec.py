# This file is full pipeline: detect --> align --> recogise 
# (1st part is detect, 2nd part is both align and recognise)

import os
os.chdir('/Users/duynguyen/DuyNguyen/Gitkraken/SeesawFaceNet')
import cv2
from PIL import Image
from mtcnn import MTCNN_Alignment
from ultraface import Ultraface_detect
from seesaw import Seesaw_Recognise
import numpy as np
from pathlib import Path
from utils import convert_pil_rgb2bgr

seesaw_model = Seesaw_Recognise(pretrained_path='pretrained_model/DW_SeesawFaceNetv2.pth', 
                                save_facebank_path='facebank/', device='cpu')

img_folder = 'data_infer/ThuThuy'
img_folder = Path(img_folder)
names = []
distances = []

for file in img_folder.iterdir():
    if not file.is_file() or file.suffix == '':
        continue
    else:
        try:
            # detect face --> if the input is not face image
            image = cv2.imread(str(file))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Convert to RGB --> dont know why the default is BGR
            faces, boxes = seesaw_model.detect_model.detect_face(image)
            img = faces[0] # should only has 1 face in an image
            # align
            img,_ = seesaw_model.alignment_model.align(img)
            # Convert to BGR
            img = convert_pil_rgb2bgr(img)
            # recognise
            name, distance = seesaw_model.infer([img])
            names.append(name)
            distances.append(distance[0].numpy())
        except:
            print(str(file))

print(distances)
print(names)


'''
files = img_folder.iterdir()

file = next(files)
print(str(file))
image = cv2.imread(str(file))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Convert to RGB --> dont know why the default is BGR
faces, boxes = seesaw_model.detect_model.detect_face(image)
img = faces[0] # should only has 1 face in an image
# align
img,_ = seesaw_model.alignment_model.align(img)
# Convert to BGR
img = convert_pil_rgb2bgr(img)
name, distance = seesaw_model.infer([img])
'''

'''
img_path = 'data_infer/MD/duy_27.jpg'
image = cv2.imread(str(img_path))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Convert to RGB --> dont know why the default is BGR
faces, boxes = seesaw_model.detect_model.detect_face(image)
img = faces[0] # should only has 1 face in an image
img,_ = seesaw_model.alignment_model.align(img)
img = convert_pil_rgb2bgr(img)
name, distance = seesaw_model.infer([img])
print(name)
'''