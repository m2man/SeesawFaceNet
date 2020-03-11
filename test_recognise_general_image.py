import os
os.chdir('/Users/duynguyen/DuyNguyen/Gitkraken/SeesawFaceNet')

import cv2
from seesaw import Seesaw_Recognise
import numpy as np
from pathlib import Path
from utils import convert_pil_rgb2bgr

seesaw_model = Seesaw_Recognise(pretrained_path='pretrained_model/DW_SeesawFaceNetv2.pth', 
                                save_facebank_path='facebank/', device='cpu')

folder = 'general_images/'
list_images_names = os.listdir(folder)
outputs = []
for img_name in list_images_names:
    print(f"Processing {img_name}")
    img_full_name = folder + img_name
    img = cv2.imread(img_full_name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert to RGB --> dont know why the default is BGR
    a = seesaw_model.infer_general_image(img, plot_result=True, tta=False)
    outputs.append(a)

# save a image using extension 
for idx, out in enumerate(outputs):
    out = out.save(f"test_{idx}.jpg") 