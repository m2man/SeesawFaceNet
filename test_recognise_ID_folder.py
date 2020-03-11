
# ===== NOTE ===== #
# Run this to test the recognise model on facebank folder in facebank/
# ================ #
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

ID_name = 'MD'
new_ID = False if ID_name in seesaw_model.names else True # checking new ID or already trained ID
if new_ID:
    correct_ID = 'Unknown'
else:
    correct_ID = ID_name

img_folder = 'test_recognise_facebank/'+ID_name
img_folder = Path(img_folder)
names = []
distances = []
list_files = []

for file in img_folder.iterdir():
    if not file.is_file() or file.suffix == '':
        continue
    else:
        try:
            print(f"Processing file {str(file)}")
            # detect face --> if the input is not face image
            image = cv2.imread(str(file))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Convert to RGB --> dont know why the default is BGR
            faces, boxes = seesaw_model.detect_model.detect_face(image)
            img = faces[0] # should only has 1 face in an image
            # align
            img,_ = seesaw_model.alignment_model.align(img)
            # Convert to BGR (IMPORTANT)
            img = convert_pil_rgb2bgr(img)
            # recognise
            name, distance = seesaw_model.infer([img])
            names.append(name[0])
            distances.append(distance[0].numpy())
            list_files.append(str(file))
        except:
            print('Error in ' + str(file))

mis_recognise = [] # list of files that wrong recognises
accurate = 0
for idx, name in enumerate(names):
    if name != correct_ID:
        mis_recognise.append(list_files[idx])
    else:
        accurate += 1
accuracy = accurate / len(names) * 100

print("===== RESULT SUMMARY =====")
print(f"Correct ID that the model should predicted: {correct_ID}\nCorrect Prediction / Total Prediction: {accurate} / {len(names)}\nAccuracy: {accuracy}%")
print("List of Error Recognise Files:")
print(mis_recognise)
