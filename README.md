# OmniGo Face Recognition using SeesawFaceNet
### NOTE
- This repo is only for running (there is no training code in this repo)
- Images in facebank are sensitive
- Algorithms: Ultraface (Face Detection) + MTCNN-ONet (Face Alignment) + DW-SeesawFaceNetv2 (Face Recognition)
- All algorithms are combined in *Seesaw_Recognise* class in **seesaw.py**

## How To Use
### Create New Facebank
1. If you want to create entire new database, run **create_facebank.py**. Currently the database includes only 5 IDs storing in **facebank** folder
2. All images of IDs you want to recognise shoule be store in **facebank** folder (noted in **create_facebank.py**)

### Update Current Facebank
1. If you want to update the database, e.g. add new IDs, then run **update_facebank.py** (Please read the note in the file carefully). 
2. **After that, move all IDs subfolder in new_IDs folder to facebank folder**

### Evaluation on Facebank
1. Finally testing the recognisation on unseen images of trained IDs (stored in **test_recognise_facebank** folder)
2. If you add new ID to **test_recognise_facebank** folder (but not updating database), the predicted ID should be "Unknown"

### Recognise on face-cropped images
If you want to recognise on face-cropped images (No need to run Face detection), try the following snipset

```
from PIL import Image
from utils import convert_pil_rgb2bgr
from ultraface import Ultraface_detect
seesaw_model = Seesaw_Recognise(pretrained_path='pretrained_model/DW_SeesawFaceNetv2.pth', save_facebank_path='facebank/', device='cpu') # device='cuda:0' if GPU available
                                
img = Image.open('path/to/image.jpg') # Should be RGB

# Alignment
aligned_img,_ = seesaw_model.alignment_model.align(img)]

# Convert to BGR
aligned_img_bgr = convert_pil_rgb2bgr(aligned_img_bgr) # Convert to BGR (recognise step only work on BGR format)

# Recognition
name, distance = seesaw_model.infer([img_bgr]) # Input is a list, then need to put in [ ... ]
predicted_name = name[0] # 1st and the only element in the output list
predicted_distance = distance[0].numpy() # 1st and the only element in the output list
```

### Recognition on general images and show the prediction on images
Under maintaining