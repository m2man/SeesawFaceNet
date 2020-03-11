# OmniGo Face Recognition using SeesawFaceNet
### NOTE
- This repo is only for running (there is no training code in this repo)
- Images in facebank are sensitive
- Algorithms: Ultraface (Face Detection) + MTCNN-ONet (Face Alignment) + DW-SeesawFaceNetv2 (Face Recognition)
- Input of Face Detection is cv2 image 
    - Careful since default format of cv2 is BGR channel, need to convert to RGB by using ```cv2.cvtColor(img, cv2.COLOR_BGR2RGB)```
- Input of Face Alignment is PIL image 
    - Need to convert cv2 to PIL, but simple by using ```pil_img = Image.fromarray(cv2_img)```
- Input of Face Detection is PIL image 
    - input should be in **BGR format**
    - using ```convert_pil_rgb2bgr``` function in **utils.py** to convert PIL RGB to PIL BGR
- All algorithms are combined in *Seesaw_Recognise* class in **seesaw.py**
- **Again, careful with the input format of each stage**

## How To Use
### Create New Facebank
1. If you want to create entire new database, run **create_facebank.py**. Currently the database includes only 5 IDs storing in **facebank** folder
2. All images of IDs you want to recognise shoule be store in **facebank** folder (noted in **create_facebank.py**). An example ID image is below
<p align="center">
<img src="https://user-images.githubusercontent.com/15571804/76435038-69e07b00-63ae-11ea-940d-c7da5643e06c.jpg" width="250" height="350"/>
</p>

### Update Current Facebank
1. If you want to update the database, e.g. add new IDs, then run **update_facebank.py** (Please read the note in the file carefully). 
2. **After that, move all IDs subfolder in new_IDs folder to facebank folder**

### Evaluation on Facebank
1. Finally testing the recognisation on unseen images of trained IDs (stored in **test_recognise_facebank** folder) by running **test_recognise_ID_folder.py**
2. If you add new ID to **test_recognise_facebank** folder (but not updating database), the predicted ID should be "Unknown"
3. Note that since all images in **test_recognise_facebank** folder are currently the full format (not face-cropped format), we need to run face detection on them as written in the file

### Recognise on face-cropped images
If you want to recognise on face-cropped images (No need to run Face detection), try the following snipset

```
from PIL import Image
from utils import convert_pil_rgb2bgr
from seesaw import Seesaw_Recognise

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

### Recognise on general images and show the prediction on images
1. Put your general images in **general_images** folder **then run test_recognise_general_image.py**
2. Output will be exported to **result_general_images** folder
Some results are depicted as follow (can see full in the  above-mentioned folder)

<p align="center">
<img src="https://user-images.githubusercontent.com/15571804/76436517-3ef72680-63b0-11ea-9b55-b8401c8e12c1.jpg" width="450" height="340"/>
</p>

<p align="center">
<img src="https://user-images.githubusercontent.com/15571804/76436735-7d8ce100-63b0-11ea-9fd0-3e5c2c780301.jpg" width="450" height="340"/>
</p>

<p align="center">
<img src="https://user-images.githubusercontent.com/15571804/76436814-9a291900-63b0-11ea-932b-2b02c1926944.jpg" width="350" height="410"/>
</p>