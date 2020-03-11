# OmniGo Face Recognition using SeesawFaceNet
### NOTE
- This repo is only for running (there is no training code in this repo)
- Images in facebank are sensitive
- Algorithms: Ultraface (Face Detection) + MTCNN-ONet (Face Alignment) + DW-SeesawFaceNetv2 (Face Recognition)
- All algorithms is combined in *Seesaw_Recognise* class in **seesaw.py**

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