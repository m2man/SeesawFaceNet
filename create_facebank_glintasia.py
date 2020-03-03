import os
from shutil import copyfile

Facebank_path = 'data/facebank_glintasia/'
Data_path = 'data/faces_glintasia/imgs/'

list_id = [f.path for f in os.scandir(Data_path) if f.is_dir()]

for subject_id in list_id:
    name_id = subject_id.split('/')
    name_id = name_id[-1]
    os.mkdir(Facebank_path+name_id)
    list_img = os.listdir(subject_id)
    for i in range(2):
        copyfile(subject_id+'/'+list_img[i], Facebank_path+name_id+'/'+list_img[i])
