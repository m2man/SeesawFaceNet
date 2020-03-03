import cv2
from PIL import Image
import argparse
from pathlib import Path
from multiprocessing import Process, Pipe,Value,Array
import torch
from config import get_config
from mtcnn import MTCNN
from Learner import face_learner
from utils import load_facebank, draw_box_name, prepare_facebank
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='for face verification')
    parser.add_argument("-s", "--save", help="whether save",action="store_true")
    parser.add_argument('-th','--threshold',help='threshold to decide identical faces',default=1.54, type=float)
    parser.add_argument("-u", "--update", help="whether perform update the facebank",action="store_true")
    parser.add_argument("-tta", "--tta", help="whether test time augmentation",action="store_true")
    parser.add_argument("-c", "--score", help="whether show the confidence score",action="store_true")
    parser.add_argument("-dirimg", "--dir_img", help="directory contains images", default='data_new/', type=str)
    args = parser.parse_args()

    conf = get_config(training=False)

    mtcnn = MTCNN()
    print('mtcnn loaded')
    
    learner = face_learner(conf, inference=True)
    learner.threshold = args.threshold
    if conf.device.type == 'cpu':
        learner.load_state(conf, 'DW-SeesawFaceNet_V2.pth', from_save_folder=False, model_only=True)
    else:
        learner.load_state(conf, 'DW-SeesawFaceNet_V2.pth', from_save_folder=False, model_only=True)
    learner.model.eval()
    print('learner loaded')
    
    if args.update:
        targets, names = prepare_facebank(conf, learner.model, mtcnn, tta = args.tta)
        print('facebank updated')
    else:
        targets, names = load_facebank(conf)
        print('facebank loaded')

    list_imgs = os.listdir(args.dir_img)
    error = 0
    total = len(list_imgs)

    for img_name in list_imgs:
        try:
            image = Image.open(args.dir_img + img_name)
            bboxes, faces = mtcnn.align_multi(image, conf.face_limit, conf.min_face_size)
        except ValueError:
            bboxes = []
        if len(bboxes) > 0:
            bboxes = bboxes[:,:-1] #shape:[10,4],only keep 10 highest possibiity faces
            bboxes = bboxes.astype(int)
            bboxes = bboxes + [-1,-1,1,1] # personal choice    
            results, score = learner.infer(conf, faces, targets, args.tta)
            for idx,bbox in enumerate(bboxes):
                if args.score:
                    image = draw_box_name(bbox, names[results[idx] + 1] + '_{:.2f}'.format(score[idx]), image)
                else:
                    image = draw_box_name(bbox, names[results[idx] + 1], image)
            cv2.imwrite(f"{args.dir_img}{img_name}_{names[results[idx] + 1]}.png",image)
        else:
            print('detect error')
            error += 1

    print(f"Total error {error}/{total}")

          