import sys
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import skimage
from skimage import io
import cv2
import torchvision
import box_detector 
import torchvision_model
import load_model
import classifier

if __name__ == "__main__":
    cur_time = time.time()

    # Load RetinaFace Model
    return_layers = {'layer2':1,'layer3':2,'layer4':3}
    RetinaFace = torchvision_model.create_retinaface(return_layers)
    retina_dict = RetinaFace.state_dict()
    pre_state_dict = torch.load("models/model.pt")
    pretrained_dict = {k[7:]: v for k, v in pre_state_dict.items() if k[7:] in retina_dict}
    RetinaFace.load_state_dict(pretrained_dict)
    RetinaFace = RetinaFace.cpu()
    RetinaFace.eval()

    # Load image
    img = skimage.io.imread(sys.argv[1])
    img = torch.from_numpy(img)
    img = img.permute(2,0,1)
    input_img = img.unsqueeze(0).float().cpu()
    picked_boxes, picked_landmarks, picked_scores = box_detector.get_detections(input_img, RetinaFace, score_threshold=0.5, iou_threshold=0.3)
    np_img = img.cpu().permute(1,2,0).numpy()
    np_img.astype(int)
    img = cv2.cvtColor(np_img.astype(np.uint8),cv2.COLOR_BGR2RGB)

    # Classify result faces from RetinaFace
    cropped_images = []
    for j, boxes in enumerate(picked_boxes):
        if boxes is not None:
            for box, landmark, score in zip(boxes,picked_landmarks[j],picked_scores[j]):
                cropped_images.append(img[int(box[1]):int(box[3]),int(box[0]):int(box[2])])
    classified_labels = classifier.predict_age_gender_race(cropped_images)
    print("Time spent :", time.time() - cur_time)

    # 확인용 이미지 생성
    for j, boxes in enumerate(picked_boxes):
        if boxes is not None:
            for box, landmark, score, labels in zip(boxes,picked_landmarks[j],picked_scores[j],classified_labels):
                cropped_images.append(img[int(box[1]):int(box[3]),int(box[0]):int(box[2])])
                cv2.rectangle(img,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),(0,0,255),1)
                cv2.circle(img,(int(landmark[0]),int(landmark[1])),radius=1,color=(0,0,255),thickness=1)
                cv2.circle(img,(int(landmark[2]),int(landmark[3])),radius=1,color=(0,255,0),thickness=1)
                cv2.circle(img,(int(landmark[4]),int(landmark[5])),radius=1,color=(255,0,0),thickness=1)
                cv2.circle(img,(int(landmark[6]),int(landmark[7])),radius=1,color=(0,255,255),thickness=1)
                cv2.circle(img,(int(landmark[8]),int(landmark[9])),radius=1,color=(255,255,0),thickness=1)
                cv2.putText(img, text="/".join(map(str,labels)), org=(int(box[0]),int(box[1])), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.3,thickness=1, lineType=cv2.LINE_AA, color=(255, 255, 255))

    cv2.imwrite("out.jpg", img)
