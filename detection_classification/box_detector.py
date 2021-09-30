import utils
import numpy as np
import torch
import torch.nn as nn
import torchvision.ops as ops

def get_detections(img_batch, model, score_threshold=0.5, iou_threshold=0.5):
    model.eval()
    with torch.no_grad():
        classifications, bboxes, landmarks = model(img_batch)
        batch_size = classifications.shape[0]
        picked_boxes = []
        picked_landmarks = []
        picked_scores = []
        
        for i in range(batch_size):
            classification = torch.exp(classifications[i,:,:])
            bbox = bboxes[i,:,:]
            landmark = landmarks[i,:,:]

            scores, argmax = torch.max(classification, dim=1)
            argmax_indice = argmax==0
            scores_indice = scores > score_threshold
            positive_indices = argmax_indice & scores_indice
            scores = scores[positive_indices]

            if scores.shape[0] == 0:
                picked_boxes.append(None)
                picked_landmarks.append(None)
                picked_scores.append(None)
                continue

            bbox = bbox[positive_indices]
            landmark = landmark[positive_indices]

            keep = ops.boxes.nms(bbox, scores, iou_threshold)
            keep_boxes = bbox[keep]
            keep_landmarks = landmark[keep]
            keep_scores = scores[keep]
            keep_scores.unsqueeze_(1)
            picked_boxes.append(keep_boxes)
            picked_landmarks.append(keep_landmarks)
            picked_scores.append(keep_scores)
        
        return picked_boxes, picked_landmarks, picked_scores
