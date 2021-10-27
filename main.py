import os
import sys
import glob
import fractions
from PIL import Image
import warnings
warnings.filterwarnings('ignore')
import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from models.models import create_model
from models.classifier import predict_age_gender_race
from models.parsing_model import BiSeNet
from util.face_detect_crop_multi import Face_detect_crop
from util.reverse2original import reverse2wholeimage
from util.norm import SpecificNorm


transformer_Arcface = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def _totensor(array):
    tensor = torch.from_numpy(array)
    img = tensor.transpose(0, 1).transpose(0, 2).contiguous()
    return img.float().div(255)


if __name__ == '__main__':
    MY_HOME_DIR = "."    
    crop_size = 224
    model = create_model(MY_HOME_DIR)
    model.eval()
    app = Face_detect_crop(name='weights', root=MY_HOME_DIR)
    app.prepare(ctx_id=0, det_thresh=0.6, det_size=(640,640))

    
    with torch.no_grad():
        # target image 로부터 여러 사람들의 얼굴을 인식
        target_image = cv2.imread(sys.argv[1])
        img_b_whole, b_mat_list = app.get(target_image, crop_size)
        
        
        # 웹을 통해 사용자 선택 받아오기
        user_click = [i for i in range(7)]
        img_b_selected = [img_b_whole[i] for i in user_click if i < len(img_b_whole)]
        people_no = len(img_b_selected)
        
        
        # 사용자가 선택한 얼굴들에 대해서 나이,성별 분류한 뒤 각각에 해당하는 GAN 이미지 복사해놓기
        target_id_nonorm_list = []
        cls_labels = predict_age_gender_race(MY_HOME_DIR, img_b_selected)
        for idx, cls in enumerate(cls_labels):
            print(cls)
            # labels = (1, 20, "Asian")
            # os.system(f"cp images/GAN/{"Male" if labels[0]==1 else "Female"}/{str(labels[1]).zfill(2)}")
            continue
        
        
        # 얼굴 수 만큼 해당 경로에 gan 이미지(source)들이 들어가있으므로, 각각에 대해 normalized latent_id 추출
        source_id_norm_list = []
        source_path = os.path.join(MY_HOME_DIR, 'images/tmp/SRC_*')
        source_images_path = sorted(glob.glob(source_path))
        for idx, source_image in enumerate(source_images_path):
            if idx >= people_no:
                break
            person = cv2.imread(source_image)
            person_align_crop, _ = app.get(person, crop_size, threshold=0.2)
            img_pil = Image.fromarray(cv2.cvtColor(person_align_crop[0],cv2.COLOR_BGR2RGB))
            img_trans = transformer_Arcface(img_pil)
            img_id = img_trans.view(-1, img_trans.shape[0], img_trans.shape[1], img_trans.shape[2]).cpu()
            latent_id = F.normalize(model.netArc(F.interpolate(img_id, scale_factor=0.5)), p=2, dim=1)
            source_id_norm_list.append(latent_id.clone())
            
        
        # 선택한 얼굴과 gan 얼굴을 합치고
        result = []
        matrix = []
        original = []
        img_b_tensor = [_totensor(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))[None,...].cpu() for img in img_b_selected]
        for idx in range(people_no):
            res = model(None, img_b_tensor[idx], source_id_norm_list[idx], None, True)[0]
            result.append(res)
            matrix.append(b_mat_list[idx])
            original.append(img_b_tensor[idx])
        
        
        # 합친 얼굴을 이미지에 적용시켜서 파일로 저장
        net = BiSeNet(n_classes=19).cpu()
        net.load_state_dict(torch.load(os.path.join(MY_HOME_DIR, 'weights/bisenet.pth'), map_location=torch.device('cpu')))
        net.eval()
        final_img = reverse2wholeimage(original, result, matrix, crop_size, target_image, net, SpecificNorm())
        cv2.imwrite(os.path.join(MY_HOME_DIR, "images/output.jpg"), final_img)
