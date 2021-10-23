import cv2
import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
from models.models import create_model
from options.test_options import TestOptions
from util.face_detect_crop_single import Face_detect_crop
from util.reverse2original import reverse2wholeimage
import os
from util.norm import SpecificNorm
from models.parsing_model import BiSeNet
import warnings
warnings.filterwarnings('ignore')

transformer_Arcface = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def _totensor(array):
    tensor = torch.from_numpy(array)
    img = tensor.transpose(0, 1).transpose(0, 2).contiguous()
    return img.float().div(255)

if __name__ == '__main__':
    opt = TestOptions().parse()
    crop_size = 224

    model = create_model(opt)
    model.eval()

    app = Face_detect_crop(name='antelope', root='./weights')
    app.prepare(ctx_id= 0, det_thresh=0.6, det_size=(640,640))

    with torch.no_grad():
        
        # source image
        pic_a = opt.source
        img_a_whole = cv2.imread(pic_a)
        img_a_align_crop, _ = app.get(img_a_whole,crop_size)
        img_a_align_crop_pil = Image.fromarray(cv2.cvtColor(img_a_align_crop[0],cv2.COLOR_BGR2RGB)) 
        img_a = transformer_Arcface(img_a_align_crop_pil)
        img_id = img_a.view(-1, img_a.shape[0], img_a.shape[1], img_a.shape[2]).cpu()

        # target image
        pic_b = opt.target
        img_b_whole = cv2.imread(pic_b)
        img_b_align_crop_list, b_mat_list = app.get(img_b_whole,crop_size)
        
        # latent facial id
        latent_id = F.normalize(model.netArc(F.interpolate(img_id, scale_factor=0.5)), p=2, dim=1)

        # face swap
        b_align_crop_tensor = _totensor(cv2.cvtColor(img_b_align_crop_list[0],cv2.COLOR_BGR2RGB))[None,...].cpu()
        swap_result = model.netG.forward(b_align_crop_tensor, latent_id)[0]


        # net for what?
        net = BiSeNet(n_classes=19).cpu()
        net.load_state_dict(torch.load(os.path.join('./weights', '79999_iter.pth')))
        net.eval()

        # ?
        reverse2wholeimage([b_align_crop_tensor], [swap_result], b_mat_list, crop_size, img_b_whole, net, SpecificNorm(), use_mask=True)
        
        
        
        