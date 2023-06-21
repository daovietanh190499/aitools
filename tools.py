import torch
from models import MattingNetwork, BiSeNet
from torchvision.transforms import ToTensor, Normalize
from PIL import Image
import cv2
import numpy as np
import math
from facenet_pytorch import MTCNN
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_str = "cuda" if torch.cuda.is_available() else "cpu"

model_matting = MattingNetwork('mobilenetv3').to(device).eval()
model_matting.load_state_dict(torch.load('models/weights/rvm_mobilenetv3.pth', map_location=device))

model_parsing = BiSeNet(n_classes=19).to(device).eval()
model_parsing.load_state_dict(torch.load('models/weights/79999_iter.pth', map_location=device))

model_mtcnn = MTCNN(select_largest=True, device=device_str)

def white_balance(img):
    img_cv = cv2.cvtColor(np.array(img).astype('uint8'), cv2.COLOR_RGB2BGR)
    result = cv2.cvtColor(img_cv, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    return result

def smooth_face(img, size=20, sigma=5):
    img = cv2.cvtColor(np.array(img).astype('uint8'), cv2.COLOR_RGB2BGR)
    img = cv2.bilateralFilter(img,int(size),int(sigma),int(sigma))
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def rotate_by_eye(image):
    img_pil = image
    _, _, landmarks = model_mtcnn.detect(img_pil, landmarks=True)
    left_eye = landmarks[0][0]
    right_eye = landmarks[0][1]
    angle = math.atan2(abs(left_eye[1] - right_eye[1]), abs(left_eye[0] - right_eye[0]))*180/math.pi
    
    img_cv = cv2.cvtColor(np.array(image).astype('uint8'), cv2.COLOR_RGB2BGR)
    if is_white_balance:
        img_cv = white_balance(img_cv)
    img_cv = rotate_image(img_cv, -angle if is_rotate else 0)
    return img_cv

def matting(image):
    bgr = torch.tensor([.47, 1., .6]).view(3, 1, 1).to(device)
    bgr = torch.tensor([22/255, 148/255, 1.]).view(3, 1, 1).to(device)
    rec = [None] * 4                                       
    downsample_ratio = 0.25
    img = ToTensor()(image).to(device)
    src = img.unsqueeze(0)
    fgr, pha, *rec = model_matting(src, *rec, downsample_ratio) 
    com = fgr * pha + bgr * (1 - pha)
    com = com.squeeze(0)
    source_img = com.permute(1, 2, 0).detach().cpu().numpy()*255
    source_img = cv2.cvtColor(source_img.astype('uint8'), cv2.COLOR_RGB2BGR)
    return source_img

def parsing(image):
    img = ToTensor()(image).to(device)
    img_parse = Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(img)
    src_parse = img.unsqueeze(0)

    out = model_parsing(src)[0]
    parsing = out.squeeze(0).detach().cpu().numpy().argmax(0)
    
    contours, hierarchy = cv2.findContours((parsing > 0).astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    sorted_contours= sorted(contours, key=cv2.contourArea, reverse= True)
    largest_item= sorted_contours[0]
    all_box = cv2.boundingRect(sorted_contours[0])

    atts = ['bg','skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g', 'l_ear', 'r_ear', 'ear_r',
            'nose', 'mouth', 'u_lip', 'l_lip', 'neck', 'neck_l', 'cloth', 'hair', 'hat']
    colors = [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0]
    
    parsing_copy = parsing.copy()
    for i, color in enumerate(colors):
        parsing_color = (parsing == i)*parsing
        if color == 0:
            parsing = parsing - parsing_color
    parsing = parsing > 0
    mask = np.zeros((parsing_copy.shape[0], parsing_copy.shape[1], 3))
    mask_1d = ((parsing_copy == 1).astype('uint8') + (parsing_copy == 10).astype('uint8')) > 0
    mask[mask_1d] = np.ones((3,))

    contours, hierarchy = cv2.findContours(parsing.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    sorted_contours= sorted(contours, key=cv2.contourArea, reverse= True)
    largest_item= sorted_contours[0]
    face_box = cv2.boundingRect(sorted_contours[0])

    face_box = list(face_box)
    face_box[3] = face_box[1] - all_box[1] + face_box[3]
    face_box[1] = all_box[1]

    return mask, all_box, face_box

def crop_image(image):
    mask, all_box, face_box = parsing(image)
    image = cv2.cvtColor(np.array(image).astype('uint8'), cv2.COLOR_RGB2BGR)
    result_imgs = []
    img_spec = [[30, 40], [40, 60]]
    for isp in img_spec:
        width, height, toptop, face_length, num_img = isp[0], isp[1], isp[2], isp[3], isp[4]
        new_face_box = [0,0,0,0]
        
        upper_length = face_box[3]/(face_length/toptop)
        lower_length = face_box[3]/(face_length/(height - toptop - face_length))

        total_length = upper_length + face_box[3] + lower_length
        total_width = total_length*(width/height)

        new_face_box[0] = round(face_box[0] - (total_width - face_box[2])/2)
        new_face_box[1] = round(face_box[1] - upper_length)
        new_face_box[2] = round(total_width)
        new_face_box[3] = round(total_length)

        result_img = image[new_face_box[1]: new_face_box[1] + new_face_box[3], new_face_box[0]: new_face_box[0] + new_face_box[2]]
        result_img = result_img*255
        
        result_imgs.append((result_img, new_face_box))
    return result_imgs

from models import load_textdetector_model, dispatch_textdetector, dispatch_inpainting, load_inpainting_model, OCRMIT48pxCTC
import torch
import cv2
import numpy as np
from googletrans import Translator

trans = Translator(service_urls=[
      'translate.google.com'
])

use_cuda = torch.cuda.is_available()

print("Load Models")

setup_params = OCRMIT48pxCTC.setup_params
setup_params['device']['select'] = 'cuda' if torch.cuda.is_available() else 'cpu'
setup_params['chunk_size']['select'] = 16
ocr = OCRMIT48pxCTC(**setup_params)
load_textdetector_model(use_cuda)
load_inpainting_model(use_cuda, 'default')

def infer(img, lang):
    mask, mask_refined, blk_list = dispatch_textdetector(img, use_cuda)
    torch.cuda.empty_cache()

    str_bboxes = [' '.join([str(num) for num in blk.xyxy]) for blk in blk_list]

    ocr.ocr_blk_list(img, blk_list)
    torch.cuda.empty_cache()

    texts = [' '.join(blk.text) for blk in blk_list]

    kernel = np.ones((9,9), np.uint8)
    mask_refined = cv2.dilate(mask_refined, kernel, iterations=2)
    img_inpainted =  dispatch_inpainting(True, False, use_cuda, img, mask_refined, 2048)
    torch.cuda.empty_cache()
    
    if len(texts) > 0:
        texts_translated = trans.translate('\n'.join(texts), dest = 'en').text.split("\n")
        texts_translated_2 = trans.translate('\n'.join(texts_translated), dest = lang).text.split("\n")

    return img_inpainted, texts, texts_translated, texts_translated_2, str_bboxes
        
def sub(img, lang='vi'):
    img = cv2.cvtColor(np.array(img).astype('uint8'), cv2.COLOR_RGB2BGR)
    res =  infer(img, lang)
    return res
