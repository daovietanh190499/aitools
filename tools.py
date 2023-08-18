import torch
from models import MattingNetwork, BiSeNet
from torchvision.transforms import ToTensor, Normalize
from PIL import Image
import cv2
import numpy as np
import math
from facenet_pytorch import MTCNN
import os
import PIL.Image
from manga_ocr import MangaOcr
from scipy.ndimage.filters import gaussian_filter, median_filter, maximum_filter, minimum_filter
from skimage import img_as_float

mocr = MangaOcr()

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

# def white_balance(image):
#     # Convert image to Lab color space
#     lab_image = cv2.cvtColor(cv2.cvtColor(np.array(image).astype('uint8'), cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2LAB)
#     # Split the Lab image into L, a, and b channels
#     l_channel, a_channel, b_channel = cv2.split(lab_image)
#     # Calculate mean and standard deviation of the a and b channels
#     a_mean, a_std = cv2.meanStdDev(a_channel)
#     b_mean, b_std = cv2.meanStdDev(b_channel)
#     # Adjust the a and b channels using mean values
#     a_channel = cv2.subtract(a_channel, a_mean[0])
#     b_channel = cv2.subtract(b_channel, b_mean[0])
#     # Merge the adjusted channels back to Lab image
#     lab_image = cv2.merge((l_channel, a_channel, b_channel))
#     # Convert Lab image back to BGR color space
#     result_image = cv2.cvtColor(lab_image, cv2.COLOR_LAB2BGR)
#     return result_image

def compress_image(image):
    img_cv = cv2.cvtColor(np.array(image).astype('uint8'), cv2.COLOR_RGB2BGR)
    return img_cv

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
    img_cv = rotate_image(img_cv, -angle)
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

    out = model_parsing(src_parse)[0]
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
    image = cv2.cvtColor(np.array(image).astype('uint8'), cv2.COLOR_RGB2BGR)
    image = np.pad(image, ((int(image.shape[0]/4), int(image.shape[0]/4)), (int(image.shape[1]/4), int(image.shape[1]/4)), (0,0)), 'edge')
    mask, all_box, face_box = parsing(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
    result_imgs = []
    img_spec = [[30, 40, 2.5, 25], [40, 60, 5, 30]]
    for isp in img_spec:
        width, height, toptop, face_length = isp[0], isp[1], isp[2], isp[3]
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
        result_imgs.append(result_img)
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

    for blk in blk_list:
        x1, y1, x2, y2 = blk.xyxy
        blk.text = mocr(Image.fromarray(img[y1:y2, x1:x2]))

    torch.cuda.empty_cache()

    texts = [' '.join(blk.text) for blk in blk_list]

    kernel = np.ones((9,9), np.uint8)
    mask_refined = cv2.dilate(mask_refined, kernel, iterations=2)
    img_inpainted =  dispatch_inpainting(True, False, use_cuda, img, mask_refined, 2048)
    torch.cuda.empty_cache()
    
    texts_translated = []
    texts_translated_2 = []
    if len(texts) > 0:
        texts_translated = trans.translate('\n'.join(texts), dest = 'en').text.split("\n")
        texts_translated_2 = trans.translate('\n'.join(texts_translated), dest = lang).text.split("\n")

    return img_inpainted, texts, texts_translated, texts_translated_2, str_bboxes
        
def sub(img, lang='vi'):
    img = cv2.cvtColor(np.array(img).astype('uint8'), cv2.COLOR_RGB2BGR)
    res =  infer(img, lang)
    return res

from models import load_superesolution_model, dispatch_superesolution

upsampler = load_superesolution_model('/home/coder/aitools/models/weights/RealESRGAN_x4plus.pth')

def superesolution(img): 
    img = cv2.cvtColor(np.array(img).astype('uint8'), cv2.COLOR_RGB2BGR)
    output = dispatch_superesolution(upsampler, img)
    return output

def xrayenhance(image):
    radius = 13
    amount = 1
    image = img_as_float(image) # ensuring float values for computations
    blurred_image = gaussian_filter(image, sigma=radius)
    mask = image - blurred_image # keep the edges created by the filter
    sharpened_image = image + mask * amount
    sharpened_image = np.clip(sharpened_image, float(0), float(1)) # Interval [0.0, 1.0]
    sharpened_image = (sharpened_image*255).astype(np.uint8) # Interval [0,255]
    return sharpened_image

class_names = []
with open("models/object_detection/classes.txt", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]

net = cv2.dnn.readNet("models/weights/yolov4-tiny.weights", "models/weights/yolov4-tiny.cfg")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)

CONFIDENCE_THRESHOLD = 0.2
NMS_THRESHOLD = 0.4
COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]

def object_detection(image):
    img = np.array(image).astype('uint8')
    classes, scores, boxes = model.detect(img, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    for (classid, score, box) in zip(classes, scores, boxes):
        color = COLORS[int(classid) % len(COLORS)]
        label = "%s : %f" % (class_names[classid], score)
        cv2.rectangle(img, box, color, 2)
        cv2.putText(img, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img
