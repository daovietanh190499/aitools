import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from datetime import datetime
import torch
import cv2
from PIL import Image
from io import BytesIO
import base64
import traceback

from tools import white_balance, smooth_face, rotate_by_eye, matting, crop_image, sub
from code_offline import code_scan

app = FastAPI()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

async def read_image(request):
    form = await request.form()
    file = await form["file"].read()
    image = Image.open(BytesIO(file))
    return image

def img2str(result):
    _, buffer = cv2.imencode('.jpg', result)
    img_str = base64.b64encode(buffer)
    return img_str

@app.post('/white_balance')
async def white_balance_(request: Request):
    image = await read_image(request)
    try:
        result = white_balance(image)
        img_str = img2str(result)
        return {"message": "success", "images": [img_str]}
    except Exception as e:
        print(traceback.format_exc())
        return JSONResponse(content={"message": "server failure"}, status_code=500)

@app.post('/smooth_face')
async def smooth_face_(request: Request):
    image = await read_image(request)
    try:
        result = smooth_face(image)
        img_str = img2str(result)
        return {"message": "success", "images": [img_str]}
    except Exception as e:
        print(traceback.format_exc())
        return JSONResponse(content={"message": "server failure"}, status_code=500)

@app.post('/rotate_by_eye')
async def rotate_by_eye_(request: Request):
    image = await read_image(request)
    try:
        result = rotate_by_eye(image)
        img_str = img2str(result)
        return {"message": "success", "images": [img_str]}
    except Exception as e:
        print(traceback.format_exc())
        return JSONResponse(content={"message": "server failure"}, status_code=500)

@app.post('/matting')
async def matting_(request: Request):
    image = await read_image(request)
    try:
        result = matting(image)
        img_str = img2str(result)
        return {"message": "success", "images": [img_str]}
    except Exception as e:
        print(traceback.format_exc())
        return JSONResponse(content={"message": "server failure"}, status_code=500)

@app.post('/crop_image')
async def crop_image_(request: Request):
    image = await read_image(request)
    try:
        result = crop_image(image)
        new_result = []
        for r in result:
            img_str = img2str(r)
            new_result.append(img_str)
        return {"message": "success", "images": new_result}
    except Exception as e:
        print(traceback.format_exc())
        return JSONResponse(content={"message": "server failure"}, status_code=500)
    
@app.post('/sub')
async def sub_(request: Request):
    form = await request.form()
    image = await read_image(request)
    param = [image, 'vi']
    keys = ['image', 'lang']
    for i, key in enumerate(keys[2:]):
        param[i] = form[key] if key in form else param[i]
    
    try:
        img, texts, trans, trans2, bbs = sub(*tuple(param))
        _, buffer = cv2.imencode('.jpg', img)
        img_str = base64.b64encode(buffer)
        sub_text = ""
        for text, tran, tran2, bb in zip(texts, trans, trans2, bbs):
            sub_text += bb + '\n' + text + '\n' + tran + '\n' + tran2 + '\n'
        return {"message": "success", "image": img_str, "sub_text": sub_text}
    except Exception as e:
        print(traceback.format_exc())
        return JSONResponse(content={"message": "server failure"}, status_code=500)

uvicorn.run(app, host='0.0.0.0', port=8000)