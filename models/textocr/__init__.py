from typing import Tuple, List, Dict, Union
import numpy as np
import cv2

from ..textblockdetector.textblock import TextBlock

from typing import Dict
class ModuleParamParser:

    setup_params: Dict = None

    def __init__(self, **setup_params) -> None:
        if setup_params:
            self.setup_params = setup_params

    def updateParam(self, param_key: str, param_content):
        if isinstance(self.setup_params[param_key], str):
            self.setup_params[param_key] = param_content
        else:
            param_dict = self.setup_params[param_key]
            if param_dict['type'] == 'selector':
                param_dict['select'] = param_content
import torch
DEFAULT_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class OCRBase(ModuleParamParser):

    def __init__(self, **setup_params) -> None:
        super().__init__(**setup_params)
        self.name = ''
        self.setup_ocr()

    def setup_ocr(self):
        raise NotImplementedError

    def run_ocr(self, img: np.ndarray, blk_list: List[TextBlock] = None) -> Union[List[TextBlock], str]:
        if blk_list is None:
            return self.ocr_img(img)
        elif isinstance(blk_list, TextBlock):
            blk_list = [blk_list]
        return self.ocr_blk_list(blk_list)

    def ocr_blk_list(self, img: np.ndarray, blk_list: List[TextBlock]):
        raise NotImplementedError

    def ocr_img(self, img: np.ndarray) -> str:
        raise NotImplementedError

from .mit48px_ctc import OCR48pxCTC
OCR48PXMODEL: OCR48pxCTC = None
OCR48PXMODEL_PATH = r'models/weights/mit48pxctc_ocr.ckpt'

def load_48px_model(model_path, device, chunk_size=16) -> OCR48pxCTC:
    model = OCR48pxCTC(model_path, device, max_chunk_size=chunk_size)
    return model

class OCRMIT48pxCTC(OCRBase):
    setup_params = {
        'chunk_size': {
            'type': 'selector',
            'options': [
                8,
                16,
                24,
                32
            ],
            'select': 16
        },
        'device': {
            'type': 'selector',
            'options': [
                'cpu',
                'cuda'
            ],
            'select': DEFAULT_DEVICE
        },
        'description': 'mit48px_ctc'
    }
    device = DEFAULT_DEVICE
    chunk_size = 16

    def setup_ocr(self):
        
        global OCR48PXMODEL
        self.device = self.setup_params['device']['select']
        self.chunk_size = int(self.setup_params['chunk_size']['select'])
        if OCR48PXMODEL is None:
            self.model = OCR48PXMODEL = \
                load_48px_model(OCR48PXMODEL_PATH, self.device, self.chunk_size)
        else:
            self.model = OCR48PXMODEL
            self.model.to(self.device)
            self.model.max_chunk_size = self.chunk_size

    def ocr_img(self, img: np.ndarray) -> str:
        return self.model.ocr_img(img)

    def ocr_blk_list(self, img: np.ndarray, blk_list: List[TextBlock]):
        return self.model(img, blk_list)

    def updateParam(self, param_key: str, param_content):
        super().updateParam(param_key, param_content)
        device = self.setup_params['device']['select']
        chunk_size = int(self.setup_params['chunk_size']['select'])
        if self.device != device:
            self.model.to(device)
        self.chunk_size = chunk_size
        self.model.max_chunk_size = chunk_size