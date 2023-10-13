import sys
import requests
from tqdm import tqdm
from .config import *

# import pydevd_pycharm
# pydevd_pycharm.settrace('49.7.62.197', port=10090, stdoutToServer=True, stderrToServer=True)

sys.path.append(utils_path)

from .node import *

def urldownload_progressbar(url, file_path):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    progress_bar = tqdm(total=total_size, unit='B', unit_scale=True)
    with open(file_path, 'wb') as f:
        for chunk in response.iter_content(1024):
            if chunk:
                f.write(chunk)
                progress_bar.update(len(chunk))

    progress_bar.close()

print("Start Setting weights")
for url, filename in zip(urls, filenames):
    if os.path.exists(filename):
        continue
    print(f"Start Downloading: {url}")
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    urldownload_progressbar(url, filename)

NODE_CLASS_MAPPINGS = {
    "RetainFace": RetainFace,
    "FaceFusion": FaceFusion,
    "RatioMerge2Image": RatioMerge2Image,
    "MaskMerge2Image": MaskMerge2Image,
    "ReplaceBoxImg": ReplaceBoxImg,
    "ExpandMaskBox": ExpandMaskFaceWidth,
    "BoxCropImage": BoxCropImage,
    "ColorTransfer": ColorTransfer,
    "FaceSkin": FaceSkin,
    "MaskDilateErode": MaskDilateErode,
    "SkinRetouching": SkinRetouching,
    "PortraitEnhancement": PortraitEnhancement,
    "ResizeImage": ResizeImage,
    "GetImageInfo": GetImageInfo,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "RetainFace": "RetainFace",
    "FaceFusion": "FaceFusion",
    "RatioMerge2Image": "RatioMerge2Image",
    "MaskMerge2Image": "MaskMerge2Image",
    "ReplaceBoxImg": "ReplaceBoxImg",
    "ExpandMaskBox": "ExpandMaskBox",
    "BoxCropImage": "BoxCropImage",
    "ColorTransfer": "ColorTransfer",
    "FaceSkin": "FaceSkin",
    "MaskDilateErode": "MaskDilateErode",
    "SkinRetouching": "SkinRetouching",
    "PortraitEnhancement": "PortraitEnhancement",
    "ResizeImage": "ResizeImage",
    "GetImageInfo": "GetImageInfo",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
