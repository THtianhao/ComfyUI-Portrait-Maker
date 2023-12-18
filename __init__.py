import os, sys

main_path = os.path.dirname(__file__)
sys.path.append(main_path)

import portrait.install
import requests
from tqdm import tqdm
from portrait.nodes import *

# import pydevd_pycharm
# pydevd_pycharm.settrace('49.7.62.197', port=10090, stdoutToServer=True, stderrToServer=True)

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
    print(f"Start Downloading: {url} Download To {filename}")
    print(f"开始下载: {url} 下载到 {filename}")
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    urldownload_progressbar(url, filename)

NODE_CLASS_MAPPINGS = {
    "PM_RetinaFace": RetinaFacePM,
    "PM_FaceFusion": FaceFusionPM,
    "PM_RatioMerge2Image": RatioMerge2ImagePM,
    "PM_MaskMerge2Image": MaskMerge2ImagePM,
    "PM_ReplaceBoxImg": ReplaceBoxImgPM,
    "PM_ExpandMaskBox": ExpandMaskFaceWidthPM,
    "PM_BoxCropImage": BoxCropImagePM,
    "PM_ColorTransfer": ColorTransferPM,
    "PM_FaceSkin": FaceSkinPM,
    "PM_MaskDilateErode": MaskDilateErodePM,
    "PM_SkinRetouching": SkinRetouchingPM,
    "PM_PortraitEnhancement": PortraitEnhancementPM,
    "PM_ImageScaleShort": ImageScaleShortPM,
    "PM_ImageResizeTarget": ImageResizeTargetPM,
    "PM_GetImageInfo": GetImageInfoPM,
    "PM_MakeUpTransfer": MakeUpTransferPM,
    "PM_FaceShapMatch": FaceShapMatchPM,
    "PM_SuperColorTransfer": SuperColorTransferPM,
    "PM_SuperMakeUpTransfer": SuperMakeUpTransferPM,
    "PM_Similarity": SimilarityPM,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "PM_RetinaFace": "RetinaFace PM",
    "PM_FaceFusion": "FaceFusion PM",
    "PM_RatioMerge2Image": "RatioMerge2Image PM",
    "PM_MaskMerge2Image": "MaskMerge2Image PM",
    "PM_ReplaceBoxImg": "ReplaceBoxImg PM",
    "PM_ExpandMaskBox": "ExpandMaskBox PM",
    "PM_BoxCropImage": "BoxCropImage PM",
    "PM_ColorTransfer": "ColorTransfer PM",
    "PM_FaceSkin": "FaceSkin PM",
    "PM_MaskDilateErode": "MaskDilateErode PM",
    "PM_SkinRetouching": "SkinRetouching PM",
    "PM_PortraitEnhancement": "PortraitEnhancement PM",
    "PM_ImageScaleShort": "ImageScaleShort PM",
    "PM_ImageResizeTarget": "ImageResizeTarget PM",
    "PM_GetImageInfo": "GetImageInfo PM",
    "PM_MakeUpTransfer": "MakeUpTransfer PM",
    "PM_FaceShapMatch": "FaceShapMatch PM",
    "PM_SuperColorTransfer": "SuperColorTransfer PM",
    "PM_SuperMakeUpTransfer": "SuperMakeUpTransfer PM",
    "PM_Similarity": "Similarity PM",

}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
