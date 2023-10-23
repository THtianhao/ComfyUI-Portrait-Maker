import sys
import os

main_path = os.path.dirname(__file__)
sys.path.append(main_path)
from portrait.config import *

import subprocess
import threading

import requests
from tqdm import tqdm
from portrait.nodes import *

# import pydevd_pycharm
# pydevd_pycharm.settrace('49.7.62.197', port=10090, stdoutToServer=True, stderrToServer=True)


def handle_stream(stream, prefix):
    for line in stream:
        print(prefix, line, end="")

def run_script(cmd, cwd='.'):
    process = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)

    stdout_thread = threading.Thread(target=handle_stream, args=(process.stdout, ""))
    stderr_thread = threading.Thread(target=handle_stream, args=(process.stderr, "[!]"))

    stdout_thread.start()
    stderr_thread.start()

    stdout_thread.join()
    stderr_thread.join()

    return process.wait()

print("##  installing dependencies")

requirements_path = os.path.join(root_path, "requirements.txt")
run_script([sys.executable, '-s', '-m', 'pip', 'install', '-q', '-r', requirements_path])

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
    "RetainFace": RetainFace,
    "FaceFusionPM": FaceFusionPM,
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
    "ImageScaleShort": ImageScaleShort,
    "ImageResizeTarget": ImageResizeTarget,
    "GetImageInfo": GetImageInfo,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "RetainFace": "RetainFace PM",
    "FaceFusionPM": "FaceFusion PM",
    "RatioMerge2Image": "RatioMerge2Image PM",
    "MaskMerge2Image": "MaskMerge2Image PM",
    "ReplaceBoxImg": "ReplaceBoxImg PM",
    "ExpandMaskBox": "ExpandMaskBox PM",
    "BoxCropImage": "BoxCropImage PM",
    "ColorTransfer": "ColorTransfer PM",
    "FaceSkin": "FaceSkin PM",
    "MaskDilateErode": "MaskDilateErode PM",
    "SkinRetouching": "SkinRetouching PM",
    "PortraitEnhancement": "PortraitEnhancement PM",
    "ImageScaleShort": "ImageScaleShort PM",
    "ImageResizeTarget": "ImageResizeTarget PM",
    "GetImageInfo": "GetImageInfo PM",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
