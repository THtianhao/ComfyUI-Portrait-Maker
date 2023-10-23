import sys
import os
from folder_paths import folder_names_and_paths

root_path = os.path.dirname(__file__)
utils_path = os.path.join(root_path, "utils")
models_path = os.path.join(root_path, "models")
# save_dirs
urls = [
    "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/ChilloutMix-ni-fp16.safetensors",
    "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/control_v11p_sd15_openpose.pth",
    "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/control_v11p_sd15_canny.pth",
    "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/control_v11f1e_sd15_tile.pth",
    "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/control_sd15_random_color.pth",
    "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/FilmVelvia3.safetensors",
    "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/body_pose_model.pth",
    "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/facenet.pth",
    "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/hand_pose_model.pth",
    "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/vae-ft-mse-840000-ema-pruned.ckpt",
    "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/face_skin.pth",
]
filenames = [
    os.path.join(folder_names_and_paths['checkpoints'][0][0], "Chilloutmix-Ni-pruned-fp16-fix.safetensors"),
    os.path.join(folder_names_and_paths['controlnet'][0][0], "control_v11p_sd15_openpose.pth"),
    os.path.join(folder_names_and_paths['controlnet'][0][0], "control_v11p_sd15_canny.pth"),
    os.path.join(folder_names_and_paths['controlnet'][0][0], "control_v11f1e_sd15_tile.pth"),
    os.path.join(folder_names_and_paths['controlnet'][0][0], "control_sd15_random_color.pth"),
    os.path.join(folder_names_and_paths['loras'][0][0], "FilmVelvia3.safetensors"),
    os.path.join(folder_names_and_paths['controlnet'][0][0], "body_pose_model.pth"),
    os.path.join(folder_names_and_paths['controlnet'][0][0], "facenet.pth"),
    os.path.join(folder_names_and_paths['controlnet'][0][0], "hand_pose_model.pth"),
    os.path.join(folder_names_and_paths['vae'][0][0], "vae-ft-mse-840000-ema-pruned.ckpt"),
    os.path.join(models_path, "face_skin.pth"),
]
# prompts
validation_prompt = "easyphoto_face, easyphoto, 1person"
DEFAULT_POSITIVE = '(cloth:1.7), (best quality), (realistic, photo-realistic:1.2), detailed skin, beautiful, cool, finely detail, light smile, extremely detailed CG unity 8k wallpaper, huge filesize, best quality, realistic, photo-realistic, ultra high res, raw photo, put on makeup'
DEFAULT_NEGATIVE = '(bags under the eyes:1.5), (Bags under eyes:1.5), (glasses:1.5), (naked:2.0), nude, (nsfw:2.0), breasts, penis, cum, (worst quality:2), (low quality:2), (normal quality:2), over red lips, hair, teeth, lowres, watermark, badhand, (normal quality:2), lowres, bad anatomy, bad hands, normal quality, mural,'

sys.path.append(root_path)

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
