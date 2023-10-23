import numpy as np
import torch
from PIL import ImageOps
from PIL import Image

# import pydevd_pycharm
# pydevd_pycharm.settrace('49.7.62.197', port=10090, stdoutToServer=True, stderrToServer=True)

def img_to_tensor(input):
    i = ImageOps.exif_transpose(input)
    image = i.convert("RGB")
    image = np.array(image).astype(np.float32) / 255.0
    tensor = torch.from_numpy(image)[None,]
    return tensor

def img_to_mask(input):
    i = ImageOps.exif_transpose(input)
    image = i.convert("RGB")
    new_np = np.array(image).astype(np.float32) / 255.0
    mask_tensor = torch.from_numpy(new_np).permute(2, 0, 1)[0:1, :, :]
    return mask_tensor

def np_to_tensor(input):
    image = input.astype(np.float32) / 255.0
    tensor = torch.from_numpy(image)[None,]
    return tensor

def tensor_to_img(image):
    image = image[0]
    i = 255. * image.cpu().numpy()
    img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8)).convert("RGB")
    return img

def tensor_to_np(image):
    image = image[0]
    i = 255. * image.cpu().numpy()
    result = np.clip(i, 0, 255).astype(np.uint8)
    return result

def np_to_mask(input):
    new_np = input.astype(np.float32) / 255.0
    tensor = torch.from_numpy(new_np).permute(2, 0, 1)[0:1, :, :]
    return tensor
