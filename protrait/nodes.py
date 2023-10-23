import os

import cv2
import numpy as np
from PIL import Image
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from utils.face_process_utils import call_face_crop, color_transfer, Face_Skin
from utils.img_utils import img_to_tensor, tensor_to_img, tensor_to_np, np_to_tensor, np_to_mask, img_to_mask
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

# import pydevd_pycharm
# pydevd_pycharm.settrace('49.7.62.197', port=10090, stdoutToServer=True, stderrToServer=True)

class RetainFace:
    def __init__(self):
        self.retinaface_detection = pipeline(Tasks.face_detection, 'damo/cv_resnet50_face-detection_retinaface', model_revision='v2.0.2')

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("IMAGE",),
                             "multi_user_facecrop_ratio": ("FLOAT", {"default": 1, "min": 0, "max": 10, "step": 0.1})
                             }}

    RETURN_TYPES = ("IMAGE", "MASK", "BOX")
    RETURN_NAMES = ("crop_image", "crop_mask", "crop_box")
    FUNCTION = "retain_face"
    CATEGORY = "protrait/model"

    def retain_face(self, image, multi_user_facecrop_ratio):
        np_image = np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
        image = Image.fromarray(np_image)
        retinaface_boxes, retinaface_keypoints, retinaface_masks, retinaface_tensor = call_face_crop(self.retinaface_detection, image, multi_user_facecrop_ratio)
        crop_image = image.crop(retinaface_boxes[0])
        return (img_to_tensor(crop_image), retinaface_tensor, retinaface_boxes[0])

class FaceFusionPM:

    def __init__(self):
        self.image_face_fusion = pipeline(Tasks.image_face_fusion, model='damo/cv_unet-image-face-fusion_damo', model_revision='v1.3')

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"source_image": ("IMAGE",),
                             "swap_image": ("IMAGE",),
                             "mode": (["ali", "roop"],),
                             }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "img_face_fusion"

    CATEGORY = "protrait/model"

    def img_face_fusion(self, source_image, swap_image, mode):
        result_image = None
        if mode == "ali":
            source_image = tensor_to_img(source_image)
            swap_image = tensor_to_img(swap_image)
            fusion_image = self.image_face_fusion(dict(template=source_image, user=source_image))[
                OutputKeys.OUTPUT_IMG]
            # swap_face(target_img=output_image, source_img=roop_image, model="inswapper_128.onnx", upscale_options=UpscaleOptions())
            result_image = Image.fromarray(cv2.cvtColor(fusion_image, cv2.COLOR_BGR2RGB))
        else:
            app = FaceAnalysis(name='buffalo_l')

        return (img_to_tensor(fusion_image),)

class RatioMerge2Image:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image1": ("IMAGE",),
                             "image2": ("IMAGE",),
                             "fusion_rate": ("FLOAT", {"default": 0.5, "min": 0, "max": 1, "step": 0.1})
                             }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "image_ratio_merge"

    CATEGORY = "protrait/model"

    def image_ratio_merge(self, image1, image2, fusion_rate):
        rate_fusion_image = image1 * (1 - fusion_rate) + image2 * fusion_rate
        return (rate_fusion_image,)

class ReplaceBoxImg:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"origin_image": ("IMAGE",),
                             "box_area": ("BOX",),
                             "replace_image": ("IMAGE",),
                             }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "replace_box_image"

    CATEGORY = "protrait/model"

    def replace_box_image(self, origin_image, box_area, replace_image):
        origin_image[:, box_area[1]:box_area[3], box_area[0]:box_area[2], :] = replace_image
        return (origin_image,)

class MaskMerge2Image:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image1": ("IMAGE",),
                             "image2": ("IMAGE",),
                             "mask": ("MASK",),
                             },
                }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "image_mask_merge"

    CATEGORY = "protrait/model"

    def image_mask_merge(self, image1, image2, mask, box=None):
        mask = mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])).movedim(1, -1).expand(-1, -1, -1, 3)
        image1 = image1 * mask + image2 * (1 - mask)
        return (image1,)

class ExpandMaskFaceWidth:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"mask": ("MASK",),
                             "box": ("BOX",),
                             "expand_width": ("FLOAT", {"default": 0.15, "min": 0, "max": 10, "step": 0.1})
                             }}

    RETURN_TYPES = ("MASK", "BOX")
    FUNCTION = "expand_mask_face_width"

    CATEGORY = "protrait/model"

    def expand_mask_face_width(self, mask, box, expand_width):
        h, w = mask.shape[1], mask.shape[2]

        new_mask = mask.clone().zero_()
        copy_box = np.copy(np.int32(box))

        face_width = copy_box[2] - copy_box[0]
        copy_box[0] = np.clip(np.array(copy_box[0], np.int32) - face_width * expand_width, 0, w - 1)
        copy_box[2] = np.clip(np.array(copy_box[2], np.int32) + face_width * expand_width, 0, w - 1)

        # get new input_mask
        new_mask[0, copy_box[1]:copy_box[3], copy_box[0]:copy_box[2]] = 255
        return (new_mask, copy_box)

class BoxCropImage:

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"image": ("IMAGE",),
                     "box": ("BOX",), }
                }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("crop_image",)
    FUNCTION = "box_crop_image"
    CATEGORY = "protrait/model"

    def box_crop_image(self, image, box):
        image = image[:, box[1]:box[3], box[0]:box[2], :]
        return (image,)

class ColorTransfer:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "transfer_from": ("IMAGE",),
            "transfer_to": ("IMAGE",),
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "color_transfer"

    CATEGORY = "protrait/model"

    def color_transfer(self, transfer_from, transfer_to):
        transfer_result = color_transfer(tensor_to_np(transfer_from), tensor_to_np(transfer_to))  # 进行颜色迁移
        return (np_to_tensor(transfer_result),)

class FaceSkin:
    def __init__(self):
        self.retinaface_detection = pipeline(Tasks.face_detection, 'damo/cv_resnet50_face-detection_retinaface', model_revision='v2.0.2')
        self.face_skin = Face_Skin(os.path.join(models_path, "face_skin.pth"))

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"image": ("IMAGE",), }
                }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "face_skin_mask"

    CATEGORY = "protrait/model"

    def face_skin_mask(self, image):
        face_skin_one = self.face_skin.detect(tensor_to_img(image), self.retinaface_detection, [1, 2, 3, 4, 5, 10, 12, 13])
        return (face_skin_one,)

class MaskDilateErode:

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"mask": ("MASK",), }
                }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "mask_dilate_erode"

    CATEGORY = "protrait/model"

    def mask_dilate_erode(self, mask):
        out_mask = Image.fromarray(np.uint8(cv2.dilate(tensor_to_np(mask), np.ones((96, 96), np.uint8), iterations=1) - cv2.erode(tensor_to_np(mask), np.ones((48, 48), np.uint8), iterations=1)))
        return (img_to_mask(out_mask),)

class SkinRetouching:

    def __init__(self):
        self.skin_retouching = pipeline('skin-retouching-torch', model='damo/cv_unet_skin_retouching_torch', model_revision='v1.0.2')

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"image": ("IMAGE",)}
                }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "skin_retouching_pass"
    CATEGORY = "protrait/model"

    def skin_retouching_pass(self, image):
        output_image = cv2.cvtColor(self.skin_retouching(tensor_to_img(image))[OutputKeys.OUTPUT_IMG], cv2.COLOR_BGR2RGB)
        return (np_to_tensor(output_image),)

class PortraitEnhancement:

    def __init__(self):
        self.portrait_enhancement = pipeline(Tasks.image_portrait_enhancement, model='damo/cv_gpen_image-portrait-enhancement', model_revision='v1.0.0')

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"image": ("IMAGE",), }
                }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "protrait_enhancement_pass"

    CATEGORY = "protrait/model"

    def protrait_enhancement_pass(self, image):
        output_image = cv2.cvtColor(self.portrait_enhancement(tensor_to_img(image))[OutputKeys.OUTPUT_IMG], cv2.COLOR_BGR2RGB)
        return (np_to_tensor(output_image),)

class ImageScaleShort:

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "image": ("IMAGE",),
            "size": ("INT", {"default": 512, "min": 0, "max": 2048, "step": 1}),
            "crop_face": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "image_scale_short"

    CATEGORY = "protrait/model"

    def image_scale_short(self, image, size, crop_face):
        input_image = tensor_to_img(image)
        short_side = min(input_image.width, input_image.height)
        resize = float(short_side / size)
        new_size = (int(input_image.width // resize), int(input_image.height // resize))
        input_image = input_image.resize(new_size, Image.Resampling.LANCZOS)
        if crop_face:
            new_width = int(np.shape(input_image)[1] // 32 * 32)
            new_height = int(np.shape(input_image)[0] // 32 * 32)
            input_image = input_image.resize([new_width, new_height], Image.Resampling.LANCZOS)
        return (img_to_tensor(input_image),)

class ImageResizeTarget:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "image": ("IMAGE",),
            "width": ("INT", {"default": 512, "min": 0, "max": 2048, "step": 1}),
            "height": ("INT", {"default": 512, "min": 0, "max": 2048, "step": 1}),
        }}

    RETURN_TYPES = ("IMAGE",)

    FUNCTION = "image_resize_target"

    CATEGORY = "protrait/model"

    def image_resize_target(self, image, width, height):
        imagepi = tensor_to_img(image)
        out = imagepi.resize([width, height], Image.Resampling.LANCZOS)
        return (img_to_tensor(out),)

class GetImageInfo:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "image": ("IMAGE",),
        }}

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")

    FUNCTION = "get_image_info"

    CATEGORY = "protrait/model"

    def get_image_info(self, image):
        width = image.shape[2]
        height = image.shape[1]
        return (width, height)
