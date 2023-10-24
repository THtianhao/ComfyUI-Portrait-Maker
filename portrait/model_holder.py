from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import insightface
from insightface.app import FaceAnalysis
from .utils.face_process_utils import Face_Skin
from .utils.psgan_utils import PSGAN_Inference

from .config import *

retinaface_detection = None
image_face_fusion = None
face_analysis = None
face_skin = None
roop = None
skin_retouching = None
portrait_enhancement = None
psgan_interface = None

def get_retinaface_detection():
    global retinaface_detection
    if retinaface_detection is None:
        retinaface_detection = pipeline(Tasks.face_detection, 'damo/cv_resnet50_face-detection_retinaface', model_revision='v2.0.2')
    return retinaface_detection

def get_image_face_fusion():
    global image_face_fusion
    if image_face_fusion is None:
        image_face_fusion = pipeline(Tasks.image_face_fusion, model='damo/cv_unet-image-face-fusion_damo', model_revision='v1.3')
    return image_face_fusion

def get_face_analysis():
    global face_analysis
    if face_analysis is None:
        face_analysis = FaceAnalysis(name='buffalo_l')
    return face_analysis

def get_roop():
    global roop
    if roop is None:
        roop = insightface.model_zoo.get_model('inswapper_128.onnx', download=True, download_zip=True, root=root_path)
    return roop

def get_face_skin():
    global face_skin
    if face_skin is None:
        face_skin = Face_Skin(os.path.join(models_path, "face_skin.pth"))
    return face_skin

def get_skin_retouching():
    global skin_retouching
    if skin_retouching is None:
        skin_retouching = pipeline('skin-retouching-torch', model='damo/cv_unet_skin_retouching_torch', model_revision='v1.0.2')
    return skin_retouching

def get_portrait_enhancement():
    global portrait_enhancement
    if portrait_enhancement is None:
        portrait_enhancement = pipeline(Tasks.image_portrait_enhancement, model='damo/cv_gpen_image-portrait-enhancement', model_revision='v1.0.0')
    return portrait_enhancement

def get_pagan_interface():
    global psgan_interface
    if psgan_interface is None:
        face_landmarks_model_path = os.path.join(models_path, "face_landmarks.pth")
        makeup_transfer_model_path = os.path.join(models_path, "makeup_transfer.pth")
        psgan_interface = PSGAN_Inference("cuda", makeup_transfer_model_path, get_retinaface_detection(), get_face_skin(), face_landmarks_model_path)
    return psgan_interface
