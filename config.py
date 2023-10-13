import os, glob
from folder_paths import folder_names_and_paths

root_path = os.path.dirname(__file__)
utils_path = os.path.join(os.path.dirname(__file__), "utils")
models_path = os.path.join(os.path.dirname(__file__), "models")
# save_dirs
urls = [
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
    os.path.join(folder_names_and_paths['controlnet'][0][0], "control_v11p_sd15_openpose.pth"),
    os.path.join(folder_names_and_paths['controlnet'][0][0], "control_v11p_sd15_canny.pth"),
    os.path.join(folder_names_and_paths['controlnet'][0][0], "control_v11f1e_sd15_tile.pth"),
    os.path.join(folder_names_and_paths['controlnet'][0][0], "control_sd15_random_color.pth"),
    os.path.join(folder_names_and_paths['loras'][0][0], "FilmVelvia3.safetensors"),
    os.path.join(folder_names_and_paths['controlnet'][0][0], "body_pose_model.pth"),
    os.path.join(folder_names_and_paths['controlnet'][0][0], "facenet.pth"),
    os.path.join(folder_names_and_paths['controlnet'][0][0], "hand_pose_model.pth"),
    os.path.join(folder_names_and_paths['vae'][0][0], "VAE/vae-ft-mse-840000-ema-pruned.ckpt"),
    os.path.join(models_path, "face_skin.pth"),
]
# prompts
validation_prompt = "easyphoto_face, easyphoto, 1person"
DEFAULT_POSITIVE = '(cloth:1.7), (best quality), (realistic, photo-realistic:1.2), detailed skin, beautiful, cool, finely detail, light smile, extremely detailed CG unity 8k wallpaper, huge filesize, best quality, realistic, photo-realistic, ultra high res, raw photo, put on makeup'
DEFAULT_NEGATIVE = '(bags under the eyes:1.5), (Bags under eyes:1.5), (glasses:1.5), (naked:2.0), nude, (nsfw:2.0), breasts, penis, cum, (worst quality:2), (low quality:2), (normal quality:2), over red lips, hair, teeth, lowres, watermark, badhand, (normal quality:2), lowres, bad anatomy, bad hands, normal quality, mural,'
