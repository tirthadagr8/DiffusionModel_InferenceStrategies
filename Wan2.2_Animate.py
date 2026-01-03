import torch
from PIL import Image
from diffsynth.core import load_state_dict
from diffsynth.utils.data import save_video, VideoData
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from modelscope import dataset_snapshot_download, snapshot_download
import time
import numpy as np

vram_config = {
    "offload_dtype": torch.bfloat16,
    "offload_device": "cpu",
    "onload_dtype": torch.bfloat16,
    "onload_device": "cuda",
    "preparing_dtype": torch.bfloat16,
    "preparing_device": "cuda",
    "computation_dtype": torch.bfloat16,
    "computation_device": "cuda",
}

pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Wan-AI/Wan2.2-Animate-14B", origin_file_pattern="diffusion_pytorch_model*.safetensors",**vram_config),
        ModelConfig(model_id="Wan-AI/Wan2.2-Animate-14B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth",**vram_config),
        ModelConfig(model_id="Wan-AI/Wan2.2-Animate-14B", origin_file_pattern="Wan2.1_VAE.pth",**vram_config),
        ModelConfig(model_id="Wan-AI/Wan2.2-Animate-14B", origin_file_pattern="models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",**vram_config),
    ],
    tokenizer_config=ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/"),
)

# 🔥 FORCE animate adapter to GPU
pipe.animate_adapter.to("cuda", dtype=torch.bfloat16)

# dataset_snapshot_download(
#     dataset_id="DiffSynth-Studio/examples_in_diffsynth",
#     local_dir="./",
#     allow_file_pattern="data/examples/wan/animate/*",
# )
def resize_video_frames(video_array, target_size=(1280, 720)):
    resized_frames = []
    for frame in video_array:
        # If it's already a PIL Image, use it directly
        if isinstance(frame, Image.Image):
            img = frame
        else:
            # If it's a numpy array, convert to PIL Image
            img = Image.fromarray(frame.astype('uint8'))
        # Resize and keep as PIL Image
        img = img.resize(target_size, Image.LANCZOS)
        resized_frames.append(img)  # Append PIL Image, NOT array
    return resized_frames
# Animate

target_width,target_height = 864,480 # 1280x720 or 854x480  should be divisible by 16
total_frames = 121
input_image = Image.open("<redacted>").resize((target_width,target_height))
animate_pose_video = resize_video_frames(VideoData("<redacted>").raw_data(),target_size=(target_width,target_height))[:total_frames-4]
animate_face_video = resize_video_frames(VideoData("<redacted>").raw_data(), target_size=(512, 512))[:total_frames-4]
#Shape of animate_pose_video: 77 (1408, 640)
#Shape of animate_face_video: 77 (512, 512)
print("Shape of animate_pose_video:", len(animate_pose_video), animate_pose_video[0].size)
print("Shape of animate_face_video:", len(animate_face_video), animate_face_video[0].size)
start_time = time.time()
video = pipe(
    prompt="<redacted>",
    seed=0, tiled=True,
    input_image=input_image,
    animate_pose_video=animate_pose_video,
    animate_face_video=animate_face_video,
    num_frames=total_frames, height=target_height, width=target_width, # with 48G VRAM, can only do 81 frames of 1280x720 and 121 frames of 864x480
    num_inference_steps=20, cfg_scale=1,
)
save_video(video, "video_1_Wan2.2-Animate-14B.mp4", fps=24, quality=5)
print(f"Generation time: {(time.time() - start_time):.2f} seconds")
# Replace
# snapshot_download("Wan-AI/Wan2.2-Animate-14B", allow_file_pattern="relighting_lora.ckpt", local_dir="models/Wan-AI/Wan2.2-Animate-14B")
# lora_state_dict = load_state_dict("models/Wan-AI/Wan2.2-Animate-14B/relighting_lora.ckpt", torch_dtype=torch.bfloat16, device="cuda")["state_dict"]
# pipe.load_lora(pipe.dit, state_dict=lora_state_dict,**vram_config)

# for name, param in pipe.dit.named_parameters():
#     if "lora" in name:
#         param.data = param.data.to(torch.bfloat16)


# input_image = Image.open("data/examples/wan/animate/replace_input_image.png")
# animate_pose_video = VideoData("data/examples/wan/animate/replace_pose_video.mp4").raw_data()[:81-4]
# animate_face_video = VideoData("data/examples/wan/animate/replace_face_video.mp4").raw_data()[:81-4]
# animate_inpaint_video = VideoData("data/examples/wan/animate/replace_inpaint_video.mp4").raw_data()[:81-4]
# animate_mask_video = VideoData("data/examples/wan/animate/replace_mask_video.mp4").raw_data()[:81-4]
# video = pipe(
#     prompt="视频中的人在做动作",
#     seed=0, tiled=True,
#     input_image=input_image,
#     animate_pose_video=animate_pose_video,
#     animate_face_video=animate_face_video,
#     animate_inpaint_video=animate_inpaint_video,
#     animate_mask_video=animate_mask_video,
#     num_frames=81, height=720, width=1280,
#     num_inference_steps=20, cfg_scale=1,
# )
# save_video(video, "video_2_Wan2.2-Animate-14B.mp4", fps=15, quality=5)
