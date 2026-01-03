import torch
from PIL import Image
from diffsynth.core import load_state_dict
from diffsynth.utils.data import save_video, VideoData
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
import time
import numpy as np
import gc

# --- Configuration ---
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

# --- Load Pipeline ---
pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Wan-AI/Wan2.2-Animate-14B", origin_file_pattern="diffusion_pytorch_model*.safetensors", **vram_config),
        ModelConfig(model_id="Wan-AI/Wan2.2-Animate-14B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", **vram_config),
        ModelConfig(model_id="Wan-AI/Wan2.2-Animate-14B", origin_file_pattern="Wan2.1_VAE.pth", **vram_config),
        ModelConfig(model_id="Wan-AI/Wan2.2-Animate-14B", origin_file_pattern="models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth", **vram_config),
    ],
    tokenizer_config=ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/"),
)

# Force animate adapter to GPU
pipe.animate_adapter.to("cuda", dtype=torch.bfloat16)
pipe.load_lora(pipe.dit, "",alpha=1.1)

# --- Helper Functions ---
def resize_video_frames(video_array, target_size=(1280, 720)):
    resized_frames = []
    for frame in video_array:
        if isinstance(frame, Image.Image):
            img = frame
        else:
            # Convert numpy to PIL
            img = Image.fromarray(frame.astype('uint8'))
        img = img.resize(target_size, Image.LANCZOS)
        resized_frames.append(img)
    return resized_frames

# --- Settings ---
# Note: Keeping your requested logic of 121 gen frames / 117 input frames
target_width, target_height = 1280, 720 
gen_frames = 41
input_frames = 37  # n*4+1 (e.g. 29*4+1 = 117)

# --- Load Data ---
# 1. Initial Reference Image
first_input_image = Image.open("").resize((target_width, target_height))

# 2. Load FULL source videos
print("Loading and resizing full source videos (Pose, Face, Inpaint, Mask)...")
raw_pose = VideoData("").raw_data()
raw_face = VideoData("").raw_data()
raw_inpaint = VideoData("").raw_data()
raw_mask = VideoData("").raw_data()

# Resize all video streams
# Note: Face is usually 512x512 for Wan, others match target resolution
full_pose_video = resize_video_frames(raw_pose, target_size=(target_width, target_height))
full_face_video = resize_video_frames(raw_face, target_size=(512, 512)) 
full_inpaint_video = resize_video_frames(raw_inpaint, target_size=(target_width, target_height))
full_mask_video = resize_video_frames(raw_mask, target_size=(target_width, target_height))

# --- Continuous Generation Loop ---
final_video_frames = []
current_input_image = first_input_image
start_time = time.time()

# Determine number of segments based on shortest video
min_len = 42#min(len(full_pose_video), len(full_face_video), len(full_inpaint_video), len(full_mask_video))
num_segments = min_len // input_frames

print(f"Starting Continuous Replacement: {num_segments} segments.")
print(f"Frame Config: Input Chunk={input_frames}, Output Chunk={gen_frames}")

for i in range(num_segments):
    print(f"\n--- Processing Replacement Segment {i+1}/{num_segments} ---")
    
    # 1. Determine Slices
    start_idx = i * input_frames
    end_idx = start_idx + input_frames
    
    # Slice all controls
    chunk_pose = full_pose_video[start_idx:end_idx]
    chunk_face = full_face_video[start_idx:end_idx]
    chunk_inpaint = full_inpaint_video[start_idx:end_idx]
    chunk_mask = full_mask_video[start_idx:end_idx]

    # 2. Run Pipeline
    segment_video = pipe(
        prompt="视频中的人在做动作",
        seed=0, 
        tiled=True,
        input_image=current_input_image,
        animate_pose_video=chunk_pose,
        animate_face_video=chunk_face,
        animate_inpaint_video=chunk_inpaint,
        animate_mask_video=chunk_mask,
        num_frames=gen_frames, 
        height=target_height, 
        width=target_width,
        num_inference_steps=20, 
        cfg_scale=1,
    )
    
    # 3. Stitching
    if i == 0:
        final_video_frames.extend(segment_video)
    else:
        # Skip the first frame of subsequent segments to avoid duplicates
        final_video_frames.extend(segment_video[1:])
    
    # 4. Update Context
    current_input_image = segment_video[-1]
    
    # 5. VRAM Cleanup
    del segment_video, chunk_pose, chunk_face, chunk_inpaint, chunk_mask
    torch.cuda.empty_cache()
    gc.collect()

# --- Save Final Result ---
print(f"\nGeneration Complete!")
print(f"Total Generation time: {(time.time() - start_time):.2f} seconds")
save_video(final_video_frames, ".mp4", fps=24, quality=5)
print("Video saved to .mp4")
