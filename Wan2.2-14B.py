import torch
from PIL import Image
from diffsynth.utils.data import save_video
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from modelscope import dataset_snapshot_download

vram_config = {
    "offload_dtype": torch.float8_e4m3fn,
    "offload_device": "cpu",
    "onload_dtype": torch.float8_e4m3fn,
    "onload_device": "cuda",
    "preparing_dtype": torch.float8_e4m3fn,
    "preparing_device": "cuda",
    "computation_dtype": torch.bfloat16,
    "computation_device": "cuda",
}

pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Wan-AI/Wan2.2-I2V-A14B", origin_file_pattern="high_noise_model/diffusion_pytorch_model*.safetensors",**vram_config),
        ModelConfig(model_id="Wan-AI/Wan2.2-I2V-A14B", origin_file_pattern="low_noise_model/diffusion_pytorch_model*.safetensors",**vram_config),
        ModelConfig(model_id="Wan-AI/Wan2.2-I2V-A14B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth",**vram_config),
        ModelConfig(model_id="Wan-AI/Wan2.2-I2V-A14B", origin_file_pattern="Wan2.1_VAE.pth",**vram_config),
    ],
    tokenizer_config=ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/"),
)

# pipe.load_lora(pipe.dit,"<redacted>",alpha=1)
# pipe.load_lora(pipe.dit2,"<redacted>",alpha=1)

dataset_snapshot_download(
    dataset_id="DiffSynth-Studio/examples_in_diffsynth",
    local_dir="./",
    allow_file_pattern=["data/examples/wan/cat_fightning.jpg"]
)
input_image = Image.open("<redacted>").resize((1280, 720)).convert("RGB")
# prompt1:
# prompt2:

video = pipe(
    prompt="The person in the image gradually transitions into dancing, shifting their weight and moving their arms and legs in smooth, rhythmic motions. The dance feels natural and fluid, with expressive body language, subtle head movements, and dynamic energy, as if they are fully immersed in the music.",
    negative_prompt="",
    seed=0, tiled=True,
    input_image=input_image,
    switch_DiT_boundary=0.9,
    num_frames=45, # num_frames % 4 != 1
)
save_video(video, "video_Wan2.2-I2V-A14B.mp4", fps=15, quality=5)
