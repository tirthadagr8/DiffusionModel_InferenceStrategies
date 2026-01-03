import torch
import gc
from diffusers import Flux2Pipeline, AutoModel
from transformers import Mistral3ForConditionalGeneration, BitsAndBytesConfig
from diffusers.utils import load_image

# --- SETTINGS ---
repo_id = "models/black-forest-labs/FLUX.2-dev/"

lora_path = "./LoRA/flux.2-turbo-lora.safetensors"
TURBO_SIGMAS = [1.0, 0.6509, 0.4374, 0.2932, 0.1893, 0.1108, 0.0495, 0.00031]

device = "cuda:0"
prompt = "Make the image cool looking, add some supercool effects in background. Lower the hand position that is holding phone."
loaded_image = load_image("")

# --- STEP 1: ENCODE PROMPT (Text Encoder Only) ---
print(">>> STEP 1: Loading Text Encoder (8-bit)...")

quant_config_8bit = BitsAndBytesConfig(
    load_in_8bit=True,
)

text_encoder = Mistral3ForConditionalGeneration.from_pretrained(
    repo_id, 
    subfolder="text_encoder", 
    torch_dtype=torch.float16, 
    quantization_config=quant_config_8bit,
)

pipe_pre = Flux2Pipeline.from_pretrained(
    repo_id,
    text_encoder=text_encoder,
    transformer=None,
    vae=None,
    torch_dtype=torch.float16,
)

print(">>> Encoding Prompt...")
with torch.no_grad():
    # CHANGE 1: We unpack `prompt_embeds` and ignore the second value (_)
    # Flux2 encode_prompt returns (embeds, pooled_embeds), but the generation pipe
    # only wants the first one.
    prompt_embeds, _ = pipe_pre.encode_prompt(
        prompt=prompt,
        #prompt_2=None,
        device=device
    )

# --- CLEANUP STEP 1 ---
print(">>> Unloading Text Encoder to free VRAM...")
del text_encoder
del pipe_pre
gc.collect()
torch.cuda.empty_cache()

# --- STEP 2: GENERATE IMAGE (Transformer Only) ---
print(">>> STEP 2: Loading Transformer (8-bit)...")

temp_pipe = Flux2Pipeline.from_pretrained(
    repo_id,
    torch_dtype=torch.float32,  # VAE will be float32
    text_encoder=None,
    transformer=None,
    safety_checker=None,
)
vae = temp_pipe.vae  # This is AutoencoderKLFlux2
del temp_pipe
gc.collect()

dit = AutoModel.from_pretrained(
    repo_id, 
    subfolder="transformer", 
    torch_dtype=torch.float16, 
    quantization_config=quant_config_8bit,
)

pipe = Flux2Pipeline.from_pretrained(
    repo_id,
    text_encoder=None, 
    transformer=dit,
    vae=vae,
    torch_dtype=torch.float16,
)

print(">>> Loading LoRA...")
pipe.load_lora_weights(lora_path,
                    weight_name="distill_weights", 
                    adapter_name="default",
                    low_cpu_mem_usage=True)
pipe.set_adapters(["default"], adapter_weights=[0.8])

# Enable CPU offload for the VAE (which is not quantized)
# pipe.enable_model_cpu_offload()

print(">>> Generating Image...")
image = pipe(
    prompt_embeds=prompt_embeds, 
    image=[loaded_image],
    generator=torch.Generator(device=device).manual_seed(42),
    num_inference_steps=8,
    guidance_scale=4,
    sigmas = TURBO_SIGMAS,
).images[0]

image.save("./output/flux2_output_final.png")
print(">>> Done.")
