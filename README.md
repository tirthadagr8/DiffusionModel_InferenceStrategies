# DiffusionModel_InferenceStrategies

## Hardware Specs
+ GPU: NVIDIA RTX PRO 5000 BlackWell 48G
+ RAM: 96G

## Metrics:
+ Flux Image editing (Text encoder + Transformer with 8bit quant and 25 steps): ~88s
+ Flux Image editing (Text encoder + Transformer with 8bit quant, turbo lora and 8 steps): ~67s
+ Hunyuan 1.5 Image to Video (5 second 480p): 131s
