#!/bin/bash

source ~/venv/bin/activate
cd <redacted>
python preprocess_data.py --ckpt_path "<redacted>" --video_path "<redacted>" \
--refer_path "<redacted>" \
--save_path "<redacted>" --fps 30 \
--replace_flag
# --retarget_flag --use_flux\
