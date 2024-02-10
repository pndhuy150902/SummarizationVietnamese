#!/bin/sh
accelerate launch --config_file=../config/deepspeed_stage_2.yaml train_model.py