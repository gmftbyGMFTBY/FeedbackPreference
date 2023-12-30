#!/bin/bash

srun -p llmit2 --quotatype spot --gres=gpu:4 -N1 --cpus-per-task=24 python train.py
