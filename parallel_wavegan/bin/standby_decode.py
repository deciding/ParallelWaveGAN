#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

"""Decode with trained Parallel WaveGAN Generator."""

import logging
import os
import time

import torch
torch.set_num_threads(8)
import yaml

from parallel_wavegan.utils import load_model
#from parallel_wavegan.datasets import MelDataset


class PWG:
    def __init__(self, checkpoint, config=None):
        # load config
        if config is None:
            dirname = os.path.dirname(checkpoint)
            config = os.path.join(dirname, "config.yml")
        with open(config) as f:
            config = yaml.load(f, Loader=yaml.Loader)
        self.config = config

    def load_new_model(self, checkpoint):

        # setup model
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        device=torch.device('cpu')
        self.device = device

        model = load_model(checkpoint, self.config)
        #logging.info(f"Loaded model parameters from {checkpoint}.")
        model.remove_weight_norm()

        model = model.eval().to(device)
        return model

    def infer_waveform_by_model(self, model, mel, batched=False):

        # start generation
        #real-time factor sec of output per 1 sec wall time
        with torch.no_grad():
            c = torch.tensor(mel, dtype=torch.float).to(self.device)
            start = time.time()
            y = model.inference(c).view(-1)
            rtf = (time.time() - start) / (len(y) / self.config["sampling_rate"])

            y = y.cpu().numpy()

            # report average RTF
            #logging.info(f"Finished generation with RTF = {rtf:.03f}.")
            #print(f"Finished generation with RTF = {rtf:.03f}.")
            return y

