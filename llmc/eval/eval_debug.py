import gc
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn as nn
from datasets import load_dataset, load_from_disk
from loguru import logger
from tqdm import tqdm

from .eval_base import BaseEval


class DebugEval(BaseEval):
    @torch.no_grad()
    def eval_func(self, model, testenc, seq_len, bs, eval_pos):
        if isinstance(testenc, list):
            testenc = [x.input_ids for x in testenc]
        else:
            testenc = testenc.input_ids
        nsamples = self.num_samples if (self.num_samples and self.num_samples>0) else len(testenc)

        # Loop through each batch
        for i in range(0, nsamples, bs):
            logger.info(f'index : {(i + 1) // bs}/{nsamples // bs}')
            # Calculate end index
            j = min(i + bs, nsamples)

            # Prepare inputs and move to gpu
            inputs = (torch.cat(testenc[i:j]) if isinstance(testenc, list) else testenc[i:j]).cuda()

            # Forward pass through the model
            lm_logits = model.model(inputs).logits
            model.reset_kv()



        # # Empty CUDA cache to save memory
        # if isinstance(testenc, list):
        #     for x in testenc:
        #         x.cpu()
        # else:
        #     testenc.cpu()
        # torch.cuda.empty_cache()

        return 0
