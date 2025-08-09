import random
from typing import List, Optional, Union
import gc

from easydict import EasyDict
import numpy as np
import torch
from loguru import logger

from llmc.utils.registry_factory import MODEL_REGISTRY


class V4Eval:
    def __init__(self, config):
        self.eval_config = config.eval
        self.seq_len = self.eval_config["seq_len"]
        self.eval_dataset_name = self.eval_config["name"]
        self.eval_dataset_path = self.eval_config["path"]
        self.eval_bs = self.eval_config["bs"]
        self.eval_limit = self.eval_config.get("limit", -1)
        self.special_config = self.eval_config.get("special", {})
        self.inference_per_block = self.eval_config.get("inference_per_block", False)

    def eval(
        self,
        model_llmc,
        eval_pos=None,
    ):
        try:
            from xq_eval.get_v4_acc import evaluate_task
        except Exception as e:
            logger.error(f"Plese make sure git submodule updated!")
            raise e
        res_acc = {}
        task_config = {
            "name": self.eval_dataset_name,
            "test_data_path": self.eval_dataset_path,
        }
        if self.special_config:
            task_config.update(self.special_config)

        task_name = task_config["name"]

        handles = []
        if self.inference_per_block:
            handles = self.register_hooks(model_llmc)
        else:
            if model_llmc.mm_model:
                model_llmc.mm_model.cuda()
            else:
                model_llmc.model.cuda()

        if model_llmc.mm_model:
            model_llmc.mm_model.eval()
        else:
            model_llmc.model.eval()

        result = evaluate_task(
            task_config,
            model_llmc.mm_model if model_llmc.mm_model else model_llmc.model,
            model_llmc.processor,
            None,
            res_acc,
            n_samples=self.eval_limit,
            is_acc=True,
        )
        res_acc[f"{task_name}_result"] = result

        if self.inference_per_block:
            for h in handles:
                h.remove()

        if model_llmc.mm_model:
            model_llmc.mm_model.cpu()
        else:
            model_llmc.model.cpu()

        gc.collect()
        torch.cuda.empty_cache()
        return res_acc
