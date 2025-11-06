import torch
from loguru import logger

from llmc.utils.registry_factory import MODEL_REGISTRY
from llmc.utils.utils import patch_attr

from .eval_base import BaseEval


class KLDivergenceEval(BaseEval):
    def __init__(self, model, config):
        super().__init__(model, config)
        self.ref_path = self.eval_cfg.get('ref_path', self.config.model.path)

    @torch.no_grad()
    def eval_func(self, model, testenc, seq_len, bs, eval_pos):
        handles_origin = []
        with patch_attr(self.config.model, 'path', self.ref_path):
            model_origin = MODEL_REGISTRY[self.config.model.type](self.config)
        if self.inference_per_block:
            handles_origin = self.register_hooks(model_origin)
        else:
            if model_origin.mm_model:
                model_origin.mm_model.cuda()
            else:
                model_origin.model.cuda()

        if model_origin.mm_model:
            model_origin.mm_model.eval()
        else:
            model_origin.model.eval()

        testenc = testenc.input_ids
        nsamples = testenc.numel() // seq_len

        kl_ret = 0
        total_tokens = 0

        # Loop through each batch
        for i in range(0, nsamples, bs):
            logger.info(f'index : {(i + 1) // bs}/{nsamples // bs}')
            # Calculate end index
            j = min(i + bs, nsamples)

            # Prepare inputs and move to gpu
            inputs = testenc[:, (i * seq_len): (j * seq_len)].cuda()
            inputs = inputs.reshape(j - i, seq_len)

            # Forward pass through the models
            logits1 = model_origin.model(inputs).logits
            logits2 = model.model(inputs).logits
            model.reset_kv()

            kl = torch.nn.functional.kl_div(
                torch.log_softmax(logits2, dim=-1),
                torch.softmax(logits1, dim=-1),
                reduction='batchmean',
            )
            total_tokens += (j - i) * seq_len

            kl_ret += kl.item() * (j - i) * seq_len

        # Calculate average KL divergence
        kl_ret = kl_ret / total_tokens if total_tokens > 0 else 0.0

        # Empty CUDA cache to save memory
        testenc.cpu()
        torch.cuda.empty_cache()

        if model_origin.mm_model:
            model_origin.mm_model.cpu()
        else:
            model_origin.model.cpu()

        if self.inference_per_block:
            for h in handles_origin:
                h.remove()

        return kl_ret
