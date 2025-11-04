import torch
from loguru import logger

from llmc.utils.registry_factory import MODEL_REGISTRY
from llmc.utils.utils import patch_attr

from .eval_base import BaseEval


class SNREval(BaseEval):
    def __init__(self, model, config):
        super().__init__(model, config)
        self.ref_path = self.eval_cfg.get('ref_path', self.config.model.path)

    @torch.no_grad()
    def eval_func(self, model, testenc, seq_len, bs, eval_pos):
        from llmc.compression.quantization.measure import torch_snr_error
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
        _res = {}
        hooks = []
        def add_hook_for_head(layer, name):
            def hook(module, input, output):
                _res[name] = (input[0].float().cpu())
            hooks.append(layer.register_forward_hook(hook))
        add_hook_for_head(model_origin.get_head_layers()[0], "y_real")
        add_hook_for_head(model.get_head_layers()[0], "y_pred")

        if model_origin.mm_model:
            model_origin.mm_model.eval()
        else:
            model_origin.model.eval()

        testenc = testenc.input_ids
        nsamples = testenc.numel() // seq_len

        snr_ret = 0
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
            snr = torch_snr_error(y_pred=_res["y_pred"], y_real=_res["y_real"]).item()

            total_tokens += (j - i) * seq_len

            snr_ret += snr * (j - i) * seq_len

        # Calculate average KL divergence
        snr_ret = snr_ret / total_tokens if total_tokens > 0 else 0.0
        for h in hooks:
            h.remove()
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

        return snr_ret
