import copy
import os

from loguru import logger

from llmc.compression.quantization.constant import MEASUREMENT
from llmc.eval import (
    AccuracyEval,
    CustomGenerate,
    DecodePerplexityEval,
    HumanEval,
    PerplexityEval,
    DebugEval,
    TokenConsistencyEval,
    KLDivergenceEval,
    VideoGenerateEval,
    VQAEval,
    V4Eval,
    MSEEval,
)
from llmc.utils import deploy_all_modality
from llmc.compression.quantization.module_utils import StatFakeQuantLinear
from llmc.compression.quantization.measure import MeasurePrinter, MeasureRecorder

def get_eval_list(model, config):
    eval_list = []
    if int(os.environ["RANK"]) == 0:
        if "eval" in config:
            if "type" in config.eval and config.eval.type == "decode_ppl":
                if "pretrain" in config.eval.eval_pos:
                    raise ValueError(
                        "Unsupported: Evaluating decode_ppl with a pretrained model. "
                    )
                    # Pretrained models do not use key-value caching.
                    # Please use a transformed model to evaluate decode_ppl
                    # for the original model.

            if not isinstance(config.eval, list):
                eval_config_list = [config.eval]
            else:
                eval_config_list = config.eval
            for eval_config in eval_config_list:
                config_tmp = copy.deepcopy(config)
                config_tmp.eval = eval_config
                if "type" not in config_tmp.eval:
                    config_tmp.eval["type"] = "ppl"
                if "eval" in config_tmp and len(config_tmp.eval.eval_pos):
                    name_list = (
                        config_tmp.eval.name
                        if not isinstance(config_tmp.eval.name, str)
                        else [config_tmp.eval.name]
                    )
                    for name in name_list:
                        config_for_eval = copy.deepcopy(config_tmp)
                        config_for_eval.eval.name = name
                        if len(name_list) != 1:  # eval multi datasets
                            config_for_eval.eval.path = os.path.join(
                                config_tmp.eval.path, name
                            )
                        if "type" not in config_tmp.eval:
                            config_tmp.eval.type == "ppl"
                        if config_tmp.eval.type == "acc":
                            eval_class = AccuracyEval(config_for_eval)
                        elif config_tmp.eval.type == "vqa":
                            eval_class = VQAEval(config_for_eval)
                        elif config_tmp.eval.type == "v4":
                            eval_class = V4Eval(config_for_eval)
                        elif (
                            config_tmp.eval.type == "code"
                            and config_tmp.eval.name == "human_eval"
                        ):
                            eval_class = HumanEval(model, config_for_eval)
                        elif config_tmp.eval.type == "generate_only":
                            eval_class = CustomGenerate(model, config_for_eval)
                        elif config_tmp.eval.type == "token_acc":
                            eval_class = TokenConsistencyEval(model, config_for_eval)
                        elif config_tmp.eval.type == "kl":
                            eval_class = KLDivergenceEval(model, config_for_eval)
                        elif config_tmp.eval.type == "mse":
                            eval_class = MSEEval(model, config_for_eval)
                        elif config_tmp.eval.type == "ppl":
                            eval_class = PerplexityEval(model, config_for_eval)
                        elif config_tmp.eval.type == "debug":
                            eval_class = DebugEval(model, config_for_eval)
                        elif config_tmp.eval.type == "decode_ppl":
                            eval_class = DecodePerplexityEval(model, config_for_eval)
                        elif config_tmp.eval.type == "video_gen":
                            eval_class = VideoGenerateEval(model, config_for_eval)
                        else:
                            raise ValueError(
                                f"Unsupported eval type: {config_tmp.eval.type}"
                            )
                        eval_list.append((eval_class, config_for_eval))
    return eval_list

def print_debug_info(res_measure, method=MEASUREMENT):
    if res_measure:
        method_str = "MEASUREMENT"
        if method == "snr":
            method_str = "NOISE:SIGNAL POWER RATIO"
            order="large_to_small"
        if method == "cosine":
            method_str = "COSINE SIMILARITY"
            order="small_to_large"
        if method == "mse":
            method_str = "MSE LOSS(UNSCALED)"
            order="large_to_small"
        if method == "kl":
            method_str = "KL DIVERGENCE"
            order="large_to_small"
        MeasurePrinter(
                    res_measure,
                    order=order,
                    measure=method_str,
                    percentage=method in {"snr", "cosine"},
                ).print()

def eval_model(model, blockwise_opts, eval_list, eval_pos):
    global global_step
    ret = None
    if int(os.environ["RANK"]) == 0:
        do_eval = False
        for _, config_for_eval in eval_list:
            if eval_pos in config_for_eval.eval.eval_pos:
                do_eval = True

        def eval_func(r=None):
            for eval_class, config_for_eval in eval_list:
                if eval_pos in config_for_eval.eval.eval_pos:
                    res = eval_class.eval(model, eval_pos)
                    eval_name = config_for_eval.eval.type
                    dataset_name = config_for_eval.eval.name
                    logger.info(f"EVAL: {eval_name} on {dataset_name} is {res}")
                    if r:
                        ret[eval_pos][dataset_name] = res

        def add_hook_for_interested_layers_step1(interested_output, interested_layers):
            def hook(model, input, output):
                if output.shape[1] > 1:
                    interested_output[id(model)].append(output.cpu())
            hooks = []
            for layer in interested_layers:
                hooks.append(layer.register_forward_hook(hook))
            return hooks

        def add_hook_for_interested_layers_step2(recorder, interested_output, interested_layers):
            def hook(model, input, output):
                global global_step
                if global_step >= len(interested_output[id(model)]) or output.shape[1] == 1:
                    return
                recorder.update(y_pred=output, y_real=interested_output[id(model)][global_step].to(output.device))
                global_step += 1
            hooks = []
            for layer in interested_layers:
                hooks.append(layer.register_forward_hook(hook))
            return hooks

        if do_eval:
            ret = {eval_pos: {}}
            if eval_pos == "transformed":
                deploy_all_modality(blockwise_opts, "origin_float")
            elif eval_pos in ["fake_quant", "fake_quant_wo_kv"]:
                deploy_all_modality(blockwise_opts, eval_pos)
            elif eval_pos in [
                "stat_fake_quant_qdq", 
                "stat_fake_quant_graph",
                "stat_fake_quant_subset",
                "stat_fake_quant_block",
                ]:
                deploy_all_modality(blockwise_opts, "stat_fake_quant")
                quantable_modules = {}
                for name, module in model.model.named_modules():
                    if type(module) == StatFakeQuantLinear:
                        quantable_modules[name] = module
                if eval_pos == "stat_fake_quant_graph":
                    for name, module in quantable_modules.items():
                        quantable_modules[name] = module
                        module.graph_stat_step[0] = 1
                        module.quant_status[0] = 1
                elif eval_pos == "stat_fake_quant_qdq":
                    for name, module in quantable_modules.items():
                        quantable_modules[name] = module
                        module.op_stat_status[0] = 1
                        module.quant_status[0] = 1

            eval_func(ret)
            if eval_pos == "stat_fake_quant_qdq":
                for name, module in quantable_modules.items():
                    quantable_modules[name] = module
                    module.op_stat_status[0] = 0
                    module.quant_status[0] = 0
                res_measure_qdq_w = {}
                res_measure_qdq_a = {}
                res_measure_qdq_o = {}
                for name, module in quantable_modules.items():
                    if module.recorder_qdq_a.num_of_elements > 0:
                        res_measure_qdq_a[name] = module.recorder_qdq_a.measure
                        module.recorder_qdq_a.clear()
                    if module.recorder_qdq_w.num_of_elements > 0:
                        res_measure_qdq_w[name] = module.recorder_qdq_w.measure
                        module.recorder_qdq_w.clear()
                    if module.recorder_qdq_o.num_of_elements > 0:
                        res_measure_qdq_o[name] = module.recorder_qdq_o.measure
                        module.recorder_qdq_o.clear()
                print("="*10 + "OP analysis (Act Input)" + "="*10)
                print_debug_info(res_measure_qdq_a)
                print("="*10 + "OP analysis (Weight)" + "="*10)
                print_debug_info(res_measure_qdq_w)
                print("="*10 + "OP analysis (Act Output)" + "="*10)
                print_debug_info(res_measure_qdq_o)
        
            elif eval_pos == "stat_fake_quant_graph":
                for name, module in quantable_modules.items():
                    module.graph_stat_step[0] = 2
                    module.quant_status[0] = 0
                eval_func()
                res_measure = {}
                for name, module in quantable_modules.items():
                    res_measure[name] = module.recorder_graph.measure
                    module.tmp_qdq = []
                    module.step_cnt = 0
                    module.graph_stat_step[0] = 0
                    module.quant_status[0] = 0
                    
                print("="*10 + "Graph diff analysis" + "="*10)
                print_debug_info(res_measure)
            elif eval_pos in ["stat_fake_quant_subset", "stat_fake_quant_block"]:
                interested_layers = model.get_interested_layers()
                interested_output = {id(mod): [] for mod in interested_layers}
                hooks = add_hook_for_interested_layers_step1(interested_output, interested_layers)
                eval_func()
                for hook in hooks:
                    hook.remove()
                res_measure = {}
                recorders = {}
                if eval_pos == "stat_fake_quant_subset":
                    subset_names = model.get_quantable_subset_names()
                    for subset_name in subset_names:
                        recorders[subset_name] = MeasureRecorder(measurement=MEASUREMENT, flatten_start_dim=-1)
                        for name, module in quantable_modules.items():
                            if name.endswith(subset_name):
                                module.quant_status[0] = 1
                            else:
                                module.quant_status[0] = 0
                        hooks = add_hook_for_interested_layers_step2(recorders[subset_name], interested_output, interested_layers)
                        global_step = 0
                        eval_func()
                        for hook in hooks:
                            hook.remove()
                        res_measure[subset_name] = recorders[subset_name].measure

                elif eval_pos == "stat_fake_quant_block":
                    quantable_modules_set = set(quantable_modules.values())
                    blocks = model.get_blocks()
                    for block_idx, block in enumerate(blocks):
                        recorders[block_idx] = MeasureRecorder(measurement=MEASUREMENT, flatten_start_dim=-1)
                        for name, module in block.named_modules():
                            if module in quantable_modules_set:
                                module.quant_status[0] = 1

                        hooks = add_hook_for_interested_layers_step2(recorders[block_idx], interested_output, interested_layers)
                        global_step = 0
                        eval_func()
                        for hook in hooks:
                            hook.remove()
                        for name, module in block.named_modules():
                            if module in quantable_modules_set:
                                module.quant_status[0] = 0
                        res_measure["block_{}".format(block_idx)] = recorders[block_idx].measure
                for name, module in quantable_modules.items():
                    module.quant_status[0] = 0
                print("="*10 + "Layerwise err analysis" + "="*10)
                print_debug_info(res_measure)
    
    return ret

