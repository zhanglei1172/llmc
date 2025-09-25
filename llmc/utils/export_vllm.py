import json


def update_vllm_quant_config(
    model,
    config,
    save_quant_path,
    vllm_quant_method='compressed-tensors',

):
    modaility = config.quant.get("modaility", "language")
    _quant_config = config.quant[modaility] if modaility in config.quant else config.quant
    need_pack = _quant_config.weight.get('need_pack', False)
    weight_quant_type = _quant_config.weight.get('quant_type', 'int-quant')
    if 'act' in _quant_config:
        act_quant_type = _quant_config.act.get('quant_type', 'int-quant')
        assert act_quant_type == weight_quant_type
    else:
        act_quant_type = None
    if act_quant_type is not None and act_quant_type == 'float-quant':
        if _quant_config.act.get('static', False):
            quant_config = {
                'activation_scheme': 'static',
                'ignored_layers': [
                    model.skip_layer_name()
                ],
                'quant_method': 'fp8'
            }
            config_file = save_quant_path + '/config.json'
            with open(config_file, 'r') as file:
                config_vllm = json.load(file)
            config_vllm['quantization_config'] = quant_config
            with open(config_file, 'w') as file:
                json.dump(config_vllm, file, indent=4)
            return
        elif _quant_config.weight.get('granularity', 'per_block'):
            quant_config = {
                'activation_scheme': 'dynamic',
                'fmt': 'e4m3',
                'quant_method': 'fp8',
                'weight_block_size': [
                    _quant_config.weight.block_size,
                    _quant_config.weight.block_size
                ]
            }
            config_file = save_quant_path + '/config.json'
            with open(config_file, 'r') as file:
                config_vllm = json.load(file)
            config_vllm['quantization_config'] = quant_config
            with open(config_file, 'w') as file:
                json.dump(config_vllm, file, indent=4)
            return
        else:
            vllm_quant_format = 'float-quantized'
            quant_type = 'float'
            w_num_bits = 8
            a_num_bits = 8
    elif need_pack:
        vllm_quant_format = 'pack-quantized'
        quant_type = 'int'
        w_num_bits = _quant_config.weight.bit
    elif weight_quant_type == 'float-quant':
        vllm_quant_format = 'float-quantized'
        quant_type = 'float'
        w_num_bits = 8
    else:
        vllm_quant_format = 'int-quantized'
        quant_type = 'int'
        w_num_bits = _quant_config.weight.bit
        if 'act' in _quant_config:
            a_num_bits = _quant_config.act.bit

    if _quant_config.weight.granularity == 'per_group':
        group_size = _quant_config.weight.group_size
    else:
        group_size = None

    if 'act' in _quant_config and 'static' in _quant_config.act:
        dynamic = not _quant_config.act.static
    else:
        dynamic = True

    quant_config = {
        'config_groups': {
            'group_0': {
                'targets': ['Linear'],  # Now only support "Linear".
                'input_activations': {
                    'dynamic': dynamic,
                    'group_size': None,   # Don't support activations per-group quant.
                    'num_bits': a_num_bits,
                    'observer': 'minmax',
                    'observer_kwargs': {},
                    'strategy': 'token'
                                if _quant_config.act.granularity == 'per_token'
                                else 'tensor',
                    'symmetric': _quant_config.act.symmetric,
                    'type': quant_type
                } if 'act' in _quant_config else None,
                'weights': {
                    'dynamic': False,
                    'group_size': group_size,
                    'num_bits': w_num_bits,
                    'observer': 'minmax',  # Now only support "minmax".
                    'observer_kwargs': {},
                    'strategy': (
                        'group'
                        if _quant_config.weight.granularity == 'per_group'
                        else 'channel'
                    ),
                    'symmetric': _quant_config.weight.symmetric,
                    'type': quant_type,
                },
            }
        },
        'format': vllm_quant_format,
        'ignore': model.skip_layer_name(),
        'quant_method': vllm_quant_method,
    }

    config_file = save_quant_path + '/config.json'
    with open(config_file, 'r') as file:
        config_vllm = json.load(file)
    if weight_quant_type == 'int-quant' and 'quantization_config' in config_vllm:
        del config_vllm['quantization_config']
    config_vllm['compression_config'] = quant_config
    with open(config_file, 'w') as file:
        json.dump(config_vllm, file, indent=4)
