import copy
import json
import os
import re
import warnings
from collections import UserDict
from collections.abc import Mapping, Sized
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from trl.trainer.utils import DataCollatorForCompletionOnlyLM, DataCollatorForLanguageModeling

from accelerate.utils.operations import honor_type
from transformers.tokenization_utils_base import (
    BatchEncoding,
    PaddingStrategy,
    EncodedInput,
    TensorType,
    logger,
)

from transformers.utils import (
    PaddingStrategy,
    TensorType,
    is_tf_tensor,
    is_torch_tensor,
    to_py_obj,
)

def pad(
    self,
    encoded_inputs: Union[
        BatchEncoding,
        List[BatchEncoding],
        Dict[str, EncodedInput],
        Dict[str, List[EncodedInput]],
        List[Dict[str, EncodedInput]],
    ],
    padding: Union[bool, str, PaddingStrategy] = True,
    max_length: Optional[int] = None,
    pad_to_multiple_of: Optional[int] = None,
    padding_side: Optional[str] = None,
    return_attention_mask: Optional[bool] = None,
    return_tensors: Optional[Union[str, TensorType]] = None,
    verbose: bool = True,
) -> BatchEncoding:
    if self.__class__.__name__.endswith("Fast"):
        if not self.deprecation_warnings.get("Asking-to-pad-a-fast-tokenizer", False):
            logger.warning_advice(
                f"You're using a {self.__class__.__name__} tokenizer. Please note that with a fast tokenizer,"
                " using the `__call__` method is faster than using a method to encode the text followed by a call"
                " to the `pad` method to get a padded encoding."
            )
            self.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

    # If we have a list of dicts, let's convert it in a dict of lists
    # We do this to allow using this method as a collate_fn function in PyTorch Dataloader
    if isinstance(encoded_inputs, (list, tuple)) and isinstance(encoded_inputs[0], Mapping):
        encoded_inputs = {key: [example.get(key) for example in encoded_inputs] for key in encoded_inputs[0].keys()} # TODO get

    # The model's main input name, usually `input_ids`, has been passed for padding
    if self.model_input_names[0] not in encoded_inputs:
        raise ValueError(
            "You should supply an encoding or a list of encodings to this method "
            f"that includes {self.model_input_names[0]}, but you provided {list(encoded_inputs.keys())}"
        )

    required_input = encoded_inputs[self.model_input_names[0]]

    if required_input is None or (isinstance(required_input, Sized) and len(required_input) == 0):
        if return_attention_mask:
            encoded_inputs["attention_mask"] = []
        return encoded_inputs

    # If we have PyTorch/TF/NumPy tensors/arrays as inputs, we cast them as python objects
    # and rebuild them afterwards if no return_tensors is specified
    # Note that we lose the specific device the tensor may be on for PyTorch

    first_element = required_input[0]
    if isinstance(first_element, (list, tuple)):
        # first_element might be an empty list/tuple in some edge cases so we grab the first non empty element.
        for item in required_input:
            if len(item) != 0:
                first_element = item[0]
                break
    # At this state, if `first_element` is still a list/tuple, it's an empty one so there is nothing to do.
    if not isinstance(first_element, (int, list, tuple)):
        if is_tf_tensor(first_element):
            return_tensors = "tf" if return_tensors is None else return_tensors
        elif is_torch_tensor(first_element):
            return_tensors = "pt" if return_tensors is None else return_tensors
        elif isinstance(first_element, np.ndarray):
            return_tensors = "np" if return_tensors is None else return_tensors
        else:
            raise ValueError(
                f"type of {first_element} unknown: {type(first_element)}. "
                "Should be one of a python, numpy, pytorch or tensorflow object."
            )

        for key, value in encoded_inputs.items():
            encoded_inputs[key] = to_py_obj(value)

    # Convert padding_strategy in PaddingStrategy
    padding_strategy, _, max_length, _ = self._get_padding_truncation_strategies(
        padding=padding, max_length=max_length, verbose=verbose
    )

    required_input = encoded_inputs[self.model_input_names[0]]
    if required_input and not isinstance(required_input[0], (list, tuple)):
        encoded_inputs = self._pad(
            encoded_inputs,
            max_length=max_length,
            padding_strategy=padding_strategy,
            pad_to_multiple_of=pad_to_multiple_of,
            padding_side=padding_side,
            return_attention_mask=return_attention_mask,
        )
        return BatchEncoding(encoded_inputs, tensor_type=return_tensors)

    batch_size = len(required_input)
    assert all(len(v) == batch_size for v in encoded_inputs.values()), (
        "Some items in the output dictionary have a different batch size than others."
    )

    if padding_strategy == PaddingStrategy.LONGEST:
        max_length = max(len(inputs) for inputs in required_input)
        padding_strategy = PaddingStrategy.MAX_LENGTH

    batch_outputs = {}
    for i in range(batch_size):
        inputs = {k: v[i] for k, v in encoded_inputs.items()}
        outputs = self._pad(
            inputs,
            max_length=max_length,
            padding_strategy=padding_strategy,
            pad_to_multiple_of=pad_to_multiple_of,
            padding_side=padding_side,
            return_attention_mask=return_attention_mask,
        )

        for key, value in outputs.items():
            if value is not None: # TODO 
                if key not in batch_outputs:
                    batch_outputs[key] = []
                batch_outputs[key].append(value)

    return BatchEncoding(batch_outputs, tensor_type=return_tensors)

emp_t = torch.tensor([])

def concatenate(data, dim=0):
    """
    Recursively concatenate the tensors in a nested list/tuple/dictionary of lists of tensors with the same shape.

    Args:
        data (nested list/tuple/dictionary of lists of tensors `torch.Tensor`):
            The data to concatenate.
        dim (`int`, *optional*, defaults to 0):
            The dimension on which to concatenate.

    Returns:
        The same data structure as `data` with all the tensors concatenated.
    """
    if isinstance(data[0], (tuple, list)):
        return honor_type(data[0], (concatenate([d[i] for d in data], dim=dim) for i in range(len(data[0]))))
    elif isinstance(data[0], Mapping):
        keys = set(k for d in data for k in d.keys())
        return type(data[0])({k: concatenate([d.get(k, emp_t) for d in data], dim=dim) for k in keys})
    elif not isinstance(data[0], torch.Tensor):
        raise TypeError(f"Can only concatenate tensors but got {type(data[0])}")
    return torch.cat(data, dim=dim)

class CustomDataCollatorForCompletionOnlyLM(DataCollatorForCompletionOnlyLM):

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        batch = DataCollatorForLanguageModeling.torch_call(self, examples)

        if self.instruction_template is None:
            for i in range(len(examples)):
                response_token_ids_start_idx = None

                for idx in np.where(batch["labels"][i] == self.response_token_ids[0])[0]:
                    # `response_token_ids` is `'### Response:\n'`, here we are just making sure that the token IDs match
                    if (
                        self.response_token_ids
                        == batch["labels"][i][idx : idx + len(self.response_token_ids)].tolist()
                    ):
                        response_token_ids_start_idx = idx

                if response_token_ids_start_idx is None:
                    pass # TODO do not ignore
                    # warnings.warn(
                    #     f"Could not find response key `{self.response_template}` in the "
                    #     f'following instance: {self.tokenizer.decode(batch["input_ids"][i])} '
                    #     f"This instance will be ignored in loss calculation. "
                    #     f"Note, if this happens often, consider increasing the `max_seq_length`."
                    # )
                    # batch["labels"][i, :] = self.ignore_index
                else:
                    response_token_ids_end_idx = response_token_ids_start_idx + len(self.response_token_ids)

                    # Make pytorch loss function ignore all tokens up through the end of the response key
                    batch["labels"][i, :response_token_ids_end_idx] = self.ignore_index

        else:
            for i in range(len(examples)):
                response_token_ids_idxs = []
                human_token_ids_idxs = []

                for assistant_idx in np.where(batch["labels"][i] == self.response_token_ids[0])[0]:
                    # find the indexes of the start of a response.
                    if (
                        self.response_token_ids
                        == batch["labels"][i][assistant_idx : assistant_idx + len(self.response_token_ids)].tolist()
                    ):
                        response_token_ids_idxs.append(assistant_idx + len(self.response_token_ids))

                if len(response_token_ids_idxs) == 0:
                    warnings.warn(
                        f"Could not find response key `{self.response_template}` in the "
                        f'following instance: {self.tokenizer.decode(batch["input_ids"][i])} '
                        f"This instance will be ignored in loss calculation. "
                        f"Note, if this happens often, consider increasing the `max_seq_length`."
                    )
                    batch["labels"][i, :] = self.ignore_index

                human_token_ids = self.instruction_token_ids
                for human_idx in np.where(batch["labels"][i] == human_token_ids[0])[0]:
                    # find the indexes of the start of a human answer.
                    if human_token_ids == batch["labels"][i][human_idx : human_idx + len(human_token_ids)].tolist():
                        human_token_ids_idxs.append(human_idx)

                if len(human_token_ids_idxs) == 0:
                    warnings.warn(
                        f"Could not find instruction key `{self.instruction_template}` in the "
                        f'following instance: {self.tokenizer.decode(batch["input_ids"][i])} '
                        f"This instance will be ignored in loss calculation. "
                        f"Note, if this happens often, consider increasing the `max_seq_length`."
                    )
                    batch["labels"][i, :] = self.ignore_index

                if (
                    len(human_token_ids_idxs) > 0
                    and len(response_token_ids_idxs) > 0
                    and human_token_ids_idxs[0] > response_token_ids_idxs[0]
                ):
                    human_token_ids_idxs = [0] + human_token_ids_idxs

                for idx, (start, end) in enumerate(zip(human_token_ids_idxs, response_token_ids_idxs)):
                    # Make pytorch loss function ignore all non response tokens
                    if idx != 0:
                        batch["labels"][i, start:end] = self.ignore_index
                    else:
                        batch["labels"][i, :end] = self.ignore_index

                if len(response_token_ids_idxs) < len(human_token_ids_idxs):
                    batch["labels"][i, human_token_ids_idxs[-1] :] = self.ignore_index

        if self.padding_free:
            # remove padding, `attention_mask` and add `position_ids`
            attn_mask = batch.pop("attention_mask")
            batch["input_ids"] = batch["input_ids"][attn_mask.bool()].unsqueeze(0)
            batch["position_ids"] = attn_mask.cumsum(1)[attn_mask.bool()].unsqueeze(0) - 1
            batch["labels"] = batch["labels"][attn_mask.bool()].unsqueeze(0)
            batch["labels"][batch["position_ids"] == 0] = self.ignore_index

        return batch
