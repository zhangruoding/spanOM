"""
Adapter transformers for AllenNLP.
Parameter-Efficient Transfer Learning for NLP. https://arxiv.org/abs/1902.00751
"""

from typing import Optional, Dict, Any, Union, List

import torch.nn as nn

from transformers import BertModel, ElectraModel, RobertaModel

from allennlp.common.checks import ConfigurationError
from allennlp.modules.token_embedders import TokenEmbedder, PretrainedTransformerEmbedder
from spanom.bert_embedder.adapter import Adapter, AdapterBertOutput


@TokenEmbedder.register("adapter_transformer")
class AdapterTransformerEmbedder(PretrainedTransformerEmbedder):
    """
    目前只针对 *BERT 结构，插入adapter.
    """
    def __init__(
        self,
        model_name: str,
        *,
        adapter_layers: int = 12,
        adapter_kwargs: Optional[Dict[str, Any]] = None,
        external_param: Union[bool, List[bool]] = False,
        max_length: int = None,
        gradient_checkpointing: Optional[bool] = None,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        transformer_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            model_name,
            max_length=max_length,
            train_parameters=False,
            last_layer_only=True,
            gradient_checkpointing=gradient_checkpointing,
            tokenizer_kwargs=tokenizer_kwargs,
            transformer_kwargs=transformer_kwargs
        )

        self.adapters = insert_adapters(adapter_layers, adapter_kwargs, external_param, self.transformer_model)
        self.adapter_layers = adapter_layers
        self.adapter_kwargs = adapter_kwargs



def insert_adapters(
    adapter_layers: int, adapter_kwargs: Dict[str, Any],
    external_param: Union[bool, List[bool]], transformer_model: BertModel
) -> nn.ModuleList:
    """
    初始化 adapters, 插入到 BERT, 并返回 adapters. 目前只支持 *BERT 结构!

    # Parameters

    adapter_layers : `int`, required.
        从 BERT 最后一层开始，多少层插入adapter。
    adapter_kwargs : `Dict`, required.
        初始化 `Adapter` 的参数。
    external_param : `Union[bool, List[bool]]`
        adapter 的参数是否留空以便外部注入。
    transformer_model : `BertModel`
        预训练模型。

    # Returns

    adapters_groups : `nn.ModuleList`, required.
        所插入的所有 adapter, 用于绑定到模型中。
    """
    if not isinstance(transformer_model, (BertModel, ElectraModel, RobertaModel)):
        raise ConfigurationError("目前只支持 *BERT 结构")

    if isinstance(external_param, bool):
        param_place = [external_param for _ in range(adapter_layers)]
    elif isinstance(external_param, list):
        param_place = [False for _ in range(adapter_layers)]
        for i, e in enumerate(external_param, 1):
            param_place[-i] = e
    else:
        raise ConfigurationError("wrong type of external_param!")

    adapter_kwargs.update(in_features=transformer_model.config.hidden_size)
    adapters_groups = nn.ModuleList([
        nn.ModuleList([
            Adapter(external_param=param_place[i], **adapter_kwargs),
            Adapter(external_param=param_place[i], **adapter_kwargs)
        ]) for i in range(adapter_layers)
    ])

    for i, adapters in enumerate(adapters_groups, 1):
        layer = transformer_model.encoder.layer[-i]
        layer.output = AdapterBertOutput(layer.output, adapters[0])
        layer.attention.output = AdapterBertOutput(layer.attention.output, adapters[1])

    return adapters_groups

if __name__ == "__main__":
    transformer_model="./bert-base-cased"

    matched_embedder=AdapterTransformerEmbedder(model_name=transformer_model,
                        max_length=512,
                        adapter_layers=10,
                        adapter_kwargs={"adapter_size": 32,
                        "bias": True},
                        tokenizer_kwargs={"do_lower_case": False})
    print(matched_embedder)