from itertools import chain
from typing import Optional

import torch
import torch.nn as nn

from allennlp.modules.augmented_lstm import BiAugmentedLstm
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder, PytorchSeq2SeqWrapper


@Seq2SeqEncoder.register("biaugmented_lstm")
class BiAugmentedLstmSeq2SeqEncoder(PytorchSeq2SeqWrapper):
    """
    Registered as a `Seq2SeqEncoder` with name "biaugmented_lstm".
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        recurrent_dropout_probability: float = 0.0,
        bidirectional: bool = True,
        padding_value: float = 0.0,
        use_highway: bool = True,
        stateful: bool = False
    ) -> None:
        module = BiAugmentedLstm(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            recurrent_dropout_probability=recurrent_dropout_probability,
            bidirectional=bidirectional,
            padding_value=padding_value,
            use_highway=use_highway,
        )
        super().__init__(module=module, stateful=stateful)


@Seq2SeqEncoder.register("pg_lstm")
class ParameterGenerationLstmSeq2SeqEncoder(PytorchSeq2SeqWrapper):
    """
    Parameter Generation LSTM, need bucketed batch inputs.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        embedding_dim: int,
        num_layers: int = 1,
        bias: bool = True,
        dropout: float = 0.0,
        bidirectional: bool = True,
        stateful: bool = False
    ):
        module = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional
        )
        # delete original parameters.
        flat_sizes = list()
        for name in module._flat_weights_names:
            flat_sizes.append(getattr(module, name).size())
            delattr(module, name)
        module._flat_weights = list()

        super().__init__(module=module, stateful=stateful)
        # register PGN parameters.
        for name, size in zip(module._flat_weights_names, flat_sizes):
            param = nn.Parameter(torch.Tensor(*size, embedding_dim))
            setattr(self, name, param)

        self.reset_parameters()

    def reset_parameters(self):
        for name in self._module._flat_weights_names:
            param = getattr(self, name)
            nn.init.normal_(param, std=1e-3)

    def forward(
        self, inputs: torch.Tensor, mask: torch.BoolTensor, hidden_state: torch.Tensor = None,
        embedding: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # set parameters
        self.set_parameters(embedding)
        # compute output with generated LSTM.
        output = super().forward(inputs, mask, hidden_state)
        # reset parameters
        self.set_parameters()
        return output

    def set_parameters(self, embedding: Optional[torch.Tensor] = None):
        if embedding is None:
            self._module._flat_weights = list()
        elif embedding.ndim > 1:
            raise RuntimeError("embedding.ndim must be 1.")
        else:
            self._module._flat_weights = [
                getattr(self, n).matmul(embedding) for n in self._module._flat_weights_names
            ]
            self._module.flatten_parameters()


if __name__ == "__main__":
    device=torch.device('cuda:1')
    pgn_lstm=ParameterGenerationLstmSeq2SeqEncoder(
        input_size=768,
        hidden_size=300,
        embedding_dim=8,
        num_layers=2,
        bidirectional=True).to(device)
    l_embedding=torch.Tensor([1,1,1,1,1,1,1,1]).to(device=device)
    pgn_lstm.forward(inputs=None,mask=None,embedding=l_embedding)