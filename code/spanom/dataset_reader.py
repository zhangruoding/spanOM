import codecs
from typing import Dict, Iterable,  Optional
import json

import torch
from spanom.reader import srl_sentence
from allennlp.data import DatasetReader, Instance, Field, Token,Vocabulary
from allennlp.data.dataset_readers.dataset_utils import enumerate_spans
from allennlp.data.fields import TextField, SpanField, SequenceLabelField, MetadataField, ListField, TensorField
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.token_indexers import (
    PretrainedTransformerMismatchedIndexer,
)
from allennlp.data.data_loaders import MultiProcessDataLoader
from allennlp.data.samplers.bucket_batch_sampler import BucketBatchSampler
import torch.nn as nn


@DatasetReader.register("mpqa2")
class MPAQ2Reader(DatasetReader):
    def __init__(
        self,
        token_indexers: Dict[str, TokenIndexer],
        max_span_width: Optional[int] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.token_indexers = token_indexers
        self.max_span_width = max_span_width

    def text_to_instance(self, sentence: srl_sentence,language_id:int) -> Instance:  # type: ignore
        assert isinstance(sentence, srl_sentence)
        emo_dict=sentence.emo
        tokens = sentence.sentences
        text_field = TextField([Token(t) for t in tokens], self.token_indexers)
        dse_starts,dse_ends=sentence.tokenize_dse_spans()
        arg_starts,arg_ends=sentence.tokenize_arg_spans()
        elements_dse=[(s,e) for s,e in zip(dse_starts,dse_ends)]
        elements_arg=[(s,e) for s,e in zip(arg_starts,arg_ends)]

        spans, element_labels_role,element_labels_expression= list(), list(),list()
        span2id=dict()
        for start, end in enumerate_spans(tokens, max_span_width=self.max_span_width):
            # span 是包含头尾的
            span = (start, end)
            if span in elements_dse:
                element_labels_expression.append(emo_dict[span])
            else:
                element_labels_expression.append(0)
            if span in elements_arg:
                element_labels_role.append(1)
            else:
                element_labels_role.append(0)
            span2id[span] = len(spans)
            spans.append(SpanField(start, end, text_field))

        span_field = ListField(spans)
        element_role_label_field = SequenceLabelField(
            element_labels_role, span_field, label_namespace="element_labels")
        element_exp_label_field = SequenceLabelField(
            element_labels_expression, span_field, label_namespace="element_labels")
        element_labels_dse_start,element_labels_dse_end,element_labels_arg_start,element_labels_arg_end= list(), list(),list(),list()
        for i,_ in enumerate(tokens):
            if i in dse_starts:
                element_labels_dse_start.append(1)
            else:
                element_labels_dse_start.append(0)
            if i in dse_ends:
                element_labels_dse_end.append(1)
            else:
                element_labels_dse_end.append(0)
            if i in arg_starts:
                element_labels_arg_start.append(1)
            else:
                element_labels_arg_start.append(0)
            if i in arg_ends:
                element_labels_arg_end.append(1)
            else:
                element_labels_arg_end.append(0)
        element_dse_start_label_field = SequenceLabelField(
            element_labels_dse_start, text_field, label_namespace="element_labels")
        element_dse_end_label_field = SequenceLabelField(
            element_labels_dse_end, text_field, label_namespace="element_labels")
        element_arg_start_label_field = SequenceLabelField(
            element_labels_arg_start, text_field, label_namespace="element_labels")
        element_arg_end_label_field = SequenceLabelField(
            element_labels_arg_end, text_field, label_namespace="element_labels")
        
        orl_dse_start,orl_dse_end,orl_arg_start,orl_arg_end,orl_label=sentence.tokenize_argument_spans()
        orl_list=[(dse_s,dse_e,arg_s,arg_e,label) for dse_s,dse_e,arg_s,arg_e,label in zip(orl_dse_start,orl_dse_end,orl_arg_start,orl_arg_end,orl_label)]
        metadata_field = MetadataField(dict(
            srl_sentence=sentence,
            span2id=span2id,
            emo=emo_dict,
            orl=orl_list
        ))
        
        fields: Dict[str, Field] = dict(
            text=text_field,
            spans=span_field,
            element_arg_labels=element_role_label_field,
            element_dse_labels=element_exp_label_field,
            dse_start_labels=element_dse_start_label_field,
            dse_end_labels=element_dse_end_label_field,
            arg_start_labels=element_arg_start_label_field,
            arg_end_labels=element_arg_end_label_field,
            meta=metadata_field,
            language_id=TensorField(torch.tensor(language_id,dtype=torch.long))
        )
        return Instance(fields)

    def _read(self, file_path: str) -> Iterable[Instance]:
        if "ch" in file_path:
            language_id=0
        elif "en" in file_path:
            language_id=1
        elif "covid" in file_path:
            language_id=2
        with codecs.open(file_path, encoding="utf8") as f:
            for line in f.readlines():
                sen = json.loads(line)
                srl_sen = srl_sentence(sen)
                yield self.text_to_instance(srl_sen,language_id)

def main():
    file_path="./data/ch.weibo.conll.json"
    transformer_model="./bert-base-cased"
    indexer=PretrainedTransformerMismatchedIndexer(model_name=transformer_model,max_length=512)
    token_indexers={"transformer": indexer}
    reader = MPAQ2Reader(token_indexers=token_indexers,max_span_width=512)

    data_loader = MultiProcessDataLoader(
    reader,
    file_path,
    batch_sampler=BucketBatchSampler(batch_size=2, sorting_keys=["text"]),)
    data_loader.index_with(Vocabulary())
    language_embedding = nn.Embedding(2, 8)
    for batch in data_loader:
        text= batch["text"]
        print(text['transformer']['token_ids'].shape)
        print(text['transformer']['mask'].shape)
        print(text['transformer']['type_ids'].shape)
        print(text['transformer']['wordpiece_mask'].shape)
        print(text['transformer']['segment_concat_mask'].shape)
        print(text['transformer']['offsets'].shape)

        spans=batch['spans']
        print('spans: ',spans.shape)
        #print(spans)
       
        element_arg_labels=batch['element_arg_labels']
        print('element_arg_labels: ',element_arg_labels.shape)
        #print(element_arg_labels)

        element_dse_labels=batch['element_dse_labels']
        print('element_dse_labels: ',element_dse_labels.shape)
        print(element_dse_labels)

        dse_start_labels=batch['dse_start_labels']
        #print(dse_start_labels)

        dse_end_labels=batch['dse_end_labels']
        #print(dse_end_labels)

        arg_start_labels=batch['arg_start_labels']
        #print(arg_start_labels)

        arg_end_labels=batch['arg_end_labels']
        #print(arg_end_labels)

        meta=batch['meta']
        #print(meta)
        language_id=batch['language_id']
        #print(language_id)
        embedding=language_embedding(language_id[0])
        #print(embedding)
if __name__ == "__main__":
    main()
