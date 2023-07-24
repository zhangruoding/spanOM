from collections import defaultdict
from distutils.command.config import config
from tokenize import String
from typing import Any, Dict, List, Optional, Set, Tuple
from uuid import RESERVED_FUTURE

from overrides import overrides
from allennlp.modules.span_extractors import EndpointSpanExtractor
from spacy import Language
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import embedding, softmax, log_softmax, tensor
from allennlp.data import Vocabulary, TextFieldTensors
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder
from allennlp.nn import util
from zmq import device

from spanom.biaugmented_lstm import BiAugmentedLstmSeq2SeqEncoder,ParameterGenerationLstmSeq2SeqEncoder
from spanom.metric.average_dict import AccumulateDict
from spanom.reader import tokenize_data,get_dse_goldens
Span = Tuple[int, int]
Relation = Tuple[Tuple[int, int], Tuple[int, int]]
def initializer_1d(input_tensor, initializer):
    assert len(input_tensor.size()) == 1
    input_tensor = input_tensor.view(-1, 1)
    input_tensor = initializer(input_tensor)
    return input_tensor.view(-1)

class record():
    def __init__(self, best_record_file):
        self.record_file=best_record_file
        self.best=(0,0,0)
    def get_best(self):
        return self.best[0],self.best[1],self.best[2]
    def refresh_best(self,dse_f1,arg_f1,srl_f1):
        self.best=(dse_f1,arg_f1,srl_f1)
    def save(self,epoch,dse_f1,arg_f1,srl_f1,clear):
        best_dse,best_arg,best_srl=self.get_best()
        if epoch==0:
            with open(self.record_file,mode='a', encoding='utf8') as file:
                file.writelines('epoch          clear           dev_dse         dev_arg         dev_srl         best_dse        best_arg            best_srl\n')
        with open(self.record_file,mode='a', encoding='utf8') as file:
            file.writelines(f'''{epoch}             {clear}             {dse_f1}            {arg_f1}         {srl_f1}         {best_dse}        {best_arg}            {best_srl}\n''')
        file.close()
class MLPRepScorer(nn.Module):
    def __init__(self, input_size, inner_size, output_size, dropout=0.0):
        super(MLPRepScorer, self).__init__()
        self.rep_layer = nn.Linear(input_size, inner_size)
        self.dropout_layer = nn.Dropout(dropout)
        self.scorer = nn.Linear(inner_size, output_size)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.rep_layer.weight)
        initializer_1d(self.rep_layer.bias, nn.init.xavier_uniform_)
        nn.init.xavier_uniform_(self.scorer.weight)
        initializer_1d(self.scorer.bias, nn.init.xavier_uniform_)

    def forward(self, x):
        rep = self.dropout_layer(
            F.relu(self.rep_layer.forward(x))
        )
        scores = self.scorer.forward(rep)
        return scores
@Model.register("span_om")
class SpanBasedOpinionMinier(Model):

    default_predictor = "span_om"

    def __init__(
        self, config,
        vocab: Vocabulary, 
        embedder: TextFieldEmbedder,
        log_file: str,
        best_record_log_file:str,
        dir_path:str
    ):
        super().__init__(vocab)
        self.config=config
        self.embedder: TextFieldEmbedder = embedder
        self.lexical_dropout = float(config.lexical_dropout)
        if config.language_num>1:
            self.bilstm=ParameterGenerationLstmSeq2SeqEncoder(
            input_size=self.embedder.get_output_dim(),
            hidden_size=config.lstm_hidden_size,
            embedding_dim=8,
            num_layers=config.num_lstm_layers,
            bidirectional=True
            )
            self.language_embedding = nn.Embedding(config.language_num, 8)
        else:    
            self.bilstm = BiAugmentedLstmSeq2SeqEncoder(
                input_size=self.embedder.get_output_dim(),
                hidden_size=config.lstm_hidden_size,
                num_layers=config.num_lstm_layers,
                recurrent_dropout_probability=config.recurrent_dropout_prob,
                bidirectional=True,
                use_highway=True
            )
        # des starts and ends
        self.dse_start_scorer = MLPRepScorer(
            2 * config.lstm_hidden_size, config.arg_start_size, 2, dropout=config.dropout
        )
        self.dse_end_scorer = MLPRepScorer(
            2 * config.lstm_hidden_size, config.arg_end_size, 2, dropout=config.dropout
        )
        # dse rep
        self.dse_reps_0 = nn.Linear(2 * config.lstm_hidden_size, config.argu_size, bias=False)
        self.dse_reps_drop_0 = nn.Dropout(config.dropout)
        self.dse_span_extractor=EndpointSpanExtractor(input_dim=config.argu_size, combination=config.dse_span_extractor_combination)
        self.dse_reps = nn.Linear(self.dse_span_extractor.get_output_dim(), config.span_emb_size)
        self.dse_reps_drop = nn.Dropout(config.dropout)
        # dse score
        self.dse_unary_score_layers = nn.ModuleList(
            [nn.Linear(config.span_emb_size, config.ffnn_size) if i == 0
             else nn.Linear(config.ffnn_size, config.ffnn_size) for i
             in range(config.ffnn_depth)])
        self.dse_dropout_layers = nn.ModuleList([nn.Dropout(config.dropout) for _ in range(config.ffnn_depth)])
        self.dse_tri_score_projection = nn.Linear(config.ffnn_size, 3)

        # argument starts and ends
        self.arg_start_scorer = MLPRepScorer(
            2 * config.lstm_hidden_size, config.arg_start_size, 2, dropout=config.dropout)
        self.arg_end_scorer = MLPRepScorer(
            2 * config.lstm_hidden_size, config.arg_end_size, 2, dropout=config.dropout)
        # argument rep
        self.argu_reps_0 = nn.Linear(2 * config.lstm_hidden_size, config.argu_size, bias=False)
        self.argu_reps_drop_0 = nn.Dropout(config.dropout)
        self.argu_span_extractor=EndpointSpanExtractor(input_dim=config.argu_size, combination=config.argu_span_extractor_combination)
        self.argu_reps = nn.Linear(self.argu_span_extractor.get_output_dim(), config.span_emb_size)
        self.argu_reps_drop = nn.Dropout(config.dropout)
        # argu scores
        self.arg_unary_score_layers = nn.ModuleList(
            [nn.Linear(config.span_emb_size, config.ffnn_size) if i == 0
             else nn.Linear(config.ffnn_size, config.ffnn_size) for i
             in range(config.ffnn_depth)])
        self.arg_dropout_layers = nn.ModuleList([nn.Dropout(config.dropout) for _ in range(config.ffnn_depth)])
        self.arg_unary_score_projection = nn.Linear(config.ffnn_size, 2)
         # srl scores
        self.srl_unary_score_layers = nn.ModuleList(
            [nn.Linear(2*config.span_emb_size, config.ffnn_size)
             if i == 0 else nn.Linear(config.ffnn_size, config.ffnn_size)
             for i in range(config.ffnn_depth)])
        self.srl_dropout_layers = nn.ModuleList([nn.Dropout(config.dropout) for _ in range(config.ffnn_depth)])
        self.srl_tri_score_projection = nn.Linear(config.ffnn_size, 3)
        
        self._accumulate=AccumulateDict()
       
        self.epoch=0
        self.clear=True
        self.log_file=log_file
        self.dir_path=dir_path
        self._record=record(best_record_log_file)

    @overrides
    def forward(  # type: ignore
        self,
        text: TextFieldTensors,
        spans: torch.IntTensor,
        element_arg_labels: torch.IntTensor,
        element_dse_labels: torch.IntTensor ,
        dse_start_labels: torch.IntTensor ,
        dse_end_labels: torch.IntTensor ,
        arg_start_labels: torch.IntTensor ,
        arg_end_labels: torch.IntTensor ,
        language_id: torch.LongTensor = None,
        task:List =None,
        meta: List = None,
    ) -> Dict[str, torch.Tensor]:
        #shape (b,max_sentence_lengh,768)已经是句子中的word级别了，去掉了头尾的[CLS]等标志位
        bert_embedding=self.embedder.forward(token_ids=text['transformer']['token_ids'],
            mask=text['transformer']['mask'],
            offsets=text['transformer']['offsets'],
            wordpiece_mask=text['transformer']['wordpiece_mask'],
            type_ids =text['transformer']['type_ids'],
            segment_concat_mask = text['transformer']['segment_concat_mask'])
        text_mask = util.get_text_field_mask(text)
        basic_input = torch.dropout(bert_embedding, self.lexical_dropout, self.training)
        if self.config.language_num>1:
            Language_embedding=self.language_embedding(language_id[0])
            lstm_out= self.bilstm.forward(inputs=basic_input,mask= text_mask,embedding=Language_embedding)
        else:
            lstm_out= self.bilstm.forward(inputs=basic_input,mask= text_mask)
        #(batch,max_sentence,600)
        predict_dict = dict()

        # DSE starts and ends prediction
        gold_dse_starts_one_hot, gold_dse_ends_one_hot= dse_start_labels,dse_end_labels

        dse_starts_scores = self.dse_start_scorer.forward(lstm_out)#(batch,max_sentence,2)
        dse_ends_scores = self.dse_end_scorer.forward(lstm_out)#(batch,max_sentence,2)
        # 2. get the dse starts and ends with probability threshold
        true_pred_dse_starts, _ = self.ger_predicted_arg_boundary_by_prob_threshold(
            dse_starts_scores, self.config.arg_boundary_prob_threshold, mask=text_mask)#(batch,max_sentence)
        true_pred_dse_ends, _ = self.ger_predicted_arg_boundary_by_prob_threshold(
            dse_ends_scores, self.config.arg_boundary_prob_threshold, mask=text_mask)#(batch,max_sentence)
        if gold_dse_starts_one_hot!=None and gold_dse_ends_one_hot!=None:
            num_match_dse_starts, num_pred_dse_starts, num_gold_dse_starts = \
                self.eval_predicted_argument_boundaries(true_pred_dse_starts, gold_dse_starts_one_hot, mask=text_mask)
            num_match_dse_ends, num_pred_dse_ends, num_gold_dse_ends = \
                self.eval_predicted_argument_boundaries(true_pred_dse_ends, gold_dse_ends_one_hot, mask=text_mask)
            self._accumulate({"matched_dse_starts": num_match_dse_starts,
                                "pred_dse_starts": num_pred_dse_starts,
                                "gold_dse_starts": num_gold_dse_starts,

                                "matched_dse_ends": num_match_dse_ends,
                                "pred_dse_ends": num_pred_dse_ends,
                                "gold_dse_ends": num_gold_dse_ends}
                                )
            if self.training:
                dse_starts_loss = self.get_dse_boundary_fl_loss(dse_starts_scores, gold_dse_starts_one_hot, mask=text_mask)
                dse_ends_loss = self.get_dse_boundary_fl_loss(dse_ends_scores, gold_dse_ends_one_hot, mask=text_mask)
        # Shape: (batch_size, num_spans)
        span_mask = (spans[:, :, 0] >= 0).squeeze(-1)
        # Shape: (batch_size, num_spans, 2)
        spans = F.relu(spans.float()).long()

       
        dse_loss,candidate_dse_span_emb,predicted_dse_index=self._dse_span_forward(lstm_out,spans,span_mask,text_mask,element_dse_labels)
        pre_dse_element,pre_negative_element,pre_positive_element=self._decode_exp_element(spans, span_mask, predicted_dse_index)
        predict_dict.update({"pre_dse_element": pre_dse_element,
                            "pre_negative_element":pre_negative_element,
                            "pre_positive_element":pre_positive_element})
        # ARG starts and ends prediction
        # 1. compute the candiate argument starts and ends
        gold_arg_starts_one_hot, gold_arg_ends_one_hot = arg_start_labels,arg_end_labels

        arg_starts_scores = self.arg_start_scorer.forward(lstm_out)
        arg_ends_scores = self.arg_end_scorer.forward(lstm_out)
        # 2. get the arg starts and ends with probability threshold
        true_pred_starts, _ = self.ger_predicted_arg_boundary_by_prob_threshold(
            arg_starts_scores, self.config.arg_boundary_prob_threshold, mask=text_mask)
        true_pred_ends, _ = self.ger_predicted_arg_boundary_by_prob_threshold(
            arg_ends_scores, self.config.arg_boundary_prob_threshold, mask=text_mask)
        if gold_arg_starts_one_hot!=None and gold_arg_ends_one_hot!=None:
                       
            num_match_arg_starts, num_pred_arg_starts, num_gold_arg_starts = \
                self.eval_predicted_argument_boundaries(true_pred_starts, gold_arg_starts_one_hot, mask=text_mask)
            num_match_arg_ends, num_pred_arg_ends, num_gold_arg_ends = \
                self.eval_predicted_argument_boundaries(true_pred_ends, gold_arg_ends_one_hot, mask=text_mask)
            self._accumulate({"matched_arg_starts": num_match_arg_starts,
                                "pred_arg_starts": num_pred_arg_starts,
                                "gold_arg_starts": num_gold_arg_starts,
                                "matched_arg_ends": num_match_arg_ends,
                                "pred_arg_ends": num_pred_arg_ends,
                                "gold_arg_ends": num_gold_arg_ends}
                                )
            if self.training:
                arg_starts_loss = self.get_arg_boundary_fl_loss(arg_starts_scores, gold_arg_starts_one_hot, mask=text_mask)
                arg_ends_loss = self.get_arg_boundary_fl_loss(arg_ends_scores, gold_arg_ends_one_hot, mask=text_mask)

        argument_loss,candidate_argu_span_emb,predicted_arguments_index=self._arg_span_forward(lstm_out,spans,span_mask,text_mask,element_arg_labels)
        pre_arg_element=self._decode_arg_element(spans,span_mask,predicted_arguments_index)
        predict_dict.update({"pre_arg_element": pre_arg_element})
        
        # relation classic
        # Shape: (batch_size, List), (batch_size, num_relations, 1), (batch_size, num_relations,1)
        relations, relation_dse_indices, relation_arg_indices = self._generate_relation(
            pre_dse_element, pre_arg_element,meta,gold_arg_starts_one_hot[0,0])
        if relation_dse_indices.size(1)>0:
        # Shape: (batch_size, num_relations)
            relation_mask = (relation_dse_indices.squeeze(-1)>= 0)
            relation_dse_indices = F.relu(relation_dse_indices.float()).int()
            relation_dse_emb = util.batched_index_select(candidate_dse_span_emb, relation_dse_indices)
            relation_dse_emb=relation_dse_emb.squeeze(-2)
            relation_dse_emb=relation_dse_emb*relation_mask.unsqueeze(-1)

            relation_arg_indices=F.relu(relation_arg_indices.float()).int()
            relation_arg_emb = util.batched_index_select(candidate_argu_span_emb, relation_arg_indices)
            relation_arg_emb=relation_arg_emb.squeeze(-2)
            relation_arg_emb=relation_arg_emb*relation_mask.unsqueeze(-1)

            relation_emb=torch.cat([relation_dse_emb,relation_arg_emb],dim=-1)
            # (batch_size, num_relations, 3)
            relation_score=self.get_srl_scores(relation_emb,relation_mask)
            # (batch_size, num_relations)
            pre_relation_index=self.get_candidate_relation_index(relation_score,relation_mask)
            pre_relations,pre_agent,pre_target=self._decode_relation(relations,pre_relation_index,relation_mask)
            if meta!=None:
                # 注意，这个relation_golden_label是与pre_relation_index形状相同的
                # 所以它不一定覆盖所有的golden label
                relation_golden_label=self.get_relation_golden_label(pre_relation_index,relations,meta)
                matched_srl_num, sys_srl_num,matched_target_num,sys_target_num,matched_agent_num,sys_agent_num\
                     = self.eval_srl(pre_relation_index, relation_golden_label, relation_mask)
                gold_srl_num,gold_agent_num,gold_target_num=0,0,0
                for m in meta:
                    gold_srl_num+=len(m['orl'])
                    for orl in m['orl']:
                        if orl[4]==1:gold_target_num+=1
                        elif orl[4]==2:gold_agent_num+=1
                self._accumulate({"matched_srl_num": matched_srl_num,
                                    "sys_srl_num": sys_srl_num,
                                    "gold_srl_num": gold_srl_num,
                                    "matched_target_num":matched_target_num,
                                    "sys_target_num":sys_target_num,
                                    "gold_target_num":gold_target_num,
                                    "matched_agent_num":matched_agent_num,
                                    "sys_agent_num":sys_agent_num,
                                    "gold_agent_num":gold_agent_num})
                if self.training:
                    relation_loss=self._focal_loss(relation_score,relation_golden_label,relation_mask)
        else:
            pre_relations=[dict() for _ in range(len(relations))]
            pre_agent=[{"AGENT":list()}]
            pre_target=[{"TARGET":list()}]
            if meta!=None:
                gold_srl_num,gold_agent_num,gold_target_num=0,0,0
                for m in meta:
                    gold_srl_num+=len(m['orl'])
                    for orl in m['orl']:
                        if orl[4]==1:gold_target_num+=1
                        elif orl[4]==2:gold_agent_num+=1
                self._accumulate({"matched_srl_num": 0,
                                    "sys_srl_num": 0,
                                    "gold_srl_num": gold_srl_num,
                                    "matched_target_num":0,
                                    "sys_target_num":0,
                                    "gold_target_num":gold_target_num,
                                    "matched_agent_num":0,
                                    "sys_agent_num":0,
                                    "gold_agent_num":gold_agent_num})
                if self.training:
                    relation_loss=0
        predict_dict.update({"pre_relations": pre_relations,
                            "pre_agent":pre_agent,
                            "pre_target":pre_target})
        if self.training:
            loss_boundary=dse_starts_loss+dse_ends_loss+arg_starts_loss+arg_ends_loss
            loss=relation_loss+dse_loss+argument_loss            
            loss=loss+loss_boundary
            predict_dict.update({"loss": loss})

        return predict_dict

    @overrides
    def make_output_human_readable(
        self, output_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
       
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics=self._accumulate.get_metric(reset)
        if not self.training:
            if reset:
                self.print_log(metrics)
                self.epoch+=1
                if self.epoch % 10==0:
                    self.clear=not self.clear
                    
        return  metrics

    @staticmethod
    def _cross_entropy(scores: torch.Tensor, labels: torch.Tensor,
                       mask: torch.Tensor, weight: torch.Tensor = None):
        flat_mask = mask.view(-1).bool()
        flat_scores = scores.view(-1, scores.size(-1))
        flat_labels = labels.view(-1)
        loss = F.cross_entropy(
            flat_scores[flat_mask], flat_labels[flat_mask], weight, reduction='sum')
        return loss
    def _focal_loss(self,scores: torch.Tensor, labels: torch.Tensor,
                     mask: torch.Tensor):
        assert len(scores.size()) == 3  # [batch_size, max_len, X]
        scores = scores.view(-1, scores.size(-1))  
        # print(pred_scores)
        y = log_softmax(scores, -1)
        # print(y)
        y_hat =labels.view(-1, 1)
        # print(y_hat)
        loss_flat = torch.gather(y, dim=-1, index=y_hat).view(-1)
        # print(loss_flat)
        pt = loss_flat.exp()  # pt is the softmax score
        loss = -1.0 * self.config.srl_alpha * (1 - pt) ** self.config.srl_gamma * loss_flat
        losses = loss.view(*labels.size())
        losses = losses * mask.float()
        loss = losses.sum()
        return loss
    def ger_predicted_arg_boundary_by_prob_threshold(self, arg_boundary_scores, prob_threshold=0.5, mask=None):
        y = softmax(arg_boundary_scores, -1)  # [batch_size, max_sentence_length, 2]
        one_value = y[:, :, 1].squeeze(-1)
        max_y, max_index = y.max(dim=-1)
        prob_of_one_y = max_index * mask
        predicted_arg_boundary = (one_value > prob_threshold).type(max_index.type()) | prob_of_one_y.type(max_index.type())
        return max_index, predicted_arg_boundary.int()
    def get_dse_boundary_fl_loss(self, arg_boundary_scores, gold_arg_boundary, mask=None):
        assert len(arg_boundary_scores.size()) == 3  # [batch_size, max_sent_len, 2]
        arg_boundary_scores = arg_boundary_scores.view(-1, arg_boundary_scores.size(-1))  # [-1, 2], [total_words, 2]
        # print(pred_scores)
        y = log_softmax(arg_boundary_scores, -1)
        # print(y)
        y_hat = gold_arg_boundary.view(-1, 1)
        # print(y_hat)
        loss_flat = torch.gather(y, dim=-1, index=y_hat)
        # print(loss_flat)
        pt = loss_flat.exp()  # pt is the softmax score
        loss = -1.0 * self.config.dse_alpha * (1 - pt) ** self.config.dse_gamma * loss_flat
        losses = loss.view(*gold_arg_boundary.size())
        losses = losses * mask.float()
        loss = losses.sum()
        return loss
    def get_arg_boundary_fl_loss(self, arg_boundary_scores, gold_arg_boundary, mask=None):
        assert len(arg_boundary_scores.size()) == 3  # [batch_size, max_sent_len, 2]
        arg_boundary_scores = arg_boundary_scores.view(-1, arg_boundary_scores.size(-1))  # [-1, 2], [total_words, 2]
        # print(pred_scores)
        y = log_softmax(arg_boundary_scores, -1)
        # print(y)
        y_hat = gold_arg_boundary.view(-1, 1)
        # print(y_hat)
        loss_flat = torch.gather(y, dim=-1, index=y_hat)
        # print(loss_flat)
        pt = loss_flat.exp()  # pt is the softmax score
        loss = -1.0 * self.config.arg_alpha * (1 - pt) ** self.config.arg_gamma * loss_flat
        losses = loss.view(*gold_arg_boundary.size())
        losses = losses * mask.float()
        loss = losses.sum()
        return loss
    def eval_predicted_argument_boundaries(self, pred_arg_boundaries, gold_arg_boundarys, mask=None):
        pred_arg_boundaries = pred_arg_boundaries * mask.type(pred_arg_boundaries.type())
        matched = pred_arg_boundaries * gold_arg_boundarys
        matched_num = int(matched.sum())
        total_gold_predicates = int(gold_arg_boundarys.sum())
        total_pred_predicates = int(pred_arg_boundaries.sum())
        return matched_num, total_pred_predicates, total_gold_predicates
    def get_dse_tri_scores(self, span_emb):
        input = span_emb
        for i, ffnn in enumerate(self.dse_unary_score_layers):
            input = torch.relu(ffnn.forward(input))
            input = self.dse_dropout_layers[i].forward(input)
        output = self.dse_tri_score_projection.forward(input)
        return output
    def get_arg_unary_scores(self, span_emb):

        input = span_emb
        for i, ffnn in enumerate(self.arg_unary_score_layers):
            input = torch.relu(ffnn.forward(input))
            input = self.arg_dropout_layers[i].forward(input)
        output = self.arg_unary_score_projection.forward(input)
        return output
    def get_candidate_argument_index(self, argu_scores, mask):
        # argu_scores: [batch_size, max_argu_number, 2]
        y = softmax(argu_scores, -1)
        y_value, max_indexes = y.max(dim=-1)  #
        max_indexes = max_indexes * mask
        return max_indexes
    def eval_predicted_arguments(self, sys_arguments, gold_arguments, mask):
        sys_arguments = sys_arguments * mask.type(sys_arguments.type())
        matched = sys_arguments * gold_arguments
        matched_num = int(matched.sum())
        total_pred_arguments = int(sys_arguments.sum())
        total_gold_arguments = int(gold_arguments.sum())
        return matched_num, total_pred_arguments, total_gold_arguments   
    def eval_predicted_expression(self,predicted_dse_index: torch.IntTensor, dense_gold_dse_index: torch.IntTensor,mask: torch.BoolTensor):
        pre_num=(predicted_dse_index>0).int().sum().item()
        golden_num=(dense_gold_dse_index>0).int().sum().item()
        

        pre_negative=(predicted_dse_index==1).int()
        pre_negative=pre_negative*mask
        golden_negative=(dense_gold_dse_index==1).int()
        golden_negative=golden_negative*mask
        pre_negative_num=pre_negative.sum().item()
        golden_negative_num=golden_negative.sum().item()
        matched_negative=pre_negative*golden_negative
        matched_negative_num=matched_negative.sum().item()

        pre_positive=(predicted_dse_index==2).int()
        golden_positive=(dense_gold_dse_index==2).int()
        pre_positive_num=pre_positive.sum().item()
        golden_positive_num=golden_positive.sum().item()
        matched_positive=pre_positive*golden_positive
        matched_positive_num=matched_positive.sum().item()
        return matched_negative_num+matched_positive_num,pre_num,golden_num,\
            matched_negative_num,pre_negative_num,golden_negative_num,\
             matched_positive_num,   pre_positive_num,golden_positive_num
    def get_dse_element_focal_loss(self, argument_scores, gold_argument_index, candidate_argu_mask):
        """
        :param argument_scores: [num_sentence, max_candidate_span_num, 2]
        :param gold_argument_index: [num_sentence, max_candidate_span_num]
        :param candidate_argu_mask: [num_sentence, max_candidate_span_num]
        :return:
        """
        y = log_softmax(argument_scores, dim=-1)
        y = y.view(-1, argument_scores.size(2))
        y_hat = gold_argument_index.view(-1, 1)
        if self.training:
            ## randomly choose negative samples
            randn_neg_sample = torch.randint(0, 100, gold_argument_index.size(),dtype=torch.long,device=self.get_device())
            randn_neg_sample = randn_neg_sample > int(self.config.neg_threshold)  # randomly
            randn_neg_sample = randn_neg_sample | gold_argument_index.type(randn_neg_sample.type())
            candidate_argu_mask = candidate_argu_mask * randn_neg_sample.type(candidate_argu_mask.type())
        loss_flat = torch.gather(y, dim=-1, index=y_hat)
        pt = loss_flat.exp()  # pt is the softmax score
        loss = -1.0 * self.config.dse_alpha * (1 - pt) ** self.config.dse_gamma * loss_flat
        losses = loss.view(*gold_argument_index.size())
        losses = losses * candidate_argu_mask.float()
        loss = losses.sum()
        return loss     
    def get_arg_element_focal_loss(self, argument_scores, gold_argument_index, candidate_argu_mask):
        """
        :param argument_scores: [num_sentence, max_candidate_span_num, 2]
        :param gold_argument_index: [num_sentence, max_candidate_span_num]
        :param candidate_argu_mask: [num_sentence, max_candidate_span_num]
        :return:
        """
        y = log_softmax(argument_scores, dim=-1)
        y = y.view(-1, argument_scores.size(2))
        y_hat = gold_argument_index.view(-1, 1)
        if self.training:
            ## randomly choose negative samples
            randn_neg_sample = torch.randint(0, 100, gold_argument_index.size(),dtype=torch.long,device=self.get_device())
            randn_neg_sample = randn_neg_sample > int(self.config.neg_threshold)  # randomly
            randn_neg_sample = randn_neg_sample | gold_argument_index.type(randn_neg_sample.type())
            candidate_argu_mask = candidate_argu_mask * randn_neg_sample.type(candidate_argu_mask.type())
        loss_flat = torch.gather(y, dim=-1, index=y_hat)
        pt = loss_flat.exp()  # pt is the softmax score
        loss = -1.0 * self.config.arg_alpha * (1 - pt) ** self.config.arg_gamma * loss_flat
        losses = loss.view(*gold_argument_index.size())
        losses = losses * candidate_argu_mask.float()
        loss = losses.sum()
        return loss     
    def _decode_arg_element(self, items: torch.IntTensor, item_mask: torch.IntTensor,
                        labels: torch.IntTensor) -> List[Dict[Span, int]]:
        prediction = list()
        for item, mask, label in zip(items, item_mask, labels):
            item, label= item[mask], label[mask]
            indices = label > 0
            item= item[indices]
            element = list()
            for i in item.tolist():
                element.append(tuple(i))
            prediction.append(element)
        return prediction
    def _decode_exp_element(self, items: torch.IntTensor, item_mask: torch.IntTensor,
                        labels: torch.IntTensor) -> List[Dict[Span, int]]:
        prediction = list()
        pre_negative=list()
        pre_positive=list()
        for item, mask, label in zip(items, item_mask, labels):
            item, label= item[mask], label[mask]
            indices = label > 0
            item= item[indices]
            label=label[indices]
            assert(len(item)==len(label))
            element = list()
            negative=list()
            positive=list()
            for span,l in zip(item.tolist(),label.tolist()):
                element.append(tuple(span))
                if l==1:negative.append(tuple(span))
                elif l==2:positive.append(tuple(span))
            prediction.append(element)
            pre_negative.append(negative)
            pre_positive.append(positive)
        return prediction,pre_negative,pre_positive
    def _generate_relation(self,dse_element:List[List[Span]],arg_element:List[List[Span]],meta:List,
                           tensor: torch.Tensor):
        batch: List[Dict] = list()
        for dse,arg, m in zip(dse_element, arg_element,meta):
            span2id = m['span2id']
            dse2id, arg2id = dict(), dict()
            #if self.config.use_golden_dse_role:
            if self.training :
                orl_list=m['orl']
                dse = dse.copy()
                arg = arg.copy()
                if self.clear:
                    dse.clear()
                    arg.clear()
                for item in orl_list:
                    if (item[0],item[1]) not in dse:dse.append((item[0],item[1]))
                    if (item[2],item[3]) not in arg:arg.append((item[2],item[3]))
            for span in dse:
                dse2id[span]=span2id[span]
            for span in arg:
                arg2id[span]=span2id[span]
            relations = dict()
            for e, i in dse2id.items():
                for o, j in arg2id.items():
                    relations[(e, o)] = (i, j)
            batch.append(relations)
        
        size = [len(batch), max([len(b) for b in batch])]
        relations = [list() for _ in range(len(batch))]
        relation_dse_indices = tensor.new_full(size + [1], -1, dtype=torch.int)
        relation_arg_indices = tensor.new_full(size + [1], -1, dtype=torch.int)
        for i, _relations in enumerate(batch):
            for j, (k, v) in enumerate(_relations.items()):
                relations[i].append(k)
                relation_dse_indices[i, j, 0]=v[0]
                relation_arg_indices[i, j, 0]=v[1]
                
        return  relations, relation_dse_indices  ,relation_arg_indices
    def get_srl_scores(self, span_emb,mask):
        batch,num,_ = span_emb.size()
        flated_mask=mask.view(-1)
        input=span_emb.view(batch*num,-1)
        for i, ffnn in enumerate(self.srl_unary_score_layers):
            input = torch.relu(ffnn.forward(input))
            input = self.srl_dropout_layers[i].forward(input)
        output = self.srl_tri_score_projection.forward(input)
        output=output*flated_mask.unsqueeze(-1)
        srl_scores = output.view(batch, num, -1)
        return srl_scores*mask.unsqueeze(-1)
    def get_candidate_relation_index(self,relation_scores,mask):
        y = softmax(relation_scores, -1)
        y_value, max_indexes = y.max(dim=-1)  #
        max_indexes = max_indexes * mask
        return max_indexes
    def get_relation_golden_label(self,pre_relation_index: torch.IntTensor,relations,meta):
        def get_label(orl_list:list):
            label=dict()
            for orl in orl_list:
                label[((orl[0],orl[1]),(orl[2],orl[3]))]=orl[4]
            return label
        relation_labels = pre_relation_index.new_zeros(pre_relation_index.size())
        for i, (m, r) in enumerate(zip(meta, relations)):
            golden_label=get_label(m['orl'])
            for j, k in enumerate(r):
                if k in golden_label:
                    relation_labels[i, j] = golden_label[k]
        return relation_labels
    def eval_srl(self,pre_relation_index: torch.IntTensor, relation_golden_label: torch.IntTensor, pre_mask: torch.BoolTensor):
        pre_num,match_num=0,0
        pre_relation_index=pre_relation_index*pre_mask
        golden_mask=relation_golden_label>0
        pre_num=(pre_relation_index>0).int().sum().item()
        matched=(pre_relation_index==relation_golden_label).int()
        matched=matched*golden_mask
        match_num=matched.sum().item()

        pre_agent=(pre_relation_index==2).int()
        golden_agent=(relation_golden_label==2).int()
        matched_agent=pre_agent*golden_agent
        pre_agent_num=pre_agent.sum().item()
        matched_agent_num=matched_agent.sum().item()

        pre_target=(pre_relation_index==1).int()
        golden_target=(relation_golden_label==1).int()
        matched_target=pre_target*golden_target
        pre_target_num=pre_target.sum().item()
        matched_target_num=matched_target.sum().item()
        return match_num,pre_num,matched_target_num,pre_target_num,matched_agent_num,pre_agent_num
    def _decode_relation(self,relations:List[List[Tuple]],pre_relation_index: torch.IntTensor,mask: torch.BoolTensor):
        pre_relations=list()
        pre_agent=list()
        pre_target=list()
        for items, m, labels in zip(relations, mask, pre_relation_index):
            labels=labels[m]
            assert(len(items)==len(labels))
            relation=dict()
            agent=list()
            target=list()
            for item,lable in zip(items,labels):
                relation[item]=lable.item()
                if lable.item()==1:
                    target.append(item[1])
                elif lable.item()==2:
                    agent.append(item[1])
            pre_relations.append(relation)
            pre_agent.append({"AGENT":agent})
            pre_target.append({"TARGET":target})
        return pre_relations,pre_agent,pre_target
    def print_log(self,metric:AccumulateDict):
        def get_p_r_f1(mached,pred,golen):
            if mached==0 or pred==0:
                p,r,f1=0,0,0
            else:
                p=mached/pred
                r=mached/golen
                f1=2*p*r/(p+r)
            return round(p,2),round(r,2),round(f1,2)
        num_match_dse_starts=metric["matched_dse_starts"]
        num_pred_dse_starts=metric["pred_dse_starts"]
        num_gold_dse_starts=metric["gold_dse_starts"]
        p_dse_start,r_dse_start,f1_dse_start=get_p_r_f1(num_match_dse_starts,num_pred_dse_starts,num_gold_dse_starts)
        num_match_dse_ends=metric["matched_dse_ends"]
        num_pred_dse_ends=metric["pred_dse_ends"]
        num_gold_dse_ends=metric["gold_dse_ends"]
        p_dse_end,r_dse_end,f1_dse_end=get_p_r_f1(num_match_dse_ends,num_pred_dse_ends,num_gold_dse_ends)
        matched_dse_num=metric["matched_dse_num"]
        pred_dse_num=metric["sys_dse_num"]
        gold_dse_num=metric["gold_dse_num"]
        p_dse,r_dse,f1_dse=get_p_r_f1(matched_dse_num,pred_dse_num,gold_dse_num)
        matched_negative_num=metric["matched_negative_num"]
        pred_negative_num=metric["pre_negative_num"]
        gold_negative_num=metric["golden_negative_num"]
        p_negative,r_negative,f1_negative=get_p_r_f1(matched_negative_num,pred_negative_num,gold_negative_num)
        matched_positive_num=metric["matched_positive_num"]
        pred_positive_num=metric["pre_positive_num"]
        gold_positive_num=metric["golden_positive_num"]
        p_positive,r_positive,f1_positive=get_p_r_f1(matched_positive_num,pred_positive_num,gold_positive_num)
        num_match_arg_starts=metric["matched_arg_starts"]
        num_pred_arg_starts=metric[ "pred_arg_starts"] 
        num_gold_arg_starts=metric[ "gold_arg_starts"]
        p_arg_start,r_arg_start,f1_arg_start=get_p_r_f1(num_match_arg_starts,num_pred_arg_starts,num_gold_arg_starts)
        num_match_arg_ends=metric["matched_arg_ends"] 
        num_pred_arg_ends=metric[ "pred_arg_ends"] 
        num_gold_arg_ends=metric[ "gold_arg_ends"]
        p_arg_end,r_arg_end,f1_arg_end=get_p_r_f1(num_match_arg_ends,num_pred_arg_ends,num_gold_arg_ends)
        matched_arg_num=metric["matched_argu_num"]
        pred_arg_num=metric["sys_argu_num"]
        gold_arg_num=metric[ "gold_argu_num"]
        p_arg,r_arg,f1_arg=get_p_r_f1(matched_arg_num,pred_arg_num,gold_arg_num)
        matched_srl_num=metric["matched_srl_num"]
        pred_srl_num=metric["sys_srl_num"]
        gold_srl_num=metric["gold_srl_num"] 
        p_srl,r_srl,f1_srl=get_p_r_f1(matched_srl_num,pred_srl_num,gold_srl_num)
        matched_target_num=metric["matched_target_num"]
        pred_target_num=metric["sys_target_num"]
        gold_target_num=metric["gold_target_num"]
        p_target,r_target,f1_target=get_p_r_f1(matched_target_num,pred_target_num,gold_target_num)
        matched_agent_num=metric["matched_agent_num"]
        pred_agent_num=metric["sys_agent_num"]
        gold_agent_num=metric["gold_agent_num"]
        p_agent,r_agent,f1_agent=get_p_r_f1(matched_agent_num,pred_agent_num,gold_agent_num)
        file_name=self.log_file
        data=list()
        data.append(f'''################### epoch: {self.epoch}###################'''+'\n')
        data.append(f'''###################################### dse info ######################################'''+'\n')
        data.append(f'''           matched         pred            golen           p           r           f1'''+'\n')
        data.append(f'''start      {num_match_dse_starts}          {num_pred_dse_starts}           {num_gold_dse_starts}           {p_dse_start}           {r_dse_start}           {f1_dse_start}'''+'\n')
        data.append(f'''ends       {num_match_dse_ends}            {num_pred_dse_ends}         {num_gold_dse_ends}         {p_dse_end}            {r_dse_end}            {f1_dse_end}'''+'\n')
        data.append(f'''num        {matched_dse_num}           {pred_dse_num}           {gold_dse_num}          {p_dse}         {r_dse}         {f1_dse}'''+'\n')
        data.append(f'''negative        {matched_negative_num}           {pred_negative_num}           {gold_negative_num}          {p_negative}         {r_negative}         {f1_negative}'''+'\n')
        data.append(f'''positive        {matched_positive_num}           {pred_positive_num}           {gold_positive_num}          {p_positive}         {r_positive}         {f1_positive}'''+'\n')
        data.append(f'''###################################### arg info ######################################'''+'\n')
        data.append(f'''           matched         pred            golen           p           r           f1'''+'\n')
        data.append(f'''start      {num_match_arg_starts}          {num_pred_arg_starts}           {num_gold_arg_starts}           {p_arg_start}           {r_arg_start}           {f1_arg_start}'''+'\n')
        data.append(f'''ends       {num_match_arg_ends}            {num_pred_arg_ends}         {num_gold_arg_ends}         {p_arg_end}            {r_arg_end}            {f1_arg_end}'''+'\n')
        data.append(f'''num        {matched_arg_num}           {pred_arg_num}           {gold_arg_num}          {p_arg}         {r_arg}         {f1_arg}'''+'\n')
        data.append(f'''###################################### srl info ######################################'''+'\n')
        data.append(f'''           matched         pred            golen           p           r           f1'''+'\n')
        data.append(f'''srl        {matched_srl_num}          {pred_srl_num}           {gold_srl_num}           {p_srl}           {r_srl}           {f1_srl}'''+'\n')
        data.append(f'''agent      {matched_agent_num}          {pred_agent_num}           {gold_agent_num}           {p_agent}           {r_agent}           {f1_agent}'''+'\n')
        data.append(f'''target     {matched_target_num}          {pred_target_num}           {gold_target_num}           {p_target}           {r_target}           {f1_target}'''+'\n')
        
        with open(file_name,mode='a', encoding='utf8') as file:
            file.writelines(data)
        file.close()
        self.record_best(f1_dse,f1_arg,f1_srl)    
    def record_best(self,dse_f1,arg_f1,srl_f1):
        best_dse,best_arg,best_srl=self._record.get_best()
        if dse_f1>=best_dse  and srl_f1>= best_srl:
            self._record.refresh_best(dse_f1,arg_f1,srl_f1)
            self.save_model()
        self._record.save(self.epoch,dse_f1,arg_f1,srl_f1,self.clear)
    def save_model(self):
        modle_name=self.dir_path+'/best_model.pkl'
        torch.save(self.state_dict(),modle_name)
    def _arg_span_forward(self,input:torch.Tensor,spans:torch.Tensor,span_mask:torch.Tensor,text_mask:torch.Tensor,element_arg_labels: torch.IntTensor ):   
        argu_reps = self.argu_reps_drop_0(torch.relu(self.argu_reps_0.forward(input)))
        argu_reps=argu_reps*text_mask.unsqueeze(-1)
        candidate_argu_span_emb=self.argu_span_extractor(argu_reps,spans, text_mask, span_mask)
        candidate_argu_span_emb_shape=candidate_argu_span_emb.size()
        flatted_span_mask=span_mask.view(-1)
         # Shape: (batch_size*num_spans,600)
        candidate_argu_span_emb=candidate_argu_span_emb.view(-1,candidate_argu_span_emb_shape[2])
        candidate_argu_span_emb = self.argu_reps_drop(torch.relu(self.argu_reps.forward(candidate_argu_span_emb)))
        candidate_argu_span_emb=candidate_argu_span_emb*flatted_span_mask.unsqueeze(-1)
        flatted_candidate_argu_scores = self.get_arg_unary_scores(candidate_argu_span_emb)
        candidate_argu_scores=flatted_candidate_argu_scores.view(candidate_argu_span_emb_shape[0],candidate_argu_span_emb_shape[1],-1)
        candidate_argu_scores=candidate_argu_scores*span_mask.unsqueeze(-1)# Shape: (batch_size, num_spans, 2)
        candidate_argu_span_emb=candidate_argu_span_emb.view(candidate_argu_span_emb_shape[0],candidate_argu_span_emb_shape[1],-1)
        candidate_argu_span_emb=candidate_argu_span_emb*span_mask.unsqueeze(-1)
        # 1. get predicted arguments
        predicted_arguments_index = self.get_candidate_argument_index(candidate_argu_scores,span_mask.type(torch.cuda.LongTensor))
        # 2. eval predicted argument
        dense_gold_argus_index = element_arg_labels
        argument_loss=None
        if dense_gold_argus_index!=None:
            matched_argu_num, sys_argu_num, gold_argu_num = \
                self.eval_predicted_arguments(predicted_arguments_index, dense_gold_argus_index, mask=span_mask)
            self._accumulate({"matched_argu_num": matched_argu_num,
                                "sys_argu_num": sys_argu_num,
                                "gold_argu_num": gold_argu_num})
            if self.training:
                argument_loss = self.get_arg_element_focal_loss(candidate_argu_scores, dense_gold_argus_index,span_mask.type(torch.cuda.LongTensor))
        return argument_loss,candidate_argu_span_emb,predicted_arguments_index
    def _dse_span_forward(self,input:torch.Tensor,spans:torch.Tensor,span_mask:torch.Tensor,text_mask:torch.Tensor,element_dse_labels: torch.IntTensor ):
        dse_reps = self.dse_reps_drop_0(torch.relu(self.dse_reps_0.forward(input)))#(batch,max_sentence,300)
        dse_reps=dse_reps*text_mask.unsqueeze(-1)
        # Shape: (batch_size, num_spans, 600)
        candidate_dse_span_emb=self.dse_span_extractor(dse_reps,spans, text_mask, span_mask)
        flatted_span_mask=span_mask.view(-1)
        candidate_dse_span_emb_shape=candidate_dse_span_emb.size()
         # Shape: (batch_size*num_spans,600)
        candidate_dse_span_emb=candidate_dse_span_emb.view(-1,candidate_dse_span_emb_shape[2])
        candidate_dse_span_emb = self.dse_reps_drop(torch.relu(self.dse_reps.forward(candidate_dse_span_emb)))
        candidate_dse_span_emb=candidate_dse_span_emb*flatted_span_mask.unsqueeze(-1)
        flatted_candidate_dse_scores = self.get_dse_tri_scores(candidate_dse_span_emb)#（span_num,3）
        candidate_dse_scores=flatted_candidate_dse_scores.view(candidate_dse_span_emb_shape[0],candidate_dse_span_emb_shape[1],-1)
        candidate_dse_scores=candidate_dse_scores*span_mask.unsqueeze(-1)# Shape: (batch_size, num_spans, 3)
        candidate_dse_span_emb=candidate_dse_span_emb.view(candidate_dse_span_emb_shape[0],candidate_dse_span_emb_shape[1],-1)
        candidate_dse_span_emb=candidate_dse_span_emb*span_mask.unsqueeze(-1)

        # 1. get predicted dse
        predicted_dse_index =  self.get_candidate_argument_index(candidate_dse_scores,span_mask.type(torch.cuda.LongTensor))
        # 2. eval predicted dse
        dense_gold_dse_index = element_dse_labels
        dse_loss=None
        if dense_gold_dse_index!=None:
            matched_dse_num, sys_dse_num, gold_dse_num,matched_negative_num,pre_negative_num,golden_negative_num,matched_positive_num,pre_positive_num,golden_positive_num = \
                self.eval_predicted_expression(predicted_dse_index, dense_gold_dse_index, span_mask)
            self._accumulate({"matched_dse_num": matched_dse_num,
                                "sys_dse_num": sys_dse_num,
                                "gold_dse_num": gold_dse_num,
                                "matched_negative_num":matched_negative_num,
                                "pre_negative_num":pre_negative_num,
                                "golden_negative_num":golden_negative_num,
                                "matched_positive_num":matched_positive_num,
                                "pre_positive_num":pre_positive_num,
                                "golden_positive_num":golden_positive_num})
            if self.training:
                dse_loss = self.get_dse_element_focal_loss(candidate_dse_scores, dense_gold_dse_index,span_mask.type(torch.cuda.LongTensor))
        return dse_loss,candidate_dse_span_emb,predicted_dse_index
    def get_device(self):
        device=next(self.parameters()).device
        return device

if __name__ == "__main__":
    print(0)