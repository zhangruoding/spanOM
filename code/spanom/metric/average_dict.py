from typing import Dict, List, Tuple, Union
from collections import defaultdict

import torch
from allennlp.training.metrics import Metric




class AccumulateDict(Metric):
    """
    This [`Metric`](./metric.md) breaks with the typical `Metric` API and just stores values that were
    computed in some fashion outside of a `Metric`.  If you have some external code that computes
    the metric for you, for instance, you can use this to report the average result using our
    `Metric` API.
    """

    def __init__(self) -> None:
        self._total = defaultdict(float)
        self._count = defaultdict(int)

    def __call__(self, value_dict: Dict[str, torch.Tensor]):
        """
        # Parameters

        value_dict : `Dict[str, float]`
            The values to average.
        """
        for k, v in value_dict.items():
            self._count[k] += 1
            self._total[k] += v

    def get_metric(self, reset: bool = False):
        """
        # Returns

        The average of all values that were passed to `__call__`.
        """
        metrics = dict()
        for k, v in self._count.items():
            metrics[k] = self._total[k]
        if reset:
            self.reset()
        return metrics

    def reset(self):
        self._total = defaultdict(float)
        self._count = defaultdict(int)
def print_result(metric,file_name):
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
    
    data=list()
    
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
        
    with open(file_name,mode='w', encoding='utf8') as file:
        file.writelines(data)
    file.close()