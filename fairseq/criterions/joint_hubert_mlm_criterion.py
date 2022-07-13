# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import re
from dataclasses import dataclass, field
from typing import List, Optional
import time

import torch
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.models.hubert import ElHubertModel
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from fairseq.criterions.ctc_mlm_mtl import CtcMlmCriterion, CtcMLMCriterionConfig
from fairseq.criterions.hubert_criterion import HubertCriterion, HubertCriterionConfig


@dataclass
class JointHubertMlmCriterionConfig(CtcMLMCriterionConfig, HubertCriterionConfig):
    pass


@register_criterion("joint_hubert_mlm", dataclass=JointHubertMlmCriterionConfig)
class JointHubertMlmCriterion(FairseqCriterion):
    def __init__(self, task, pred_masked_weight, pred_nomask_weight, loss_weights=None, log_keys=None):
        super().__init__(task)
        
        self.ctc_mlm_criterions = CtcMlmCriterion(JointHubertMlmCriterionConfig, task, log_keys)
        self.hubert_criterions = HubertCriterion(task, pred_masked_weight, pred_nomask_weight, loss_weights, log_keys)
        self.end = None,
        self.start = None
        self.task = task


    def forward(self, model, sample, reduce=True, log_pred=False):
        self.end = time.time()
        if sample["net_input"]['mode'] == "speech_only" or "speech_only" in sample["net_input"]['mode'] :
            back=  self.hubert_criterions.forward(model,sample,reduce,log_pred)
            if self.start is not None:
                back[2]["time_data_speech"] = self.end-self.start
            self.start = time.time()
            
        elif sample["net_input"]["mode"] == "paired_data" or "paired_data" in sample["net_input"]['mode'] :
            back =  self.hubert_criterions.forward(model,sample,reduce,log_pred)
            
        elif sample["net_input"]["mode"] == "text_only" or "text_only" in sample["net_input"]['mode'] and model.training:
            sample["net_input"]["mode"] = "text_only"
            back = self.ctc_mlm_criterions.forward(model, sample, reduce)
            if self.start is not None:
                back[2]["time_data_text"] = self.end-self.start
            self.start = time.time()
            
        else:
            logging.error("mode: ",sample["net_input"]["mode"])
            back = None
        

        return back
        

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training (copied from normal cross entropy)."""
        is_dev = True
        for log in logging_outputs:
            if "w_errors" not in log:
                is_dev = False
        if is_dev:
            return CtcMlmCriterion.reduce_metrics(logging_outputs)


        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        speech_sample_size = sum(log.get("sample_size", 0)  for log in logging_outputs if "loss_hubert" in log)
        paired_n_token = sum(log.get("ntokens", 0 ) for log in logging_outputs if "loss_paired" in log )

        text_sample_size = sum(log.get("sample_size", 0)  for log in logging_outputs if "loss_to_mlm" in log or "loss_to_ctc" in log)
        text_n_token = sum(log.get("ntokens", 0 ) for log in logging_outputs if "loss_to_mlm" in log or "loss_to_ctc" in log)


        metrics.log_scalar("loss", loss_sum / sample_size / math.log(2), sample_size, round=3)
        if sample_size != ntokens:
            metrics.log_scalar("nll_loss", loss_sum / ntokens / math.log(2), ntokens, round=3)
            metrics.log_derived("ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg))
        else:
            metrics.log_derived("ppl", lambda meters: utils.get_perplexity(meters["loss"].avg))


        loss_key = []
        count_key = []
        correct_key=[]
        time_key = []
        length_key = []
        batch_key = []
        for log in  logging_outputs:
            for lk in log.keys():
                if lk not in loss_key and lk.startswith("loss_"):
                    loss_key.append(lk)
                if lk not in count_key and lk.startswith("count_"):
                    count_key.append(lk)
                if lk not in correct_key and lk.startswith("correct_"):
                    correct_key.append(lk)
                if lk not in time_key and lk.startswith("time_"):
                    time_key.append(lk)
                if lk not in length_key and lk.startswith("length_"):
                    length_key.append(lk)
                if lk not in batch_key and lk.startswith("batch_"):
                    length_key.append(lk)
        
                
        counts = {}
        for lk in count_key:
            if lk.startswith("count_"):
                val = sum(log[lk] for log in logging_outputs if lk in log)
                metrics.log_scalar(lk, val)
                counts[lk] = val

        

        for lk in loss_key:
            if lk.startswith("loss_"):
                if lk == "loss_hubert":
                    val = sum(log[lk] for log in logging_outputs if lk in log)
                    metrics.log_scalar(lk, val / speech_sample_size / math.log(2), round=3)
                elif lk == "loss_to_mlm":
                    val = sum(log[lk] for log in logging_outputs if lk in log)
                    metrics.log_scalar(lk, val / text_n_token / math.log(2), round=3)
                elif lk == "loss_to_ctc":
                    val = sum(log[lk] for log in logging_outputs if lk in log)
                    metrics.log_scalar(lk, val / text_n_token / math.log(2), round=3)
                elif lk=="loss_paired":
                    val = sum(log[lk] for log in logging_outputs if lk in log)
                    metrics.log_scalar(lk, val, round=3)

                else:
                    val = sum(log[lk] for log in logging_outputs if lk in log)
                    metrics.log_scalar(lk, val / sample_size / math.log(2), round=3)
            
        for lk in correct_key:   
            if lk.startswith("correct_"):
                val = sum(log[lk] for log in logging_outputs if lk in log)
                # print(lk, val, counts[re.sub("correct", "count", lk)])
                metrics.log_scalar(lk, val / counts[re.sub("correct", "count", lk)])

        for lk in time_key:                
            if lk.startswith("time_"):
                val = max(log[lk] for log in logging_outputs if lk in log)
                metrics.log_scalar(lk, val)
        for lk in length_key:
            if lk.startswith("length_"):
                val = sum(log[lk] for log in logging_outputs if lk in log)
                metrics.log_scalar(lk, val)
        for lk in batch_key:
            if lk.startswith("batch_"):
                val = sum(log[lk] for log in logging_outputs if lk in log)
                metrics.log_scalar(lk, val)

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        raise NotImplementedError()

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return False
