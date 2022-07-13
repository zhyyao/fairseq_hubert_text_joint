# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

from functools import reduce
import math
from argparse import Namespace
from dataclasses import dataclass, field
from random import sample
from omegaconf import II
from typing import Optional

import torch
import torch.nn.functional as F
from fairseq import metrics, models, utils,modules
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq import modules
from fairseq.dataclass import FairseqDataclass
from fairseq.data.data_utils import post_process
from fairseq.tasks import FairseqTask
from fairseq.logging.meters import safe_round
import time

from fairseq.data.data_utils import compute_mask_indices

@dataclass
class CtcMLMCriterionConfig(FairseqDataclass):
    zero_infinity: bool = field(
        default=False,
        metadata={"help": "zero inf loss when source length <= target length"},
    )
    sentence_avg: bool = II("optimization.sentence_avg")
    post_process: str = field(
        default="letter",
        metadata={
            "help": "how to post process predictions into words. can be letter, "
            "wordpiece, BPE symbols, etc. "
            "See fairseq.data.data_utils.post_process() for full list of options"
        },
    )
    wer_kenlm_model: Optional[str] = field(
        default=None,
        metadata={
            "help": "if this is provided, use kenlm to compute wer (along with other wer_* args)"
        },
    )
    wer_lexicon: Optional[str] = field(
        default=None,
        metadata={"help": "lexicon to use with wer_kenlm_model"},
    )
    wer_lm_weight: float = field(
        default=2.0,
        metadata={"help": "lm weight to use with wer_kenlm_model"},
    )
    wer_word_score: float = field(
        default=-1.0,
        metadata={"help": "lm word score to use with wer_kenlm_model"},
    )

    wer_args: Optional[str] = field(
        default=None,
        metadata={
            "help": "DEPRECATED: tuple of (wer_kenlm_model, wer_lexicon, wer_lm_weight, wer_word_score)"
        },
    )


@register_criterion("ctc_mlm", dataclass=CtcMLMCriterionConfig)
class CtcMlmCriterion(FairseqCriterion):
    def __init__(self, cfg: CtcMLMCriterionConfig, task: FairseqTask, log_keys=None):
        super().__init__(task)
        self.blank_idx = (
            task.target_dictionary.index(task.blank_symbol)
            if hasattr(task, "blank_symbol")
            else 0
        )
        self.target_dictionary = task.target_dictionary
        self.pad_idx = task.target_dictionary.pad()
        self.eos_idx = task.target_dictionary.eos()
        self.padding_idx = task.phoneme_dictionary.pad()
        self.mask = task.phoneme_dictionary.index("<mask>")
        self.post_process = cfg.post_process

        if cfg.wer_args is not None:
            (
                cfg.wer_kenlm_model,
                cfg.wer_lexicon,
                cfg.wer_lm_weight,
                cfg.wer_word_score,
            ) = eval(cfg.wer_args)

        self.zero_infinity = cfg.zero_infinity
        self.sentence_avg = cfg.sentence_avg
        self.log_keys = [] if log_keys is None else log_keys

        
    def get_loss(self, model, sample, net_output,name, target_ids,reduce=True):
        lprobs = model.get_normalized_probs(
            net_output[name].transpose(0,1), log_probs=True
        ).contiguous()  # (T, B, C) from the encoder
        
        if target_ids == -1 :
            padding_index = 1
        else:
            padding_index = target_ids
        if "src_lengths" in sample["net_input"]:
            input_lengths = sample["net_input"]["src_lengths"]
        
        else:
            if net_output["padding_mask"] is not None:
                non_padding_mask = ~net_output["padding_mask"][padding_index]
                input_lengths = non_padding_mask.long().sum(-1)
            else:
                input_lengths = lprobs.new_full(
                    (lprobs.size(1),), lprobs.size(0), dtype=torch.long
                )

        if target_ids == -1 :
            pad_mask = (sample["no_rep_target"] != self.pad_idx) & (
                sample["no_rep_target"] != self.eos_idx
            )
            targets_flat = sample["no_rep_target"].masked_select(pad_mask)
            if "no_rep_length" in sample:
                target_lengths = sample["no_rep_length"]
            else:
                target_lengths = pad_mask.sum(-1)
            
        else:
            pad_mask = (sample["target_list"][target_ids] != self.pad_idx) & (
                sample["target_list"][target_ids] != self.eos_idx
            )
            targets_flat = sample["target_list"][target_ids].masked_select(pad_mask)
            if "target_lengths_list" in sample:
                target_lengths = sample["target_lengths_list"][target_ids]
            else:
                target_lengths = pad_mask.sum(-1)
            

        with torch.backends.cudnn.flags(enabled=False):
            loss = F.ctc_loss(
                lprobs,
                targets_flat,
                input_lengths,
                target_lengths,
                blank=self.blank_idx,
                reduction="sum",
                zero_infinity=self.zero_infinity,
            )

        return loss,target_lengths, input_lengths, lprobs
    def forward(self, model, sample, reduce=True):
        
        if sample["net_input"]["mode"] == "paired_data":
            return self.forward_speech(model,sample,reduce=reduce)
        elif sample["net_input"]["mode"] == "text_only":
            return self.forward_text(model,sample,reduce=reduce)
        
    def forward_text(self, model, sample, reduce=reduce):

        masked_tokens = sample["phoneme_target"].ne(self.padding_idx)
        sample_size = masked_tokens.int().sum()
        masked_tokens = torch.where(
            masked_tokens.any(),
            masked_tokens,
            masked_tokens.new([True]),
        )
        # t_start = time.time()
        # start = time.time()
        net_output = model(**sample["net_input"],masked_tokens = masked_tokens)
        # end = time.time()
        # forw_time = end-start

        # target lprob
        if net_output["trg_out"] is not None:
            lprobs_final = model.get_normalized_probs(
                net_output["trg_out"].transpose(0,1), log_probs=True
            ).contiguous()  # (T, B, C) from the encoder

        # phf prob
        # import pdb
        # pdb.set_trace()
        t_start = time.time()
        if net_output["phf_out"] is not None:
            probs_text =  model.get_normalized_probs(
                net_output["phf_out"] ,log_probs=True
            ).contiguous()
              # (B, T, C) from the encoder
            ph_target = sample["phoneme_target"].long()
            input_lenghts = sample["phoneme_length"]

        
            if masked_tokens is not None:
                ph_target = ph_target[masked_tokens]
        if net_output["trg_out"] is not None:
            input_lengths_bpe, targets_flat_bpe, target_lengths_bpe = self.get_flat_input_special(sample, "ltr")


        # start = time.time()
        with torch.backends.cudnn.flags(enabled=False):
            # start = time.time()
            losses = []
            loss_mlm = None
            if net_output["phf_out"] is not None:
                loss_mlm =  modules.cross_entropy(
                    probs_text.view(-1, probs_text.size(-1)),
                    ph_target.view(-1),
                    ignore_index=self.padding_idx,
                    reduction="sum",
                )
                losses.append(loss_mlm)
            # mlm_time = end-start
            # import pdb
            # pdb.set_trace()
            loss_final = None
            if net_output["trg_out"] is not None:
                loss_final = F.ctc_loss(
                    lprobs_final,
                    targets_flat_bpe,
                    input_lengths_bpe,
                    target_lengths_bpe,
                    blank=self.blank_idx,
                    reduction="sum",
                    zero_infinity=self.zero_infinity,
                )
            # print(loss_final)
                losses.append(loss_final)
            loss = losses[0]
            for l in range(1,len(losses)):
                loss = loss+losses[l]
            # loss = loss_mlm
        # end = time.time()
        # ctc_time = end-start

        ntokens = sample_size
        # t_end = time.time()
        # t_time = t_end-t_start
        end = time.time()
        
        logging_output = {
            "loss": utils.item(loss.data),  # * sample['ntokens'],
            "ntokens": ntokens,
            "nsentences": sample["id"].numel(),
            "sample_size": sample_size,
            "time_text_loss": end-t_start,
            # "time_text_only_forward": forw_time,
            # "time_text_only_mlm" : mlm_time,
            # "time_text_only_ctc" : ctc_time,
            # "time_text_only_other" : (t_time - forw_time - mlm_time - ctc_time),
            # "time_text_only_total" : t_time
        }
        if loss_mlm is not None:
            logging_output["loss_to_mlm"] = utils.item(loss_mlm.data)
        if loss_final is  not None:
            logging_output["loss_to_ctc"] = utils.item(loss_final.data)
        for lk in self.log_keys:
            if lk in net_output:
                logging_output[lk] = float((net_output[lk]))

        return loss, sample_size, logging_output

    def get_flat_input(self,sample):
        target_num = len(sample["target_list"])
        targets_flat=[]
        target_lengths=[]
        for index, target in enumerate(sample["target_list"]):
            pad_mask = (target != self.pad_idx) & (
                target != self.eos_idx
            )
            targets_flat.append(target.masked_select(pad_mask))
            target_lengths.append(sample["target_lengths_list"][index])
            
        return targets_flat,target_lengths

    def get_flat_input_special(self,sample,prefix):
        input_lengths = sample["net_input"]["lengths"]
        pad_mask = (sample[prefix+"_target"] != self.pad_idx) & (
            sample[prefix+"_target"] != self.eos_idx
        )
        targets_flat = sample[prefix+"_target"].masked_select(pad_mask)
        if prefix+"_length" in sample:
            target_lengths = sample[prefix+"_length"]
        else:
            target_lengths = pad_mask.sum(-1)
        return input_lengths, targets_flat,target_lengths

    
    def forward_speech(self, model, sample, reduce=True):
        # t_start = time.time()
        # start = time.time()
        net_output = model(**sample["net_input"])
        # end = time.time()
        # forw_time = end-start

        # start = time.time()
        if net_output["phf_out"] is not None and len(sample["target_list"]) == 2:
            loss_phf, _ ,_, _= self.get_loss(model,sample,net_output,"phf_out",-1,reduce)
        # end = time.time()
        # phf_time = end - start

        # start = time.time()
        if net_output["trg_out_t"] is not None and len(sample["target_list"]) == 2:
            loss_text, _ , _, _= self.get_loss(model,sample, net_output, "trg_out_t",0,reduce)
        # end = time.time()
        # text_time = end -start
    
        # start = time.time()
        loss_final, target_lengths, input_lengths, lprobs = self.get_loss(model,sample,net_output,"trg_out",0,reduce)
        # end = time.time()
        # trg_time = end-start

        if net_output["phf_out"] is not None and len(sample["target_list"]) == 2:
            loss = loss_phf + loss_text + loss_final
        else:
            if net_output["trg_out_t"] is not None and len(sample["target_list"]) == 2:
                loss = loss_text + loss_final
            else:
                loss = loss_final

        
        ntokens = (
            sample["ntokens"][0] if "ntokens" in sample else target_lengths.sum().item()
        )
        sample_size = sample["target_list"][0].size(0) if self.sentence_avg else ntokens

        # t_end = time.time()
        # t_time = t_end-t_start
        logging_output = {
            "loss": utils.item(loss.data),  # * sample['ntokens'],
            "loss_pd_x": utils.item(loss_final.data),
            "ntokens": ntokens,
            "nsentences": sample["id"].numel(),
            "sample_size": sample_size,
            # "time_paired_forward": forw_time,
            # "time_paired_phf": phf_time,
            # "time_paired_text_trg": text_time,
            # "time_paired_trg": trg_time,
            # "time_paired_other": (t_time - phf_time - forw_time - text_time - trg_time),
            # "time_paired_total": t_time
        }
        if net_output["phf_out"] is not None and len(sample["target_list"]) == 2:
            logging_output["loss_pd_phf"] = utils.item(loss_phf.data)
        if net_output["trg_out_t"] is not None and len(sample["target_list"]) == 2:
            logging_output["loss_pd_xt"] = utils.item(loss_text.data)

        if not model.training:
            import editdistance

            with torch.no_grad():
                lprobs_t = lprobs.transpose(0, 1).float().contiguous().cpu()

                c_err = 0
                c_len = 0
                w_errs = 0
                w_len = 0
                wv_errs = 0
                t_list = sample["target_label"][0] if "target_label" in sample else sample["target_list"][0]
                for lp, t, inp_l in zip(
                    lprobs_t,
                    t_list,
                    input_lengths,
                ):
                    lp = lp[:inp_l].unsqueeze(0)

                    decoded = None
                    if self.w2l_decoder is not None:
                        decoded = self.w2l_decoder.decode(lp)
                        if len(decoded) < 1:
                            decoded = None
                        else:
                            decoded = decoded[0]
                            if len(decoded) < 1:
                                decoded = None
                            else:
                                decoded = decoded[0]

                    p = (t != self.task.target_dictionary.pad()) & (
                        t != self.task.target_dictionary.eos()
                    )
                    targ = t[p]
                    targ_units = self.task.target_dictionary.string(targ)
                    targ_units_arr = targ.tolist()

                    toks = lp.argmax(dim=-1).unique_consecutive()
                    pred_units_arr = toks[toks != self.blank_idx].tolist()

                    c_err += editdistance.eval(pred_units_arr, targ_units_arr)
                    c_len += len(targ_units_arr)

                    targ_words = post_process(targ_units, self.post_process).split()

                    pred_units = self.task.target_dictionary.string(pred_units_arr)
                    pred_words_raw = post_process(pred_units, self.post_process).split()

                    if decoded is not None and "words" in decoded:
                        pred_words = decoded["words"]
                        w_errs += editdistance.eval(pred_words, targ_words)
                        wv_errs += editdistance.eval(pred_words_raw, targ_words)
                    else:
                        dist = editdistance.eval(pred_words_raw, targ_words)
                        w_errs += dist
                        wv_errs += dist

                    w_len += len(targ_words)

                logging_output["wv_errors"] = wv_errs
                logging_output["w_errors"] = w_errs
                logging_output["w_total"] = w_len
                logging_output["c_errors"] = c_err
                logging_output["c_total"] = c_len
                
        return loss, sample_size, logging_output

    
    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""

        loss_sum = utils.item(sum(log.get("loss", 0) for log in logging_outputs))
        ntokens = utils.item(sum(log.get("ntokens", 0) for log in logging_outputs))
        nsentences = utils.item(
            sum(log.get("nsentences", 0) for log in logging_outputs)
        )
        sample_size = utils.item(
            sum(log.get("sample_size", 0) for log in logging_outputs)
        )

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar("ntokens", ntokens)
        metrics.log_scalar("nsentences", nsentences)
        if sample_size != ntokens:
            metrics.log_scalar(
                "nll_loss", loss_sum / ntokens / math.log(2), ntokens, round=3
            )

        c_errors = sum(log.get("c_errors", 0) for log in logging_outputs)
        metrics.log_scalar("_c_errors", c_errors)
        c_total = sum(log.get("c_total", 0) for log in logging_outputs)
        metrics.log_scalar("_c_total", c_total)
        w_errors = sum(log.get("w_errors", 0) for log in logging_outputs)
        metrics.log_scalar("_w_errors", w_errors)
        wv_errors = sum(log.get("wv_errors", 0) for log in logging_outputs)
        metrics.log_scalar("_wv_errors", wv_errors)
        w_total = sum(log.get("w_total", 0) for log in logging_outputs)
        metrics.log_scalar("_w_total", w_total)

        if c_total > 0:
            metrics.log_derived(
                "uer",
                lambda meters: safe_round(
                    meters["_c_errors"].sum * 100.0 / meters["_c_total"].sum, 3
                )
                if meters["_c_total"].sum > 0
                else float("nan"),
            )
        if w_total > 0:
            metrics.log_derived(
                "wer",
                lambda meters: safe_round(
                    meters["_w_errors"].sum * 100.0 / meters["_w_total"].sum, 3
                )
                if meters["_w_total"].sum > 0
                else float("nan"),
            )
            metrics.log_derived(
                "raw_wer",
                lambda meters: safe_round(
                    meters["_wv_errors"].sum * 100.0 / meters["_w_total"].sum, 3
                )
                if meters["_w_total"].sum > 0
                else float("nan"),
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
