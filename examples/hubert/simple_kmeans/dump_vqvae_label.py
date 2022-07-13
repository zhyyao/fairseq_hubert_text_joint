# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math
import os
import sys
sys.path.append(os.getcwd())

import fairseq
import soundfile as sf
import torch
import torch.nn.functional as F
import tqdm
from npy_append_array import NpyAppendArray

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("dump_hubert_feature")


class HubertFeatureReader(object):
    def __init__(self, ckpt_path, layer, max_chunk=1600000):
        (
            model,
            cfg,
            task,
        ) = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])

        self.model = model[0].eval().cuda()
        if hasattr(self.model, 'w2v_encoder'):
            self.model = self.model.w2v_encoder.w2v_model
        self.task = task
        self.layer = layer
        self.max_chunk = max_chunk
        logger.info(f"TASK CONFIG:\n{self.task.cfg}")
        logger.info(f" max_chunk = {self.max_chunk}")

    def read_audio(self, path, ref_len=None):
        wav, sr = sf.read(path)
        assert sr == self.task.cfg.sample_rate, sr
        x = torch.from_numpy(wav).float()
        x = x.view(1, -1)
        import torchaudio
        import torchaudio.compliance.kaldi as ta_kaldi
        mfccs = ta_kaldi.mfcc(
            waveform=x,
            sample_frequency=sr,
            use_energy=False
        )
        mfccs = mfccs.transpose(0, 1)
        deltas = torchaudio.functional.compute_deltas(mfccs)
        ddeltas = torchaudio.functional.compute_deltas(deltas)
        concat = torch.cat([mfccs, deltas, ddeltas], dim=0)
        concat = concat.transpose(0, 1).contiguous()
        mean = concat.mean(dim=0)
        std = concat.std(dim=0)
        concat = (concat - mean) / std
        return concat

    def get_feats(self, path, ref_len=None):
        x = self.read_audio(path, ref_len).unsqueeze(0).cuda()
        with torch.no_grad():
            indices = self.model.quantize(
                source=x,
            )
        return indices


def get_path_iterator(tsv, nshard, rank):
    with open(tsv, "r") as f:
        root = f.readline().rstrip()
        lines = [line.rstrip() for line in f]
        tot = len(lines)
        shard_size = math.ceil(tot / nshard)
        start, end = rank * shard_size, min((rank + 1) * shard_size, tot)
        assert start < end, "start={start}, end={end}"
        logger.info(
            f"rank {rank} of {nshard}, process {end-start} "
            f"({start}-{end}) out of {tot}"
        )

        lines = lines[start:end]

        def iterate():
            for line in lines:
                subpath, nsample = line.split("\t")
                yield f"{root}/{subpath}", int(nsample)

        return iterate, len(lines)


def dump_feature(
    tsv_dir, split, ckpt_path, layer, nshard, rank, feat_dir, max_chunk
):
    reader = HubertFeatureReader(ckpt_path, layer, max_chunk)
    generator, num = get_path_iterator(f"{tsv_dir}/{split}.tsv", nshard, rank)
    iterator = generator()

    feat_path = f"{feat_dir}/{split}_{rank}_{nshard}.label"

    os.makedirs(feat_dir, exist_ok=True)
    if os.path.exists(feat_path):
        os.remove(feat_path)



    with open(feat_path, "w") as f:
        for path, nsample in tqdm.tqdm(iterator, total=num):
            feat = reader.get_feats(path, nsample)
            feat = list(feat.cpu().numpy())
            f.write(' '.join(map(str, feat))+"\n")
    logger.info("finished successfully")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("tsv_dir")
    parser.add_argument("split")
    parser.add_argument("ckpt_path")
    parser.add_argument("layer", type=int)
    parser.add_argument("nshard", type=int)
    parser.add_argument("rank", type=int)
    parser.add_argument("feat_dir")
    parser.add_argument("--max_chunk", type=int, default=1600000)
    args = parser.parse_args()
    logger.info(args)

    dump_feature(**vars(args))
