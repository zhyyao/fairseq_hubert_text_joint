# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pdb
import io
import logging
import math
import os
import sys
import numpy as np
import numpy as np
sys.path.append(os.getcwd())

import fairseq
import soundfile as sf
import torch
import torch.nn.functional as F
import tqdm
from fairseq.data.audio.audio_utils import (
        parse_path,
        read_from_stored_zip,
        is_sf_audio_data,
    )

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("dump_hubert_feature")


class Vqwav2vecReader(object):
    def __init__(self, ckpt_path):
        (
            model,
            cfg,
            task,
        ) = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])

        self.model = model[0].eval().cuda()
        if hasattr(self.model, 'w2v_encoder'):
            self.model = self.model.w2v_encoder.w2v_model
        self.task = task
        logger.info(f"TASK CONFIG:\n{self.task.cfg}")

    def read_audio(self, path, ref_len=None):
        wav, sr = sf.read(path)
        assert sr == self.task.cfg.sample_rate, sr
        if wav.ndim == 2:
            wav = wav.mean(-1)
        assert wav.ndim == 1, wav.ndim
        if ref_len is not None and abs(ref_len - len(wav)) > 160:
            logging.warning(f"ref {ref_len} != read {len(wav)} ({path})")
        return wav

    def get_feats(self, path, ref_len=None):
        x = self.read_audio(path, ref_len)
        with torch.no_grad():
            x = torch.from_numpy(x).float().cuda()
            if self.task.cfg.normalize:
                x = F.layer_norm(x, x.shape)
            x = x.view(1, -1)
            z = self.model.feature_extractor(x)
            _, idx = self.model.vector_quantizer.forward_idx(z)

        return idx


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
                line = line.split("\t")
                subpath = line[0]
                nsample = line[1]
                path_or_fp = os.path.join(root, subpath)
                _path, slice_ptr = parse_path(path_or_fp)
                if len(slice_ptr) == 2:
                    byte_data = read_from_stored_zip(_path, slice_ptr[0], slice_ptr[1])
                    assert is_sf_audio_data(byte_data)
                    path_or_fp = io.BytesIO(byte_data)
                yield path_or_fp, int(nsample)

        return iterate, len(lines)


def dump_feature(
    tsv_dir, split, ckpt_path, nshard, rank, lab_dir
):
    reader = Vqwav2vecReader(ckpt_path)
    generator, num = get_path_iterator(f"{tsv_dir}/{split}.tsv", nshard, rank)
    iterator = generator()

    label_path1 = f"{lab_dir}/{split}_{rank}_{nshard}.vq1"
    label_path2 = f"{lab_dir}/{split}_{rank}_{nshard}.vq2"

    os.makedirs(lab_dir, exist_ok=True)
    f1 = open(label_path1, 'w')
    f2 = open(label_path2, 'w')

    for path, nsample in tqdm.tqdm(iterator, total=num):
        idx = reader.get_feats(path, nsample)
        idx1 = idx[0, :, 0]
        idx2 = idx[0, :, 1]
        f1.write(" ".join(map(str, list(np.array(idx1.cpu())))) + "\n")
        f2.write(" ".join(map(str, list(np.array(idx1.cpu())))) + "\n")
    f1.close()
    f2.close()

    logger.info("finished successfully")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("tsv_dir")
    parser.add_argument("split")
    parser.add_argument("ckpt_path")
    parser.add_argument("nshard", type=int)
    parser.add_argument("rank", type=int)
    parser.add_argument("lab_dir")
    args = parser.parse_args()
    logger.info(args)

    dump_feature(**vars(args))
