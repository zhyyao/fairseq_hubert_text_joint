import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from tqdm import tqdm
from dataclasses import dataclass, field
import numpy as np
from .modules import *
from fairseq.dataclass import FairseqDataclass
from fairseq.data.audio.audio_utils import mulaw_decode
from fairseq.models import BaseFairseqModel, register_model
from fairseq.models.vqvae.vector_quantizer import VQEmbeddingEMA


def get_gru_cell(gru):
    gru_cell = nn.GRUCell(gru.input_size, gru.hidden_size)
    gru_cell.weight_hh.data = gru.weight_hh_l0.data
    gru_cell.weight_ih.data = gru.weight_ih_l0.data
    gru_cell.bias_hh.data = gru.bias_hh_l0.data
    gru_cell.bias_ih.data = gru.bias_ih_l0.data
    return gru_cell


class Encoder(nn.Module):
    def __init__(self, in_channels, channels, embedding_dim):
        super(Encoder, self).__init__()
        self.blocks = nn.Sequential(
            make_conv(in_channels, channels, 3, 1, 0, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(True),
            make_conv(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(True),
            make_conv(channels, channels, 4, 2, 1, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(True),
            make_conv(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(True),
            make_conv(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(True),
            make_conv(channels, embedding_dim, 1)
        )

    def forward(self,x):
        x = self.blocks(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_channels, use_jitter, jitter_prob, speaker_cond, n_speakers, speaker_embedding_dim,
                 conditioning_channels, mu_embedding_dim, rnn_channels,fc_channels):
        super().__init__()
        self.rnn_channels = rnn_channels
        self.quantization_channels = 2**8
        self.hop_length = 160
        self.speaker_cond = speaker_cond
        self.use_jitter = use_jitter
        self.input_dim = in_channels

        if self.speaker_cond:
            self.speaker_embedding = nn.Embedding(n_speakers, speaker_embedding_dim)
            self.input_dim += speaker_embedding_dim
        if use_jitter:
            self.jitter = Jitter(jitter_prob)

        self.rnn1 = nn.GRU(self.input_dim, conditioning_channels,
                           num_layers=2, batch_first=True, bidirectional=True)
        self.mu_embedding = nn.Embedding(self.quantization_channels, mu_embedding_dim)
        self.rnn2 = nn.GRU(mu_embedding_dim + 2*conditioning_channels, rnn_channels, batch_first=True)
        self.fc1 = nn.Linear(rnn_channels, fc_channels)
        self.fc2 = nn.Linear(fc_channels, self.quantization_channels)

    def forward(self, x, z, speakers=None):
        if self.use_jitter:
            z = self.jitter(z) #!!!Transpose
        z = F.interpolate(z.transpose(1, 2), scale_factor=2)
        z = z.transpose(1, 2)

        if self.speaker_cond:
            assert speakers is not None
            speakers = self.speaker_embedding(speakers)
            speakers = speakers.unsqueeze(1).expand(-1, z.size(1), -1)

            z = torch.cat((z, speakers), dim=-1)
        z, _ = self.rnn1(z)

        z = F.interpolate(z.transpose(1, 2), scale_factor=160)
        z = z.transpose(1, 2)

        x = self.mu_embedding(x)
        x, _ = self.rnn2(torch.cat((x, z), dim=2))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def generate(self, z, speaker):
        output = []
        cell = get_gru_cell(self.rnn2)

        z = F.interpolate(z.transpose(1, 2), scale_factor=2)
        z = z.transpose(1, 2)

        speaker = self.speaker_embedding(speaker)
        speaker = speaker.unsqueeze(1).expand(-1, z.size(1), -1)

        z = torch.cat((z, speaker), dim=-1)
        z, _ = self.rnn1(z)

        z = F.interpolate(z.transpose(1, 2), scale_factor=160)
        z = z.transpose(1, 2)

        batch_size, sample_size, _ = z.size()

        h = torch.zeros(batch_size, self.rnn_channels, device=z.device)
        x = torch.zeros(batch_size, device=z.device).fill_(self.quantization_channels // 2).long()

        for m in tqdm(torch.unbind(z, dim=1), leave=False):
            x = self.mu_embedding(x)
            h = cell(torch.cat((x, m), dim=1), h)
            x = F.relu(self.fc1(h))
            logits = self.fc2(x)
            dist = Categorical(logits=logits)
            x = dist.sample()
            output.append(2 * x.float().item() / (self.quantization_channels - 1.) - 1.)

        output = np.asarray(output, dtype=np.float64)
        output = mulaw_decode(output, self.quantization_channels)
        return output

@dataclass
class VqvaeConfig(FairseqDataclass):
    #encoder config
    feature_dim: int = field(
        default=39, metadata={"help": "input feautre dimension, 39 for MFCC, 80 for Fbank"}
    )
    encoder_channel: int = field(
        default=768
    )


    # vector quantizer
    latent_vars: int = field(
        default=512,
        metadata={"help": "number of latent variables V in codebook"}
    )
    latent_dim: int = field(
        default=64,
        metadata={"help": "dimension for codebook"}
    )

    # decoder config
    use_jitter: bool = field(
        default=False, metadata={"help": "whether use jitter layer"}
    )
    jitter_prob: float = field(
        default=0.12, metadata={"help": "jitter probability"}
    )
    speaker_cond: bool = field(
        default=False, metadata={"help": "whether use speaker id"}
    )
    n_speakers: int = field(
        default=200
    )
    speaker_embedding_dim: int = field(
        default=64
    )
    mu_embedding_dim: int = field(
        default=256
    )
    conditioning_channels: int = field(
        default=128
    )
    decoder_dim: int = field(
        default=896
    )
    decoder_layers: int = field(
        default=2
    )
    fc_dim: int = field(
        default=256
    )


@register_model("vqvae", dataclass=VqvaeConfig)
class VqvaeModel(BaseFairseqModel):
    def __init__(self, cfg: VqvaeConfig):
        super(VqvaeModel, self).__init__()
        self.cfg = cfg

        self.encoder = Encoder(
            cfg.feature_dim,
            cfg.encoder_channel,
            cfg.latent_dim,
        )

        self.vector_quantizer = VQEmbeddingEMA(cfg.latent_vars, cfg.latent_dim)

        self.decoder = Decoder(
            cfg.latent_dim,
            cfg.use_jitter,
            cfg.jitter_prob,
            cfg.speaker_cond,
            cfg.n_speakers,
            cfg.speaker_embedding_dim,
            cfg.conditioning_channels,
            cfg.mu_embedding_dim,
            cfg.decoder_dim,
            cfg.fc_dim)

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        return state_dict

    @classmethod
    def build_model(cls, cfg:VqvaeConfig, task=None):
        """Build a new model instance."""

        return cls(cfg)

    def forward(self, mel, audio, speaker_id=None):
        mel = mel.permute(0, 2, 1).contiguous()

        x = self.encoder(mel)
        z, vq_loss, perplexity = self.vector_quantizer(x.transpose(1, 2))
        y = self.decoder(audio[:, :-1], z, speaker_id)
        results = {"reconstructed_x": y, "vq_loss": vq_loss, "perplexity": perplexity}
        return results

    def encode(self, x):
        z = self.encoder(x)
        z, indices = self.codebook.encode(z.transpose(1, 2))
        return z, indices

    def get_extra_losses(self, net_output):
        pen = []

        if "vq_loss" in net_output:
            pen.append(net_output["vq_loss"])


        return pen
