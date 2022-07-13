from pathlib import Path
from typing import BinaryIO, Optional, Tuple, Union, List

import numpy as np
import torch
import io
import json
import librosa
import scipy
import soundfile as sf



SF_AUDIO_FILE_EXTENSIONS = {".wav", ".flac", ".ogg"}
FEATURE_OR_SF_AUDIO_FILE_EXTENSIONS = {".npy", ".wav", ".flac", ".ogg"}

def preemphasis(x, preemph):
    return scipy.signal.lfilter([1, -preemph], [1], x)

def mulaw_encode(x, mu):
    mu = mu - 1
    fx = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    return np.floor((fx + 1) / 2 * mu + 0.5)

def mulaw_decode(y, mu):
    mu = mu - 1
    x = np.sign(y) / mu * ((1 + mu) ** np.abs(y) - 1)
    return x


def _convert_to_mono(
        waveform: torch.FloatTensor, sample_rate: int
) -> torch.FloatTensor:
    if waveform.shape[0] > 1:
        try:
            import torchaudio.sox_effects as ta_sox
        except ImportError:
            raise ImportError(
                "Please install torchaudio to convert multi-channel audios"
            )
        effects = [['channels', '1']]
        return ta_sox.apply_effects_tensor(waveform, sample_rate, effects)[0]
    return waveform


def convert_to_mono(waveform: np.ndarray, sample_rate: int) -> np.ndarray:
    if waveform.shape[0] > 1:
        _waveform = torch.from_numpy(waveform)
        return _convert_to_mono(_waveform, sample_rate).numpy()
    return waveform


def get_waveform(
        path_or_fp: Union[str, BinaryIO], normalization=True, mono=True,
        frames=-1, start=0, always_2d=True
) -> Tuple[np.ndarray, int]:
    """Get the waveform and sample rate of a 16-bit WAV/FLAC/OGG Vorbis audio.

    Args:
        path_or_fp (str or BinaryIO): the path or file-like object
        normalization (bool): Normalize values to [-1, 1] (Default: True)
        mono (bool): convert multi-channel audio to mono-channel one
        frames (int): the number of frames to read. (-1 for reading all)
        start (int): Where to start reading. A negative value counts from the end.
        always_2d (bool): always return 2D array even for mono-channel audios
    Returns:
        waveform (numpy.ndarray): 1D or 2D waveform (channels x length)
        sample_rate (float): sample rate
    """
    if isinstance(path_or_fp, str):
        ext = Path(path_or_fp).suffix
        if ext not in SF_AUDIO_FILE_EXTENSIONS:
            raise ValueError(f"Unsupported audio format: {ext}")

    try:
        import soundfile as sf
    except ImportError:
        raise ImportError(
            "Please install soundfile to load WAV/FLAC/OGG Vorbis audios"
        )

    waveform, sample_rate = sf.read(
        path_or_fp, dtype="float32", always_2d=True, frames=frames, start=start
    )
    waveform = waveform.T  # T x C -> C x T
    if mono and waveform.shape[0] > 1:
        waveform = convert_to_mono(waveform, sample_rate)
    if not normalization:
        waveform *= 2 ** 15  # denormalized to 16-bit signed integers
    if not always_2d:
        waveform = waveform.squeeze(axis=0)
    return waveform, sample_rate


def _get_kaldi_fbank(
        waveform: np.ndarray, sample_rate: int, n_bins=80
) -> Optional[np.ndarray]:
    """Get mel-filter bank features via PyKaldi."""
    try:
        from kaldi.feat.mel import MelBanksOptions
        from kaldi.feat.fbank import FbankOptions, Fbank
        from kaldi.feat.window import FrameExtractionOptions
        from kaldi.matrix import Vector

        mel_opts = MelBanksOptions()
        mel_opts.num_bins = n_bins
        frame_opts = FrameExtractionOptions()
        frame_opts.samp_freq = sample_rate
        opts = FbankOptions()
        opts.mel_opts = mel_opts
        opts.frame_opts = frame_opts
        fbank = Fbank(opts=opts)
        features = fbank.compute(Vector(waveform.squeeze()), 1.0).numpy()
        return features
    except ImportError:
        return None


def _get_torchaudio_fbank(
        waveform: np.ndarray, sample_rate, n_bins=80
) -> Optional[np.ndarray]:
    """Get mel-filter bank features via TorchAudio."""
    try:
        import torchaudio.compliance.kaldi as ta_kaldi
        waveform = torch.from_numpy(waveform)
        features = ta_kaldi.fbank(
            waveform, num_mel_bins=n_bins, sample_frequency=sample_rate
        )
        return features.numpy()
    except ImportError:
        return None


def get_fbank(path_or_fp: Union[str, BinaryIO], n_bins=80) -> np.ndarray:
    """Get mel-filter bank features via PyKaldi or TorchAudio. Prefer PyKaldi
    (faster CPP implementation) to TorchAudio (Python implementation). Note that
    Kaldi/TorchAudio requires 16-bit signed integers as inputs and hence the
    waveform should not be normalized."""
    waveform, sample_rate = get_waveform(path_or_fp, normalization=False)

    features = _get_kaldi_fbank(waveform, sample_rate, n_bins)
    if features is None:
        features = _get_torchaudio_fbank(waveform, sample_rate, n_bins)
    if features is None:
        raise ImportError(
            "Please install pyKaldi or torchaudio to enable "
            "online filterbank feature extraction"
        )

    return features


def is_npy_data(data: bytes) -> bool:
    return data[0] == 147 and data[1] == 78


def is_sf_audio_data(data: bytes) -> bool:
    is_wav = (data[0] == 82 and data[1] == 73 and data[2] == 70)
    is_flac = (data[0] == 102 and data[1] == 76 and data[2] == 97)
    is_ogg = (data[0] == 79 and data[1] == 103 and data[2] == 103)
    return is_wav or is_flac or is_ogg


def read_from_stored_zip(zip_path: str, offset: int, file_size: int) -> bytes:
    with open(zip_path, "rb") as f:
        f.seek(offset)
        data = f.read(file_size)
    return data


def parse_path(path: str) -> Tuple[str, List[int]]:
    """Parse data path which is either a path to
      1. a .npy/.wav/.flac/.ogg file
      2. a stored ZIP file with slicing info: "[zip_path]:[offset]:[length]"

        Args:
            path (str): the data path to parse

        Returns:
            file_path (str): the file path
            slice_ptr (list of int): empty in case 1;
              byte offset and length for the slice in case 2
    """

    if Path(path).suffix in FEATURE_OR_SF_AUDIO_FILE_EXTENSIONS:
        _path, slice_ptr = path, []
    else:
        _path, *slice_ptr = path.split(":")
        if not Path(_path).is_file():
            raise FileNotFoundError(f"File not found: {_path}")
    assert len(slice_ptr) in {0, 2}, f"Invalid path: {path}"
    slice_ptr = [int(i) for i in slice_ptr]
    return _path, slice_ptr

def _group_to_batches_by_utters(buffer, sorted_idx_len_pair, batch_size):
    batch_list = []

    single_batch = []
    for idx_len_pair in sorted_idx_len_pair:
        single_batch.append(buffer[idx_len_pair[0]])
        if len(single_batch) == batch_size:
            batch_list.append(single_batch)
            single_batch = []

    if len(single_batch) > 0:
        batch_list.append(single_batch)

    return batch_list

def _group_to_batches_by_frames(buffer, sorted_idx_len_pair, batch_size):
    batch_list = []
    single_batch = []
    frame_num_padded = 0
    first_utt_len = sorted_idx_len_pair[0][1]
    max_sentence = batch_size // first_utt_len // 8 * 8
    for idx_len_pair in sorted_idx_len_pair:
        if max_sentence == 0:
            max_sentence = 8
        frame_num_padded += first_utt_len
        if frame_num_padded > batch_size or len(single_batch) == max_sentence:
            if len(single_batch) > 0:
                batch_list.append(single_batch)
                single_batch = []
                first_utt_len = idx_len_pair[1]
                frame_num_padded = first_utt_len
                max_sentence = batch_size // first_utt_len // 8 * 8

        single_batch.append(buffer[idx_len_pair[0]])
    if len(single_batch) > 0:
        batch_list.append(single_batch)

    return batch_list

def _group_to_batches_by_frame_x_label(buffer, sorted_idx_len_pair, batch_size):
    batch_list = []

    single_batch = []
    frame_num_padded = 0

    max_lab_len = sorted_idx_len_pair[0][2] + 1
    max_utt_len = sorted_idx_len_pair[0][1]
    for idx_len_pair in sorted_idx_len_pair:
        if max_lab_len < idx_len_pair[2] + 1:
            max_lab_len = idx_len_pair[2] + 1
        frame_num_padded = max_utt_len * max_lab_len * (len(single_batch) )
        if frame_num_padded > batch_size:
            if len(single_batch) > 0:
                batch_list.append(single_batch)
                single_batch = []

                max_utt_len = idx_len_pair[1]
                max_lab_len = idx_len_pair[2] + 1

        single_batch.append(buffer[idx_len_pair[0]])

    if len(single_batch) > 0:
        batch_list.append(single_batch)

    return batch_list


class DataParser():
    def __init__(self):
        super().__init__()

    def _parse_data(self, data, data_type):
        if data_type.lower() == 'audio':
            parsed_data = self._parse_audio_data(data)
        elif data_type.lower() == 'info':
            parsed_data = self._parse_json_data(data)
        elif data_type.lower() == "feature":
            parsed_data = self._parse_feat_data(data)
        else:
            parsed_data = self._parse_string_data(data)
        return parsed_data

    def _parse_audio_data(self, data):
        byte_stream = io.BytesIO(data)
        with sf.SoundFile(byte_stream, 'r') as f:
            samples = f.read()
        return samples


    def _parse_json_data(self, data):
        str_data = str(data, 'utf-8')
        json_data = json.loads(str_data)

        return json_data

    def _parse_string_data(self, data):
        str_data = str(data, 'utf-8')
        return str_data

    def _parse_feat_data(self, data):
        feat = np.frombuffer(data, dtype=np.float32)
        feat = feat.reshape(-1, 80)

        return feat
