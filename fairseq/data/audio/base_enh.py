import numpy as np
import numpy
import math
import random
import os
import random
import torchaudio
import scipy.signal as sps
eps = np.finfo(np.float32).eps

def parse_scp(scp, path_list, time=False):
	with open(scp) as fid:
		for line in fid:
			tmp = line.strip().split(".wav")
			#print(tmp)
			data, sr = torchaudio.load_wav(tmp[0] + '.wav')
			data = data.detach().numpy()[0] /32767.0
			t = data.shape[0] / sr
			#print(data.shape)
			path_list.append({'inputs': data, 'duration': float(t)})

def rms(data):
	#print('data', data.shape, type(data))
	energy = data ** 2
	#print('energy', energy.shape, type(energy))
	max_e = np.max(energy)
	low_thres = max_e * (10**(-50/10))
	rms = np.mean(energy[energy >= low_thres])
	return rms

def snr_mix(clean, noise, snr):
	clean_rms = rms(clean)
	clean_rms = np.maximum(clean_rms, eps)
	noise_rms = rms(noise)
	noise_rms = np.maximum(noise_rms, eps)
	k = math.sqrt(clean_rms / (10**(snr/10) * noise_rms))
	new_noise = noise * k
	return new_noise

def mix_noise(clean, noise, snr, scale, channels=1):
	clean_length = clean.shape[0]
	noise_length = noise.shape[0]
	if clean_length > noise_length:
		st = random.randint(0, clean_length - noise_length)
		noise_t = np.zeros([clean_length])
		noise_t[st:st+noise_length] = noise
		noise = noise_t
	elif clean_length < noise_length:
		st = random.randint(0, noise_length - clean_length)
		noise = noise[st:st+clean_length]
	
	snr_noise = snr_mix(clean, noise, snr)
	return snr_noise
def addnoise(clean, noise, snr, scale):

	clean = np.squeeze(clean)
	noise = np.squeeze(noise)
	gen_noise = mix_noise(clean, noise, snr, scale)
	noisy = clean + gen_noise
	max_amp = np.max(np.abs(noisy))
	max_amp = np.maximum(max_amp, eps)
	noisy_scale = 1. / max_amp * scale
	clean = clean * noisy_scale
	noisy = noisy * noisy_scale
	gen_noise = gen_noise * noisy_scale
	return noisy, clean

def addscale(clean, scale):
	clean = np.squeeze(clean)
	max_amp = np.max(np.abs(clean))	
	max_amp = np.maximum(max_amp, eps)
	noisy_scale = 1. / max_amp * scale
	clean = clean * noisy_scale
	return clean, clean

def addreverb(cln_wav, rir_wav, channels=1, predelay=50,sample_rate=16000):
    """
    add reverberation
    args:
        cln_wav: L
        rir_wav: L x C
        rir_wav is always [Lr, C]
        predelay is ms
    return:
        wav_tgt: L x C
    """
    if len(rir_wav.shape) == 1:
        rir_wav = np.expand_dims(rir_wav, 0)
    cln_wav = np.squeeze(cln_wav)
    rir_len = rir_wav.shape[0]
    wav_tgt = np.zeros([channels, cln_wav.shape[0] + rir_len-1])
    dt = np.argmax(rir_wav, 0).min()
    et = dt+(predelay*sample_rate)//1000
    et_rir = rir_wav[:et]
    wav_early_tgt = np.zeros([channels, cln_wav.shape[0] + et_rir.shape[0]-1])
	
    
    for i in range(channels):
        wav_tgt[i] = sps.oaconvolve(cln_wav, rir_wav[:, i])
        wav_early_tgt[i] = sps.oaconvolve(cln_wav, et_rir[:, i])
    # L x C
    wav_tgt = np.transpose(wav_tgt)
    wav_tgt = wav_tgt[:cln_wav.shape[0]]
    wav_early_tgt = np.transpose(wav_early_tgt)
    wav_early_tgt = wav_early_tgt[:cln_wav.shape[0]]
    return wav_tgt, wav_early_tgt
