import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

import torchaudio
from torchaudio import datasets
from torchaudio import transforms as T
from torchaudio import models
from torchaudio import pipelines
import torchaudio.functional as AF

from torch.nn.utils.rnn import pad_sequence

import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from torchmetrics import Accuracy
from sklearn.model_selection import train_test_split
import librosa
from tqdm import tqdm

from IPython.display import Audio


# print(torch.__version__)
# print(torchaudio.__version__)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def num_params(model):
    nums = sum(p.numel() for p in model.parameters()) / 1e6
    return nums


def print_stats(waveform, sample_rate=None, src=None):
    if src:
        print("-" * 10)
        print("Source:", src)
        print("-" * 10)
    if sample_rate:
        print("Sample Rate:", sample_rate)
    print("Shape:", tuple(waveform.shape))
    print("Dtype:", waveform.dtype)
    print(f" - Max:     {waveform.max().item():6.3f}")
    print(f" - Min:     {waveform.min().item():6.3f}")
    print(f" - Mean:    {waveform.mean().item():6.3f}")
    print(f" - Std Dev: {waveform.std().item():6.3f}")
    print()
    print(waveform)
    print()


def plot_waveform(waveform, sample_rate, title="Waveform", xlim=None, ylim=None):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].plot(time_axis, waveform[c], linewidth=1)
        axes[c].grid(True)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
        if xlim:
            axes[c].set_xlim(xlim)
        if ylim:
            axes[c].set_ylim(ylim)
    figure.suptitle(title)
    plt.show()


def plot_spectrogram(specgram, title=None, ylabel="freq_bin"):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or "Spectrogram (db)")
    axs.set_ylabel(ylabel)
    axs.set_xlabel("frame")
    im = axs.imshow(T.AmplitudeToDB()(specgram), origin="lower", aspect="auto")
    fig.colorbar(im, ax=axs)
    plt.show()


# Load the audio file using librosa
file_path = "Lab4/Data/training/Apple.5kgdt7o6.ingestion-5848b89d9c-wh4fl.wav"
if os.path.exists(file_path):
    metadata = torchaudio.info(file_path)
    print(metadata)
else:
    print(f"File not found: {file_path}")


# waveform, sample_rate = torchaudio.load(file_path)
# print_stats(waveform, sample_rate)
# plot_waveform(waveform, sample_rate)

# waveform2 = waveform.mean(dim=0, keepdim=True)
# print_stats(waveform2, sample_rate)
# plot_waveform(waveform2, sample_rate)
# Audio(waveform2.numpy(), rate=sample_rate)

# waveform, sample_rate = torchaudio.load(file_path, num_frames=20000, frame_offset=10000)
# print_stats(waveform, sample_rate)
# plot_waveform(waveform, sample_rate)
# Audio(waveform.numpy(), rate=sample_rate)

# waveform, sample_rate = torchaudio.load(file_path)
# waveform = waveform[:, 10000:30000]
# print_stats(waveform, sample_rate)
# plot_waveform(waveform, sample_rate)
# Audio(waveform.numpy(), rate=sample_rate)

waveform, sample_rate = torchaudio.load(file_path)
spectrogram = T.Spectrogram(n_fft=1024, hop_length=128)
spec = spectrogram(waveform)
plot_spectrogram(spec[0])
