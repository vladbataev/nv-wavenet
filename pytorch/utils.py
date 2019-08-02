# *****************************************************************************
#  Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
# 
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met: 
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
# 
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# 
# *****************************************************************************
import os
import torch
import torch.nn.functional as F
import numpy as np

from scipy.io.wavfile import read
from wavenet import WaveNet

MAX_WAV_VALUE = 32768.0

def load_wav(full_path):
    """
    Loads wavdata into torch array
    """
    sampling_rate, data = read(full_path)
    return data.astype(np.float32), sampling_rate

def files_to_list(filename, delimiter=","):
    """
    Takes a text file of filenames and makes a list of filenames
    """
    files = []
    with open(filename, encoding='utf-8') as f:
        for line in f:
            tmp = line.strip().split(delimiter)
            if len(tmp) == 2:
                files.append(tmp)
            else:
                files.append([tmp[0], ""])
    return files

def load_filepaths_and_text(filename, split="|"):
    with open(filename, encoding='utf-8') as f:
        filepaths_and_text = [line.strip().split(split) for line in f]
    return filepaths_and_text

def to_gpu(x):
    x = x.contiguous()
    
    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)
    return torch.autograd.Variable(x)

def mu_law_decode_numpy(x, mu_quantization=256):
    assert(np.max(x) <= mu_quantization)
    assert(np.min(x) >= 0)
    mu = mu_quantization - 1.
    # Map values back to [-1, 1].
    signal = 2 * (x / mu) - 1
    # Perform inverse of mu-law transformation.
    magnitude = (1 / mu) * ((1 + mu)**np.abs(signal) - 1)
    return np.sign(signal) * magnitude

def mu_law_decode(x, mu_quantization=256):
    assert(torch.max(x) <= mu_quantization)
    assert(torch.min(x) >= 0)
    x = x.float()
    mu = mu_quantization - 1.
    # Map values back to [-1, 1].
    signal = 2 * (x / mu) - 1
    # Perform inverse of mu-law transformation.
    magnitude = (1 / mu) * ((1 + mu)**torch.abs(signal) - 1)
    return torch.sign(signal) * magnitude

def mu_law_encode(x, mu_quantization=256):
    assert(torch.max(x) <= 1.0)
    assert(torch.min(x) >= -1.0)
    mu = mu_quantization - 1.
    scaling = np.log1p(mu)
    x_mu = torch.sign(x) * torch.log1p(mu * torch.abs(x)) / scaling
    encoding = ((x_mu + 1) / 2 * mu + 0.5).long()
    return encoding

def collate_fn(batch, mel_min_value=-4.0):
    max_mel_length = batch[0][0].size()[1]
    max_audio_length = batch[0][1].size()[0]
    for i in range(1, len(batch)):
        max_mel_length = max(max_mel_length, batch[i][0].size()[1])
        max_audio_length = max(max_audio_length, batch[i][1].size()[0])
    padded_mels = []
    padded_audios = []
    seq_lens = []
    for pair in batch:
        mel, audio = pair
        padded_mel = F.pad(mel, (0, max_mel_length - mel.size(1)), value=mel_min_value)
        padded_audio = F.pad(audio, (0, max_audio_length - audio.size(0)))
        padded_mels.append(padded_mel)
        padded_audios.append(padded_audio)
        seq_lens.append(audio.size(0))
    return (torch.stack(padded_mels, 0), torch.stack(padded_audios, 0), torch.LongTensor(seq_lens))


class ExponentialMovingAverage:
    def __init__(self, decay):
        self.decay = decay
        self.shadow = {}

    def register(self, name, val):
        self.shadow[name] = val.clone()

    def update(self, name, x):
        assert name in self.shadow
        new_average = self.decay * x + (1.0 - self.decay) * self.shadow[name]
        self.shadow[name] = new_average.clone()


def load_checkpoint(checkpoint_path, model, optimizer, scheduler, ema):
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    iteration = checkpoint_dict['iteration']
    optimizer.load_state_dict(checkpoint_dict['optimizer'])
    scheduler.load_state_dict(checkpoint_dict['scheduler'])
    model_for_loading = checkpoint_dict['model']
    model.load_state_dict(model_for_loading.state_dict())
    print("Loaded checkpoint '{}' (iteration {})".format(
        checkpoint_path, iteration))
    checkpoint_dir = os.path.dirname(checkpoint_path)
    ema_path = os.path.join(checkpoint_dir, "wavenet_ema_{}".format(iteration))
    ema_dict = torch.load(ema_path, map_location='cpu')
    ema_model = ema_dict['model']
    for name, param in ema_model.named_parameters():
        if param.requires_grad:
            ema.register(name, param.data)
    return model, optimizer, scheduler, iteration, ema


def save_checkpoint(model, optimizer, scheduler, learning_rate, iteration, output_directory, ema, wavenet_config):
    checkpoint_path = "{}/wavenet_{}".format(
        output_directory, iteration)
    print("Saving model and optimizer state at iteration {} to {}".format(
        iteration, checkpoint_path))
    model_for_saving = WaveNet(**wavenet_config).cuda()
    model_for_saving.load_state_dict(model.state_dict())
    torch.save({'model': model_for_saving,
                'iteration': iteration,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'learning_rate': learning_rate}, checkpoint_path)
    ema_path = "{}/wavenet_ema_{}".format(
        output_directory, iteration)
    print("Saving ema model at iteration {} to {}".format(
        iteration, ema_path))

    state_dict = model_for_saving.state_dict()
    for name, _ in model.named_parameters():
        if name in ema.shadow:
            state_dict[name] = ema.shadow[name]
    model_for_saving.load_state_dict(state_dict)
    torch.save({'model': model_for_saving,
                'iteration': iteration,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'learning_rate': learning_rate}, ema_path)