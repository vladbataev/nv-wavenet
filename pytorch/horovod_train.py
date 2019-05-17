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
import argparse
import json
import os
import time
import torch

from torch import nn
import nv_wavenet
import utils
import horovod.torch as hvd


from tensorboardX import SummaryWriter

# =====START: ADDED FOR DISTRIBUTED======
from torch.utils.data.distributed import DistributedSampler
# =====END:   ADDED FOR DISTRIBUTED======

from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from wavenet import WaveNet
from mel2samp_onehot import Mel2SampOnehot
from utils import to_gpu


class CrossEntropyLoss(torch.nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.num_classes = wavenet_config["n_out_channels"]

    def forward(self, inputs, targets):
        """
        inputs are batch by num_classes by sample
        targets are batch by sample
        torch CrossEntropyLoss needs
            input = batch * samples by num_classes
            targets = batch * samples
        """
        targets = targets.view(-1)
        inputs = inputs.transpose(1, 2)
        inputs = inputs.contiguous()
        inputs = inputs.view(-1, self.num_classes)
        return torch.nn.CrossEntropyLoss()(inputs, targets)


def sequence_mask(sequence_length, max_len=None):
    if max_len is None:
        max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.arange(0, max_len).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand.cuda()
    seq_length_expand = sequence_length.unsqueeze(1) \
        .expand_as(seq_range_expand)
    return (seq_range_expand < seq_length_expand).float()


class MaskedCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(MaskedCrossEntropyLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss(reduce=False)
        self.num_classes = wavenet_config["n_out_channels"]

    def forward(self, input, target, lengths):
        maxlen = torch.max(lengths)
        mask = sequence_mask(lengths, maxlen)
        # (B, T, D)
        losses = self.criterion(input, target)
        return ((losses * mask).sum()) / mask.sum()


def load_checkpoint(checkpoint_path, model, optimizer, scheduler):
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    iteration = checkpoint_dict['iteration']
    optimizer.load_state_dict(checkpoint_dict['optimizer'])
    scheduler.load_state_dict(checkpoint_dict['scheduler'])
    model_for_loading = checkpoint_dict['model']
    model.load_state_dict(model_for_loading.state_dict())
    print("Loaded checkpoint '{}' (iteration {})".format(
        checkpoint_path, iteration))
    return model, optimizer, scheduler, iteration


def save_checkpoint(model, optimizer, scheduler, learning_rate, iteration, filepath):
    print("Saving model and optimizer state at iteration {} to {}".format(
        iteration, filepath))
    model_for_saving = WaveNet(**wavenet_config).cuda()
    model_for_saving.load_state_dict(model.state_dict())
    torch.save({'model': model_for_saving,
                'iteration': iteration,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'learning_rate': learning_rate}, filepath)


def train(output_directory, epochs, learning_rate,
          iters_per_checkpoint, iters_per_eval, batch_size, seed, checkpoint_path, log_dir):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # =====START: ADDED FOR DISTRIBUTED======
    if num_gpus > 1:
        hvd.init()
    torch.cuda.set_device(hvd.local_rank())
    # =====END:   ADDED FOR DISTRIBUTED======

    if train_data_config["no_chunks"]:
        criterion = MaskedCrossEntropyLoss()
    else:
        criterion = CrossEntropyLoss()
    model = WaveNet(**wavenet_config).cuda()

    trainset = Mel2SampOnehot(audio_config=audio_config, verbose=True, **train_data_config)
    validset = Mel2SampOnehot(audio_config=audio_config, verbose=False, **valid_data_config)
    # =====START: ADDED FOR DISTRIBUTED======
    train_sampler = DistributedSampler(trainset, num_replicas=hvd.size(), rank=hvd.rank()) if num_gpus > 1 else None
    valid_sampler = DistributedSampler(validset, num_replicas=hvd.size(), rank=hvd.rank()) if num_gpus > 1 else None
    # =====END:   ADDED FOR DISTRIBUTED======
    print(train_data_config)
    if train_data_config["no_chunks"]:
        collate_fn = utils.collate_fn
    else:
        collate_fn = torch.utils.data.dataloader.default_collate
    train_loader = DataLoader(trainset, num_workers=1, shuffle=False,
                              collate_fn=collate_fn,
                              sampler=train_sampler,
                              batch_size=batch_size,
                              pin_memory=True,
                              drop_last=True)
    valid_loader = DataLoader(validset, num_workers=1, shuffle=False,
                              sampler=valid_sampler, batch_size=1, pin_memory=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())

    scheduler = StepLR(optimizer, step_size=200000, gamma=0.5)

    # Load checkpoint if one exists
    iteration = 0
    if checkpoint_path != "":
        model, optimizer, scheduler, iteration = load_checkpoint(checkpoint_path, model,
                                                                 optimizer, scheduler)
        iteration += 1  # next iteration is iteration + 1

    # Get shared output_directory ready
    if hvd.rank() == 0:
        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)
            os.chmod(output_directory, 0o775)
        print("output directory", output_directory)

    hvd.broadcast_parameters(model.state_dict(), root_rank=0)

    model.train()
    epoch_offset = max(0, int(iteration / len(train_loader)))
    writer = SummaryWriter(log_dir)
    print("Checkpoints writing to: {}".format(log_dir))
    # ================ MAIN TRAINNIG LOOP! ===================
    for epoch in range(epoch_offset, epochs):
        print("Epoch: {}".format(epoch))
        for i, batch in enumerate(train_loader):
            if low_memory:
                torch.cuda.empty_cache()
            scheduler.step()
            model.zero_grad()

            if train_data_config["no_chunks"]:
                x, y, seq_lens = batch
                seq_lens = to_gpu(seq_lens)
            else:
                x, y = batch
            x = to_gpu(x).float()
            y = to_gpu(y)
            x = (x, y)  # auto-regressive takes outputs as inputs
            y_pred = model(x)
            if train_data_config["no_chunks"]:
                loss = criterion(y_pred, y, seq_lens)
            else:
                loss = criterion(y_pred, y)
            reduced_loss = loss.data[0]
            loss.backward()
            optimizer.step()

            print("{}:\t{:.9f}".format(iteration, reduced_loss))
            if hvd.rank() == 0:
                writer.add_scalar('loss', reduced_loss, iteration)
            if (iteration % iters_per_checkpoint == 0 and iteration):
                if hvd.rank() == 0:
                    checkpoint_path = "{}/wavenet_{}".format(
                        output_directory, iteration)
                    save_checkpoint(model, optimizer, scheduler, learning_rate, iteration,
                                    checkpoint_path)
            if (iteration % iters_per_eval == 0 and iteration > 0 and not config["no_validation"]):
                if low_memory:
                    torch.cuda.empty_cache()
                if hvd.rank() == 0:
                    model_eval = nv_wavenet.NVWaveNet(**(model.export_weights()))
                    for j, valid_batch in enumerate(valid_loader):
                        if train_data_config["no_chunks"]:
                            mel, audio, _ = valid_batch
                        else:
                            mel, audio = valid_batch
                        mel = to_gpu(mel).float()
                        cond_input = model.get_cond_input(mel)
                        predicted_audio = model_eval.infer(cond_input, nv_wavenet.Impl.AUTO)
                        predicted_audio = utils.mu_law_decode_numpy(predicted_audio[0, :].cpu().numpy(), 256)
                        writer.add_audio("valid/predicted_audio_{}".format(j),
                                         predicted_audio,
                                         iteration,
                                         22050)
                        audio = utils.mu_law_decode_numpy(audio[0, :].cpu().numpy(), 256)
                        writer.add_audio("valid_true/audio_{}".format(j),
                                         audio,
                                         iteration,
                                         22050)
                        if low_memory:
                            torch.cuda.empty_cache()
            iteration += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str,
                        help='JSON file for configuration')
    parser.add_argument('-r', '--rank', type=int, default=0,
                        help='rank of process for distributed')
    parser.add_argument('-g', '--group_name', type=str, default='',
                        help='name of group for distributed')
    args = parser.parse_args()

    # Parse configs.  Globals nicer in this case
    with open(args.config) as f:
        data = f.read()
    config = json.loads(data)
    train_config = config["train_config"]

    global low_memory
    low_memory = config["low_memory"]

    global audio_config
    audio_config = config["audio_config"]

    global train_data_config
    train_data_config = config["train_data_config"]

    global valid_data_config
    valid_data_config = config["valid_data_config"]

    global dist_config
    dist_config = config["dist_config"]
    global wavenet_config
    wavenet_config = config["wavenet_config"]

    num_gpus = torch.cuda.device_count()
    print("Number of gpus: {}".format(num_gpus))
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    train(**train_config)
