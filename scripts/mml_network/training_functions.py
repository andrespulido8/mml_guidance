#!/usr/bin/env python3
"""NOT USED"""
import time
import torch

import numpy as np

from torch.nn.utils import clip_grad_norm_
from data_functions import get_batch, create_inout_sequences


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        criterion,
        scheduler,
        train_data,
        input_window,
        batch_size=10,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.train_data = train_data
        self.input_window = input_window
        self.batch_size = batch_size
        self.batches = range(0, len(train_data) - 1, batch_size)
        self.log_interval = int(len(train_data) / batch_size / 5)

    def train(self, epoch):
        self.model.train()  # Turn on the evaluation mode
        total_loss = 0.0
        start_time = time.time()

        for batch, i in enumerate(self.batches):
            data, targets = get_batch(
                self.train_data, i, self.batch_size, self.input_window
            )
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, targets[:, :, -2:])
            loss.backward()
            clip_grad_norm_(self.model.parameters(), 0.7)
            self.optimizer.step()

            total_loss += loss.item()

            if batch % self.log_interval == 0 and batch > 0:
                cur_loss = total_loss / self.log_interval
                elapsed = time.time() - start_time
                print(
                    "| epoch {:3d} | {:5d}/{:5d} batches | "
                    "lr {:02.10f} | {:5.2f} ms | "
                    "loss {:5.7f}".format(
                        epoch,
                        batch,
                        len(self.train_data) // self.batch_size,
                        self.scheduler.get_lr()[0],
                        elapsed * 1000 / self.log_interval,
                        cur_loss,
                    )
                )
                total_loss = 0
                start_time = time.time()

    def evaluate(self, data_source):
        self.model.eval()  # Turn on the evaluation mode
        total_loss = 0.0
        eval_batch_size = 1000
        with torch.no_grad():
            for i in range(0, len(data_source) - 1, eval_batch_size):
                data, targets = get_batch(
                    data_source, i, eval_batch_size, self.input_window
                )
                output = self.model(data)
                total_loss += (
                    len(data[0])
                    * self.criterion(output, targets[:, :, -2:]).cpu().item()
                )
        return total_loss / len(data_source)

    def model_forecast(self, sequence, device="cpu", input_window=50, output_window=1):
        self.model.eval()
        total_loss = 0.0
        test_result = torch.Tensor(0)
        truth = torch.Tensor(0)

        seq = np.pad(sequence, (0, 3), mode="constant", constant_values=(0, 0))
        seq = create_inout_sequences(seq, input_window)
        seq = seq[:-output_window].to(device)

        seq, _ = get_batch(seq, 0, 1)
        with torch.no_grad():
            for i in range(0, output_window):
                output = self.model(seq[-output_window:])
                seq = torch.cat((seq, output[-1:]))

        seq = seq.cpu().view(-1).numpy()

        return seq

    def forecast_seq(self, sequences):
        """Sequences data has to been windowed and passed through device"""
        start_timer = time.time()
        self.model.eval()
        forecast_seq = torch.empty((0, 2))
        actual = torch.empty((0, 2))
        batch_size = 100
        with torch.no_grad():
            # for i in range(0, len(sequences) - 1):
            for i in range(0, len(sequences) - 1, batch_size):
                data, target = get_batch(sequences, i, batch_size, self.input_window)
                output = self.model(data)
                # forecast_seq = torch.cat((forecast_seq,torch.unsqueeze(output[-1, :,-2:], 0).cpu()), 0)
                forecast_seq = torch.cat((forecast_seq, output[-1, :, -2:].cpu()), 0)
                a = len(target.shape)
                if a == 3:
                    # actual = torch.cat((actual, torch.unsqueeze(target[-1,:,-2:], 0).cpu()), 0)
                    actual = torch.cat((actual, target[-1, :, -2:].cpu()), 0)
                else:
                    actual = torch.cat(
                        (actual, torch.unsqueeze(target[-1, -2:], 0).cpu()), 0
                    )

        timed = time.time() - start_timer
        print(f"{timed} sec")

        return forecast_seq, actual
