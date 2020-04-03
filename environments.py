import os
from PIL import Image

from torch.utils.data import Dataset
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import random
import pickle as pkl
import numpy as np
import pandas as pd

from signal_utils import get_multi_channel_spectral_array

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MnistMetaEnv:
    def __init__(self, height=28, length=28):
        self.channels = 1
        self.height = height
        self.length = length
        self.data = datasets.MNIST(root='./data', train=True, download=True)
        self.make_tasks()
        self.split_validation_and_training_task()
        self.to_tensor = transforms.ToTensor()
        self.resize = transforms.Resize((self.height, self.length))
        self.sample_training_task()

    def sample_training_task(self, batch_size=64):
        task = str(random.sample(self.training_task, 1)[0])
        task_idx = random.sample(self.task_to_examples[task], batch_size)

        batch = torch.tensor(np.array([self.to_tensor(self.resize(self.data[idx][0])).numpy() for idx in task_idx]),
                             dtype=torch.float, device=device)
        return batch, task

    def sample_validation_task(self, batch_size=64):
        task = str(random.sample(self.validation_task, 1)[0])
        task_idx = random.sample(self.task_to_examples[task], batch_size)

        batch = torch.tensor(np.array([self.to_tensor(self.resize(self.data[idx][0])).numpy() for idx in task_idx]),
                             dtype=torch.float, device=device)
        return batch, task

    def make_tasks(self):
        self.task_to_examples = {}
        self.all_tasks = set(self.data.train_labels.numpy())
        for i, digit in enumerate(self.data.train_labels.numpy()):
            if str(digit) not in self.task_to_examples:
                self.task_to_examples[str(digit)] = []
            self.task_to_examples[str(digit)].append(i)

    def split_validation_and_training_task(self):
        self.validation_task = {9}
        self.training_task = self.all_tasks - self.validation_task


class EMGSignalMetaEnv(Dataset):
    def __init__(self, length, output_dir):
        self.channels = self.image_channels = 8
        self.length = self.window_size = length
        self.samp_freq = 1024
        self.spectral_type = 'stft'
        self.rows = list(range(0, 20))  # TODO
        self.output_dir = output_dir
        self.window_shift = 50
        self.sample_buffer = 10

        self.validation_set_size = 1

        self.to_tensor = transforms.ToTensor()

        self.tasks = self.get_tasks()
        self.all_tasks = set(self.tasks)
        self.split_validation_and_training_task()

    def get_tasks(self):
        path = './data/reb_emg_data.csv'
        if os.path.exists(path) is False:
            print("EMG Data not found! Exiting...")
            exit()

        tasks = dict()
        raw_data = pd.read_csv('%s' % path, nrows=max(self.rows) + 1)

        multi_channel_spectral = []
        for r in self.rows:
            tasks[r] = []
            x = raw_data.values[r, 3:]
            multi_channel_spectral = get_multi_channel_spectral_array(x, self.channels, self.samp_freq,
                                                                      self.spectral_type, self.output_dir,
                                                                      (128, 128))

            # create windows
            window_start_point = self.sample_buffer
            while (window_start_point + self.window_size) < (multi_channel_spectral.shape[1] - self.sample_buffer):
                sample = multi_channel_spectral[:, window_start_point:window_start_point + self.window_size, :]
                tasks[r].append(np.array(self.to_tensor(sample)))

                # shift window over
                window_start_point += self.window_shift

            tasks[r] = np.array(tasks[r])

        self.height = multi_channel_spectral.shape[0]

        return tasks

    def split_validation_and_training_task(self):
        self.validation_task = set(random.sample(self.all_tasks, self.validation_set_size))
        self.training_task = self.all_tasks - self.validation_task
        pkl.dump(self.validation_task, open('validation_task.pkl', 'wb'))
        pkl.dump(self.training_task, open('training_task.pkl', 'wb'))

    def sample_training_task(self, batch_size=4):
        task = random.sample(self.training_task, 1)[0]
        task_idx = random.sample([i for i in range(self.tasks[task].shape[0])], batch_size)
        batch = self.tasks[task][task_idx]
        batch = torch.tensor(batch, dtype=torch.float, device=device)
        return batch, task

    def sample_validation_task(self, batch_size=4):
        task = random.sample(self.validation_task, 1)[0]
        task_idx = random.sample([i for i in range(self.tasks[task].shape[0])], batch_size)
        batch = self.tasks[task][task_idx]
        batch = torch.tensor(batch, dtype=torch.float, device=device)
        return batch, task
