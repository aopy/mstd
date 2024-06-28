# snn/stdp with dvs - 1 camera
# raw events
# use center receptive field of 12x11 size
# delay
# dataset: https://cvg.cit.tum.de/data/datasets/visual-inertial-event-dataset
# adjusting event coordinates to correct for angular resolution differences

import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, layer, learning, functional
from spikingjelly.activation_based.base import MemoryModule
import random
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
import json
import hdf5plugin
import h5py


# Seed for reproducibility
random.seed(8)
torch.manual_seed(8)


class EventDataset(Dataset):
    def __init__(self, file_path, height=720, width=1280, camera_index=3, chunk_size=100000,
                 max_events=None, temporal_window=1e3, delay=30e3, device='cpu'):
        self.file_path = file_path
        self.height = height
        self.width = width
        self.chunk_size = chunk_size
        self.max_events = max_events
        self.temporal_window = temporal_window
        self.delay = delay
        self.cached_events = None  # Cache to store events
        self.device = device

        # Calculate aspect ratio based on FoV
        fov_horizontal = 90  # degrees
        fov_vertical = 65  # degrees
        self.aspect_ratio = math.tan(math.radians(fov_horizontal / 2)) / math.tan(math.radians(fov_vertical / 2))

        # Determine new dimensions based on aspect ratio
        self.new_width = int(self.height * self.aspect_ratio)
        self.new_height = self.height

        print(f"New dimensions of main frame: width {self.new_width}, height {self.new_height}")

        # Calculate the size of 1° of visual angle in pixels
        pixels_per_degree_horizontal = self.new_width / fov_horizontal
        pixels_per_degree_vertical = self.new_height / fov_vertical

        # Assuming 1° for both horizontal and vertical visual angle
        self.rf_width = int(pixels_per_degree_horizontal)
        self.rf_height = int(pixels_per_degree_vertical)

        print(f"New dimensions of center frame: width {self.rf_width}, height {self.rf_height}")

        # Center coordinates for the receptive field
        self.center_x = self.new_width // 2
        self.center_y = self.new_height // 2

    def load_events_in_chunks(self):
        if self.cached_events is None:
            print("Loading events from file...")
            events_list = []
            with h5py.File(self.file_path, 'r') as f:
                if 'events' in f and all(key in f['events'] for key in ['x', 'y', 'p', 't']):
                    total_events = f['events']['x'].shape[0]
                    total_to_load = min(total_events, self.max_events) if self.max_events else total_events
                    for start in range(0, total_to_load, self.chunk_size):
                        end = min(start + self.chunk_size, total_to_load)
                        x = f['events']['x'][start:end]
                        y = f['events']['y'][start:end]
                        p = f['events']['p'][start:end]
                        t = f['events']['t'][start:end]
                        events = np.column_stack((x, y, p, t))
                        events_list.append(events)
            self.cached_events = np.concatenate(events_list, axis=0)
        else:
            print("Using cached events...")

        yield self.cached_events

    def preprocess_events(self, events):
        on_frame = np.zeros((self.rf_height, self.rf_width), dtype=np.float32)
        off_frame = np.zeros((self.rf_height, self.rf_width), dtype=np.float32)

        # Calculate scaling factors
        scale_x = self.new_width / self.width
        scale_y = self.new_height / self.height

        # Adjust the coordinates based on scaling factors and round to nearest integers
        events[:, 0] = np.round(events[:, 0] * scale_x).astype(int)
        events[:, 1] = np.round(events[:, 1] * scale_y).astype(int)

        # Ensure events are within bounds of the entire frame
        events[:, 0] = np.clip(events[:, 0], 0, self.new_width - 1)
        events[:, 1] = np.clip(events[:, 1], 0, self.new_height - 1)

        # Calculate the bounds of the receptive field around the center
        x_min = self.center_x - (self.rf_width // 2)
        x_max = self.center_x + (self.rf_width // 2)
        y_min = self.center_y - (self.rf_height // 2)
        y_max = self.center_y + (self.rf_height // 2) + 1

        for event in events:
            x, y, polarity, timestamp = int(event[0]), int(event[1]), int(event[2]), event[3]
            if x_min <= x < x_max and y_min <= y < y_max:
                x_rf = x - x_min
                y_rf = y - y_min
                if 0 <= x_rf < self.rf_width and 0 <= y_rf < self.rf_height:  # Ensure indices are within bounds
                    if polarity == 1:
                        on_frame[y_rf, x_rf] = 1
                    else:
                        off_frame[y_rf, x_rf] = 1

        return on_frame, off_frame

    def create_frames_generator(self):
        events_gen = self.load_events_in_chunks()
        current_events = next(events_gen)
        timestamps = current_events[:, 3]
        min_time, max_time = timestamps.min(), timestamps.max()
        current_time = min_time

        delayed_events = np.empty((0, 4))

        while True:
            while (timestamps < current_time + self.temporal_window).any():
                try:
                    new_events = next(events_gen)
                    current_events = np.concatenate((current_events, new_events), axis=0)
                    timestamps = current_events[:, 3]
                except StopIteration:
                    break

            mask = (timestamps >= current_time) & (timestamps < current_time + self.temporal_window)
            delayed_mask = (timestamps >= current_time - self.delay) & (
                    timestamps < current_time - self.delay + self.temporal_window)

            frame_events = current_events[mask]
            delayed_frame_events = delayed_events[(delayed_events[:, 3] >= current_time - self.delay) & (
                    delayed_events[:, 3] < current_time - self.delay + self.temporal_window)]

            current_frame_on, current_frame_off = self.preprocess_events(frame_events)
            delayed_frame_on, delayed_frame_off = self.preprocess_events(delayed_frame_events)

            frame = np.stack([current_frame_on, current_frame_off, delayed_frame_on, delayed_frame_off], axis=0)
            frame = torch.tensor(frame, dtype=torch.float32).to(self.device)  # Move frame to the specified device
            yield frame

            delayed_events = np.concatenate((delayed_events, current_events[mask]), axis=0)
            delayed_events = delayed_events[delayed_events[:, 3] >= current_time - self.delay]
            current_events = current_events[~mask]
            timestamps = current_events[:, 3]
            current_time += self.temporal_window
            if current_time > max_time and current_events.size == 0:
                break

    def __len__(self):
        return 1000000  # Placeholder

    def __getitem__(self, idx):
        raise NotImplementedError("Use create_frames_generator() to iterate through the dataset.")


class LateralInhibitionLIFNode(neuron.LIFNode):
    def __init__(self, tau=2.0, v_threshold=5.0, inhibition_strength=-5.0):
        super().__init__(tau=tau, v_threshold=v_threshold)
        self.inhibition_strength = inhibition_strength
        self.inhibited_neurons_mask = None
        self.previous_v = None

    def forward(self, x):
        # Ensure self.v is a tensor
        if not isinstance(self.v, torch.Tensor):
            self.v = torch.zeros(x.size(0), x.size(1)).to(x.device)

        # Initialize previous_v if it's the first call and self.v is already a tensor
        if self.previous_v is None or self.previous_v.shape != self.v.shape:
            self.previous_v = torch.zeros_like(self.v).to(x.device)

        current_spikes = super().forward(x)  # Get current spikes from LIF dynamics

        if torch.any(current_spikes > 0):
            spiked_neurons = torch.where(current_spikes > 0)[1]
            if len(spiked_neurons) > 1:
                max_potentials = self.previous_v[0, spiked_neurons]
                max_potential_indices = (max_potentials == torch.max(max_potentials)).nonzero(as_tuple=True)[0]
                if len(max_potential_indices) > 1:
                    self.winner_idx = spiked_neurons[
                        max_potential_indices[torch.randint(len(max_potential_indices), (1,))]].item()
                else:
                    self.winner_idx = spiked_neurons[max_potential_indices[0]].item()
            else:
                self.winner_idx = spiked_neurons[0].item()

            self.inhibited_neurons_mask = torch.ones_like(current_spikes, dtype=torch.bool)
            self.inhibited_neurons_mask[0, self.winner_idx] = False
            self.v[self.inhibited_neurons_mask] = self.inhibition_strength

        output = current_spikes
        self.previous_v = self.v.clone()

        return output

    def reset(self):
        super().reset()
        self.inhibited_neurons_mask = None
        self.previous_v = None

    def enable_inhibition(self):
        self.inhibition_enabled = True

    def disable_inhibition(self):
        self.inhibition_enabled = False


class SNN(MemoryModule):
    def __init__(self, input_shape, device):
        super(SNN, self).__init__()
        self.flatten = nn.Flatten()
        input_size = input_shape[0] * input_shape[1] * input_shape[2]
        self.fc = nn.Linear(input_size, 4, bias=False)
        self.lif_neurons = LateralInhibitionLIFNode(tau=2.0, v_threshold=5.0)
        self.to(device)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc(x)
        x = self.lif_neurons(x)
        return x

    def reset(self):
        super().reset()  # Reset inherited from MemoryModule
        self.lif_neurons.reset()


def load_event_camera_data(loader):
    for data in loader:
        # print("Loaded batch of data with shape:", data.shape)
        yield data


def plot_weights(weights, input_shape=(720, 1280), num_channels=2, downsample_factor=10, save_path="weights"):
    num_neurons = weights.shape[0]
    downsampled_shape = (input_shape[0] // downsample_factor, input_shape[1] // downsample_factor)
    num_features_per_channel = input_shape[0] * input_shape[1]

    fig, axs = plt.subplots(num_neurons, num_channels, figsize=(num_channels * 5, num_neurons * 5))
    for neuron_idx in range(num_neurons):
        for channel_idx in range(num_channels):
            start_idx = channel_idx * num_features_per_channel
            end_idx = start_idx + num_features_per_channel
            neuron_weights = weights[neuron_idx, start_idx:end_idx].view(input_shape)

            # Downsample the weights for better visualization
            neuron_weights = neuron_weights.reshape(downsampled_shape[0], downsample_factor, downsampled_shape[1],
                                                    downsample_factor).mean(axis=(1, 3))

            # Normalize the weights for better visualization
            norm_weights = (neuron_weights - neuron_weights.min()) / (neuron_weights.max() - neuron_weights.min())

            ax = axs[neuron_idx, channel_idx] if num_neurons > 1 else axs[channel_idx]
            im = ax.imshow(norm_weights.cpu().detach().numpy(), cmap='viridis', origin='upper')  # Move to CPU before converting to numpy
            ax.set_title(f'Neuron {neuron_idx + 1}, Channel {channel_idx + 1}')
            ax.axis('off')
            plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig(f"{save_path}.png")
    plt.close()


if __name__ == '__main__':
    # Network parameters
    N_out = 8
    S, batch_size, width, height = 1, 1, 1280, 720
    lr, w_min, w_max = 0.0008, 0.0, 0.3

    # Calculate the correct input size for the fully connected layer
    input_shape = (4, 11, 12)  # Channels, Height, Width
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    net = SNN(input_shape, device=device)
    net.lif_neurons.enable_inhibition()
    # net.lif_neurons.disable_inhibition()
    # nn.init.uniform_(net.fc.weight.data, 0.0001, 0.01)
    nn.init.uniform_(net.fc.weight.data, 0.1, 0.3)
    # nn.init.constant_(net.fc.weight.data, 0.3)

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    learner = learning.STDPLearner(
        step_mode='s', synapse=net.fc, sn=net.lif_neurons,  # synapse=net[1], sn=net[2],
        tau_pre=5.0, tau_post=5.0,  # one neuron spikes twice
        f_pre=lambda x: torch.clamp(x, 0.0, 0.3), f_post=lambda x: torch.clamp(x, 0.0, 0.25),
    )

    # Load dataset
    file_path = 'data/running-easy-events_right.h5'
    # file_path = 'data/office-maze-events_right.h5'
    # file_path = 'data/skate-easy-events_right.h5'
    # running-easy-events_right contains 2017187149 events
    # max_events = 1000000  # Set a small fraction of the recording to test
    max_events = 10000000
    temporal_window = 10e3  # 10 ms window for high temporal resolution
    dataset = EventDataset(file_path, max_events=max_events, temporal_window=temporal_window, delay=20e3, device=device)
    # dataset = EventDataset(file_path, temporal_window=temporal_window)
    # data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)  # multiple workers/parallel loading
    frame_gen = dataset.create_frames_generator()

    # Training loop
    print("TRAINING")
    for s in range(1):
        print(s)
        # if s % 100 == 0:
        #    print(s)
        optimizer.zero_grad()
        for idx, combined_input in enumerate(frame_gen):
            print("time step (10ms) ", idx)
            # print("combined_input) ", combined_input)
            # print("Processing input with shape:", combined_input.shape)
            combined_input = torch.tensor(combined_input, dtype=torch.float32).to(device).unsqueeze(0)
            output = net(combined_input)
            # print("output ", output)
            mp = net.lif_neurons.v
            # print("mps ", mp)
            learner.step(on_grad=True)
            optimizer.step()
            net.fc.weight.data.clamp_(w_min, w_max)
            # Release memory
            del combined_input
            torch.cuda.empty_cache()

        net.reset()
        functional.reset_net(net)

    plot_weights(net.fc.weight.data, input_shape=(11, 12), num_channels=4, downsample_factor=1,
                 save_path="weights_final")

    # net.eval()
    # net.lif_neurons.disable_inhibition()
