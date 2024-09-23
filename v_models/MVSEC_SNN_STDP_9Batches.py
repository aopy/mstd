"""
Spiking Neural Network (SNN) with Spike-Timing Dependent Plasticity (STDP) using Dynamic Vision Sensor (DVS) data

Model Description:
- Utilizes a single camera setup with DVS input.
- Processes a center receptive field of size 10x10 pixels.
- Contains 9 receptive fields arranged in a 3x3 grid, each processed in separate batches.
- Input consists of 4 channels: ON events, OFF events, and their respective delayed versions.
- Employs Leaky Integrate-and-Fire (LIF) neurons with lateral inhibition to enhance selectivity.
- Features a single fully connected linear layer to integrate spiking responses from the receptive fields.

Data and Preprocessing:
- Dataset: https://daniilidis-group.github.io/mvsec/

"""

import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, layer, learning, functional
from spikingjelly.activation_based.base import MemoryModule
import random
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
import numpy as np
import hdf5plugin
import h5py


# Seed for reproducibility
random.seed(8)
torch.manual_seed(8)


class EventDataset(Dataset):
    def __init__(self, file_path, height=260, width=346, chunk_size=10000,
                 max_events=None, temporal_window=1e3, delay=30e3, start_time=None, end_time=None, device=torch.device('cpu')):
        self.file_path = file_path
        self.height = height
        self.width = width
        self.chunk_size = chunk_size
        self.max_events = max_events
        self.temporal_window = temporal_window
        self.delay = delay
        self.start_time = start_time
        self.end_time = end_time
        self.cached_events = None  # Cache to store events
        self.device = device

        # Set the receptive field size to 10x10 pixels
        self.rf_size = 10  # size of each receptive field

        print(f"Size of receptive field (pixels per degree): {self.rf_size}")

        # Center coordinates for the receptive field
        self.center_x = self.width // 2
        self.center_y = self.height // 2

        # Define the positions of the 9 receptive fields in a 3x3 grid
        self.receptive_fields = []
        for i in range(-1, 2):
            for j in range(-1, 2):
                x_min = int(self.center_x + i * self.rf_size)
                y_min = int(self.center_y + j * self.rf_size)
                x_max = x_min + self.rf_size
                y_max = y_min + self.rf_size

                # Ensure that the receptive fields are within the image bounds
                x_min = max(0, x_min)
                x_max = min(self.width, x_max)
                y_min = max(0, y_min)
                y_max = min(self.height, y_max)

                self.receptive_fields.append({'x_min': x_min, 'x_max': x_max, 'y_min': y_min, 'y_max': y_max})

    def load_events_in_chunks(self):
        if self.cached_events is None:
            print("Loading events from file...")
            events_list = []
            with h5py.File(self.file_path, 'r') as f:
                if '/davis/right/events' in f:
                    total_events = f['/davis/right/events'].shape[0]
                    print(f"Total events available: {total_events}")

                    timestamps = f['/davis/right/events'][:, 2]
                    print(f"First event timestamp: {timestamps[0]}, Last event timestamp: {timestamps[-1]}")

                    # Adjust start_time and end_time based on actual timestamps
                    if self.start_time is None:
                        self.start_time = timestamps[0]
                    if self.end_time is None:
                        self.end_time = timestamps[-1]

                    print(f"Adjusted Start time: {self.start_time}, End time: {self.end_time}")

                    total_to_load = min(total_events, self.max_events) if self.max_events else total_events
                    print(f"Total events to load: {total_to_load}")

                    for start in range(0, total_to_load, self.chunk_size):
                        end = min(start + self.chunk_size, total_to_load)
                        events = f['/davis/right/events'][start:end]

                        # Use the correct column for timestamps (column 2)
                        start_idx = np.searchsorted(events[:, 2], self.start_time, side='left')
                        end_idx = np.searchsorted(events[:, 2], self.end_time, side='right')

                        filtered_events = events[start_idx:end_idx]
                        print(
                            f"Chunk {start} to {end} — Loaded events from {start_idx} to {end_idx}, total loaded: {len(filtered_events)}")

                        events_list.append(filtered_events)

                else:
                    print("No events found for the right camera.")

            if len(events_list) > 0:
                self.cached_events = np.concatenate(events_list, axis=0)
            else:
                print("No events were loaded.")
                self.cached_events = np.empty((0, 4))
        else:
            print("Using cached events...")

        yield self.cached_events

    def preprocess_events(self, events):
        frames = []
        for idx, rf in enumerate(self.receptive_fields):
            x_min = rf['x_min']
            x_max = rf['x_max']
            y_min = rf['y_min']
            y_max = rf['y_max']

            on_frame = np.zeros((self.rf_size, self.rf_size), dtype=np.float32)
            off_frame = np.zeros((self.rf_size, self.rf_size), dtype=np.float32)

            # Select events within this receptive field
            mask = (events[:, 0] >= x_min) & (events[:, 0] < x_max) & (events[:, 1] >= y_min) & (events[:, 1] < y_max)
            rf_events = events[mask]

            # Map event coordinates to receptive field coordinates
            x_rf = rf_events[:, 0] - x_min
            y_rf = rf_events[:, 1] - y_min
            polarities = rf_events[:, 3]

            # Set events in on_frame and off_frame
            for x, y, p in zip(x_rf, y_rf, polarities):
                x_idx = int(x)
                y_idx = int(y)
                if x_idx >= 0 and x_idx < self.rf_size and y_idx >= 0 and y_idx < self.rf_size:
                    if p == 1:
                        on_frame[y_idx, x_idx] = 1
                    elif p == -1 or p == 0:
                        off_frame[y_idx, x_idx] = 1

            frames.append((on_frame, off_frame))

        return frames  # List of (on_frame, off_frame)

    def create_frames_generator(self):
        events_gen = self.load_events_in_chunks()
        current_events = next(events_gen)
        timestamps = current_events[:, 2]
        min_time, max_time = timestamps.min(), timestamps.max()
        current_time = min_time

        delayed_events = np.empty((0, 4))

        while True:
            while (timestamps < current_time + self.temporal_window).any():
                try:
                    new_events = next(events_gen)
                    current_events = np.concatenate((current_events, new_events), axis=0)
                    timestamps = current_events[:, 2]
                except StopIteration:
                    break

            mask = (timestamps >= current_time) & (timestamps < current_time + self.temporal_window)
            delayed_mask = (timestamps >= current_time - self.delay) & (
                    timestamps < current_time - self.delay + self.temporal_window)

            frame_events = current_events[mask]
            delayed_frame_events = delayed_events[(delayed_events[:, 2] >= current_time - self.delay) & (
                    delayed_events[:, 2] < current_time - self.delay + self.temporal_window)]

            # Process frames for each receptive field
            frames = []
            current_frames = self.preprocess_events(frame_events)
            delayed_frames = self.preprocess_events(delayed_frame_events)

            # For each receptive field, stack the current and delayed frames
            for (current_on, current_off), (delayed_on, delayed_off) in zip(current_frames, delayed_frames):
                frame = np.stack([current_on, current_off, delayed_on, delayed_off], axis=0)
                frames.append(frame)

            # Stack frames to create a batch
            batch_frames = np.stack(frames, axis=0)  # Shape: (batch_size, channels, height, width)
            batch_frames = torch.tensor(batch_frames, dtype=torch.float32).to(self.device)

            yield batch_frames

            delayed_events = np.concatenate((delayed_events, current_events[mask]), axis=0)
            delayed_events = delayed_events[delayed_events[:, 2] >= current_time - self.delay]
            current_events = current_events[~mask]
            timestamps = current_events[:, 2]
            current_time += self.temporal_window
            if current_time > max_time and current_events.size == 0:
                break

    def __len__(self):
        return 1000000  # Placeholder

    def __getitem__(self, idx):
        raise NotImplementedError("Use create_frames_generator() to iterate through the dataset.")


class LateralInhibitionLIFNode(neuron.LIFNode):
    def __init__(self, tau=2.0, v_threshold=5.0, v_reset=0.0, inhibition_strength=-5.0):
        super().__init__(tau=tau, v_threshold=v_threshold, v_reset=v_reset)
        self.inhibition_strength = inhibition_strength
        self.inhibited_neurons_mask = None
        self.previous_v = None

    def forward(self, x):
        # x is of shape (batch_size, num_neurons)
        if not isinstance(self.v, torch.Tensor):
            self.v = torch.zeros_like(x).to(x.device)

        if self.previous_v is None or self.previous_v.shape != self.v.shape:
            self.previous_v = torch.zeros_like(self.v).to(x.device)

        current_spikes = super().forward(x)  # Get current spikes from LIF dynamics

        # Process each sample in the batch separately
        batch_size = x.size(0)
        print("batch_size ", batch_size)
        for b in range(batch_size):
            if torch.any(current_spikes[b] > 0):
                spiked_neurons = torch.where(current_spikes[b] > 0)[0]
                if len(spiked_neurons) > 1:
                    max_potentials = self.previous_v[b, spiked_neurons]
                    max_potential_indices = (max_potentials == torch.max(max_potentials)).nonzero(as_tuple=True)[0]
                    if len(max_potential_indices) > 1:
                        winner_idx = spiked_neurons[max_potential_indices[torch.randint(len(max_potential_indices), (1,))]].item()
                    else:
                        winner_idx = spiked_neurons[max_potential_indices[0]].item()
                else:
                    winner_idx = spiked_neurons[0].item()

                inhibited_neurons_mask = torch.ones_like(current_spikes[b], dtype=torch.bool)
                inhibited_neurons_mask[winner_idx] = False
                self.v[b][inhibited_neurons_mask] = self.inhibition_strength

        self.previous_v = self.v.clone()

        return current_spikes

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
        # x is of shape (batch_size, channels, height, width)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.lif_neurons(x)
        return x

    def reset(self):
        super().reset()  # Reset inherited from MemoryModule
        self.lif_neurons.reset()


def plot_weights(weights, input_shape=(10, 10), num_channels=2, save_path="weights"):
    num_neurons = weights.shape[0]
    num_features_per_channel = input_shape[0] * input_shape[1]

    fig, axs = plt.subplots(num_neurons, num_channels, figsize=(num_channels * 10, num_neurons * 10))
    for neuron_idx in range(num_neurons):
        for channel_idx in range(num_channels):
            start_idx = channel_idx * num_features_per_channel
            end_idx = start_idx + num_features_per_channel
            neuron_weights = weights[neuron_idx, start_idx:end_idx].view(input_shape)

            ax = axs[neuron_idx, channel_idx] if num_neurons > 1 else axs[channel_idx]
            # Move to CPU before converting to numpy
            im = ax.imshow(neuron_weights.cpu().detach().numpy(), cmap='viridis', origin='upper')
            ax.set_title(f'Neuron {neuron_idx + 1}, Channel {channel_idx + 1}')
            ax.axis('off')
            plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig(f"{save_path}.png")
    plt.close()


if __name__ == '__main__':
    # Network parameters
    N_out = 4
    S, batch_size, width, height = 1, 9, 346, 260  # width=346, height=260
    lr, w_min, w_max = 0.0008, 0.0, 0.3

    # Calculate the correct input size for the fully connected layer
    input_shape = (4, 10, 10)  # Channels, Height, Width
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    net = SNN(input_shape, device=device)
    net.lif_neurons.enable_inhibition()
    # net.lif_neurons.disable_inhibition()
    nn.init.uniform_(net.fc.weight.data, 0.1, 0.3)
    # nn.init.constant_(net.fc.weight.data, 0.3)

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    learner = learning.STDPLearner(
        step_mode='s', synapse=net.fc, sn=net.lif_neurons,  # synapse=net[1], sn=net[2],
        tau_pre=5.0, tau_post=5.0,  # one neuron spikes twice
        f_pre=lambda x: torch.clamp(x, 0.0, 0.3), f_post=lambda x: torch.clamp(x, 0.0, 0.25),
    )

    # Load dataset
    file_path = 'data/indoor_flying1_data.hdf5'
    # max_events = 1000000  # Set a small fraction of the recording to test

    max_events = 100000
    dataset = EventDataset(
        file_path,
        max_events=None,
        temporal_window=0.01,  # 10 ms window for temporal resolution
        delay=0.02,
        start_time=1504645177.42,
        end_time=1504645177.42 + 1,
        device=device)
    # dataset = EventDataset(file_path, temporal_window=temporal_window)
    # data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)  # multiple workers/parallel loading

    # Training loop
    print("TRAINING")
    for s in range(1):
        print(s)
        optimizer.zero_grad()
        frame_gen = dataset.create_frames_generator()
        for idx, combined_input in enumerate(frame_gen):
            # print("time step (10ms) ", idx)
            # print(combined_input)
            # combined_input will be of shape (batch_size, channels, height, width)
            output = net(combined_input)
            mp = net.lif_neurons.v
            learner.step(on_grad=True)
            optimizer.step()
            net.fc.weight.data.clamp_(w_min, w_max)
            # Release memory
            del combined_input
            torch.cuda.empty_cache()

        net.reset()
        functional.reset_net(net)

    plot_weights(net.fc.weight.data, input_shape=(10, 10), num_channels=4,
                 save_path="weights_final10x10")
