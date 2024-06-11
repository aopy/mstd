# snn/stdp with dvs - 1 camera - no delay
# raw events

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
# random.seed(5)
# torch.manual_seed(5)


class EventDataset(Dataset):
    def __init__(self, file_path, height=720, width=1280, chunk_size=100000, max_events=None, temporal_window=1e3):
        self.file_path = file_path
        self.height = height
        self.width = width
        self.chunk_size = chunk_size
        self.max_events = max_events
        self.temporal_window = temporal_window
        self.events = self.load_events_in_chunks()

        if self.events.size == 0:
            raise ValueError("No event data found. Please check the file path.")

    def load_events_in_chunks(self):
        events = []
        try:
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
                        event_data = np.column_stack((x, y, p, t))
                        if event_data.size > 0:
                            events.append(event_data)
                else:
                    raise ValueError("The required datasets ('x', 'y', 'p', 't') are not in the 'events' group.")
        except OSError as e:
            print(f"OS error: {e}")
        except Exception as e:
            print(f"Failed to load {self.file_path}: {e}")
        print(f"Loaded {len(events)} chunks of event data.")
        return np.concatenate(events, axis=0)

    def preprocess_events(self, events):
        # Create empty ON and OFF frames
        on_frame = np.zeros((self.height, self.width), dtype=np.float32)
        off_frame = np.zeros((self.height, self.width), dtype=np.float32)

        # Clip coordinates to ensure they are within valid range
        events[:, 0] = np.clip(events[:, 0], 0, self.width - 1)
        events[:, 1] = np.clip(events[:, 1], 0, self.height - 1)

        # Populate frames with event data
        for event in events:
            x, y, polarity, timestamp = int(event[0]), int(event[1]), int(event[2]), event[3]
            if polarity == 1:
                on_frame[y, x] = 1  # Set the pixel to 1 if there is an ON event
            else:
                off_frame[y, x] = 1  # Set the pixel to 1 if there is an OFF event

        # Stack ON and OFF frames
        frame = np.stack([on_frame, off_frame], axis=0)
        return frame

    def create_frames(self):
        timestamps = self.events[:, 3]
        min_time, max_time = timestamps.min(), timestamps.max()
        current_time = min_time
        frames = []

        while current_time <= max_time:
            mask = (timestamps >= current_time) & (timestamps < current_time + self.temporal_window)
            frame_events = self.events[mask]
            frame = self.preprocess_events(frame_events)
            frames.append(frame)
            current_time += self.temporal_window

        return frames

    def __len__(self):
        return len(self.create_frames())

    def __getitem__(self, idx):
        frames = self.create_frames()
        return torch.tensor(frames[idx], dtype=torch.float32)


class LateralInhibitionLIFNode(neuron.LIFNode):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.inhibition_enabled = True
        self.winner_idx = None
        self.inhibited_neurons_mask = None  # Tracks which neurons are inhibited
        self.previous_v = None  # To store the previous membrane potentials
        self.first_spike_has_occurred = False  # Indicates if the first spike in the stimulus has occurred

    def forward(self, x):
        # Ensure self.v is a tensor
        if not isinstance(self.v, torch.Tensor):
            # self.v = torch.zeros(x.size(0), 4)
            self.v = torch.zeros(x.size(0), 8)

        # Initialize previous_v if it's the first call and self.v is already a tensor
        if self.previous_v is None or self.previous_v.shape != self.v.shape:
            print("Initializing self.previous_v")
            self.previous_v = torch.zeros_like(self.v)

        current_spikes = super().forward(x)  # Get current spikes from LIF dynamics

        if self.inhibition_enabled:
            if not self.first_spike_has_occurred and torch.any(current_spikes > 0):
                spiked_neurons = torch.where(current_spikes > 0)[1]
                if len(spiked_neurons) > 1:
                    # Get the membrane potentials of the neurons that have spiked
                    max_potentials = self.previous_v[0, spiked_neurons]
                    # Find the indices where the potential is the maximum
                    max_potential_indices = (max_potentials == torch.max(max_potentials)).nonzero(as_tuple=True)[0]
                    if len(max_potential_indices) > 1:
                        # Randomly select one of the neurons with the highest membrane potential
                        self.winner_idx = spiked_neurons[
                            max_potential_indices[torch.randint(len(max_potential_indices), (1,))]].item()
                    else:
                        self.winner_idx = spiked_neurons[max_potential_indices[0]].item()
                else:
                    self.winner_idx = spiked_neurons[0].item()

                # Set up inhibition for all other neurons
                self.inhibited_neurons_mask = torch.ones_like(current_spikes, dtype=torch.bool)
                self.inhibited_neurons_mask[0, self.winner_idx] = False
                # Apply inhibition to non-winning neurons
                self.v[self.inhibited_neurons_mask] = -5.0
                self.first_spike_has_occurred = True  # Mark that the first spike has occurred

        # Allow spikes to be processed normally, even if they are from non-winners
        output = current_spikes

        # Update previous membrane potentials after computing the output
        if self.inhibition_enabled:
            self.previous_v = self.v.clone()

        return output

    def reset(self):
        super().reset()
        self.winner_idx = None
        self.inhibited_neurons_mask = None
        self.previous_v = None  # Reset the previous membrane potentials
        self.first_spike_has_occurred = False  # Reset the first spike flag

    def enable_inhibition(self):
        self.inhibition_enabled = True

    def disable_inhibition(self):
        self.inhibition_enabled = False


class MySpikingNetwork(MemoryModule):
    def __init__(self, input_size):
        super(MySpikingNetwork, self).__init__()
        self.flatten = nn.Flatten()
        # self.fc = nn.Linear(input_size, 4, bias=False)
        self.fc = nn.Linear(input_size, 8, bias=False)
        self.lif_neurons = LateralInhibitionLIFNode(tau=2.0, v_threshold=5.0)

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


def plot_weights(weights, input_shape=(720, 1280), num_channels=2, save_path="weights"):
    num_neurons = weights.shape[0]
    num_features_per_channel = input_shape[0] * input_shape[1]

    fig, axs = plt.subplots(num_neurons, num_channels, figsize=(num_channels * 5, num_neurons * 5))
    for neuron_idx in range(num_neurons):
        for channel_idx in range(num_channels):
            start_idx = channel_idx * num_features_per_channel
            end_idx = start_idx + num_features_per_channel
            neuron_weights = weights[neuron_idx, start_idx:end_idx].view(input_shape)
            ax = axs[neuron_idx, channel_idx] if num_neurons > 1 else axs[channel_idx]
            im = ax.imshow(neuron_weights.detach().numpy(), cmap='viridis', origin='upper')
            ax.set_title(f'Neuron {neuron_idx + 1}, Channel {channel_idx + 1}')
            ax.axis('off')
            plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig(f"{save_path}.png")
    plt.close()


if __name__ == '__main__':
    # Network parameters
    N_out = 8
    # S, batch_size, width, height = 1, 1, 180, 240
    S, batch_size, width, height = 1, 1, 1280, 720
    lr, w_min, w_max = 0.0008, 0.0, 0.3
    th = 5.0
    T = 20

    # Calculate the correct input size for the fully connected layer
    input_size = 2 * width * height

    net = MySpikingNetwork(input_size=input_size)
    # net.lif_neurons.enable_inhibition()
    net.lif_neurons.disable_inhibition()
    nn.init.uniform_(net.fc.weight.data, 0.0, 0.1)
    # nn.init.constant_(net.fc.weight.data, 0.3)

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    learner = learning.STDPLearner(
        step_mode='s', synapse=net.fc, sn=net.lif_neurons,  # synapse=net[1], sn=net[2],
        tau_pre=5.0, tau_post=5.0,  # one neuron spikes twice
        f_pre=lambda x: torch.clamp(x, 0.0, 0.3), f_post=lambda x: torch.clamp(x, 0.0, 0.25),
    )

    # Load dataset
    file_path = 'data/running-easy-events_right.h5'
    # max_events = 1000000  # Set a small fraction of the recording to test
    max_events = 10000
    temporal_window = 1e3  # 1 ms window for high temporal resolution
    dataset = EventDataset(file_path, max_events=max_events, temporal_window=temporal_window)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # for data in data_loader:
    #    print(data.shape)  # Should output torch.Size([1, 2, height, width])

    # Training loop
    print("TRAINING")
    for s in range(S):
        print(s)
        # if s % 100 == 0:
        #    print(s)
        optimizer.zero_grad()
        for idx, combined_input in enumerate(load_event_camera_data(data_loader)):
            print("index ", idx)
            # print("combined_input) ", combined_input)
            # print("Processing input with shape:", combined_input.shape)
            output = net(combined_input)
            print("output ", output)
            mp = net.lif_neurons.v
            print("mps ", mp)

            learner.step(on_grad=True)
            optimizer.step()
            net.fc.weight.data.clamp_(w_min, w_max)
            # Release memory
            del combined_input
            torch.cuda.empty_cache()

        net.reset()
        functional.reset_net(net)

    # plot_weights(net.fc.weight.data, input_shape=(720, 1280), num_channels=2, save_path="weights")

    # net.eval()
    # net.lif_neurons.disable_inhibition()
