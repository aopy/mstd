import torch
import torch.nn as nn
import random
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
import numpy as np
import hdf5plugin
import h5py
from norse.torch.functional.lif import LIFParameters, LIFState
from norse.torch.functional.stdp import stdp_step_linear, STDPParameters
from norse.torch.functional.stdp import STDPState

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

        # Set the receptive field size
        self.rf_size = 10  # Receptive field size (10x10 pixels)
        print(f"Size of receptive field (pixels per degree): {self.rf_size}")

        # Center coordinates for the receptive field
        self.center_x = self.width // 2
        self.center_y = self.height // 2

        # Define the offsets for the 3x3 grid
        offsets = [-10, 0, 10]
        self.receptive_fields = []
        for dx in offsets:
            for dy in offsets:
                x_c = self.center_x + dx
                y_c = self.center_y + dy
                x_min = x_c - (self.rf_size // 2)
                x_max = x_c + (self.rf_size // 2)
                y_min = y_c - (self.rf_size // 2)
                y_max = y_c + (self.rf_size // 2)
                # Ensure bounds are within image
                x_min = max(0, x_min)
                x_max = min(self.width, x_max)
                y_min = max(0, y_min)
                y_max = min(self.height, y_max)
                self.receptive_fields.append((x_min, x_max, y_min, y_max))

    def load_events_in_chunks(self):
        if self.cached_events is None:
            print("Loading events from file...")
            events_list = []
            with h5py.File(self.file_path, 'r') as f:
                if '/davis/right/events' in f:
                    total_events = f['/davis/right/events'].shape[0]
                    timestamps = f['/davis/right/events'][:, 2]
                    if self.start_time is None:
                        self.start_time = timestamps[0]
                    if self.end_time is None:
                        self.end_time = timestamps[-1]
                    total_to_load = min(total_events, self.max_events) if self.max_events else total_events
                    for start in range(0, total_to_load, self.chunk_size):
                        end = min(start + self.chunk_size, total_to_load)
                        events = f['/davis/right/events'][start:end]
                        start_idx = np.searchsorted(events[:, 2], self.start_time, side='left')
                        end_idx = np.searchsorted(events[:, 2], self.end_time, side='right')
                        filtered_events = events[start_idx:end_idx]
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
        if len(events) == 0:
            # Return zeros for all receptive fields
            for _ in range(9):
                on_frame = np.zeros((self.rf_size, self.rf_size), dtype=np.float32)
                off_frame = np.zeros((self.rf_size, self.rf_size), dtype=np.float32)
                frames.append(np.stack([on_frame, off_frame], axis=0))
            frames = np.stack(frames, axis=0)  # Shape: (batch_size, channels, height, width)
            return frames

        # Scaling and clamping the event coordinates remain the same
        events[:, 0] = np.clip(events[:, 0], 0, self.width - 1)
        events[:, 1] = np.clip(events[:, 1], 0, self.height - 1)

        frames = []

        for x_min, x_max, y_min, y_max in self.receptive_fields:
            on_frame = np.zeros((self.rf_size, self.rf_size), dtype=np.float32)
            off_frame = np.zeros((self.rf_size, self.rf_size), dtype=np.float32)
            for event in events:
                x = int(event[0])
                y = int(event[1])
                polarity = int(event[3])
                if x_min <= x < x_max and y_min <= y < y_max:
                    x_rf = x - x_min
                    y_rf = y - y_min
                    if 0 <= x_rf < self.rf_size and 0 <= y_rf < self.rf_size:
                        if polarity == 1:
                            on_frame[y_rf, x_rf] = 1
                        elif polarity == -1 or polarity == 0:
                            off_frame[y_rf, x_rf] = 1
            frames.append(np.stack([on_frame, off_frame], axis=0))
        frames = np.stack(frames, axis=0)  # Shape: (batch_size, channels, height, width)
        return frames

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
                    timestamps = current_events[:, 2]  # Ensure timestamps are updated correctly
                except StopIteration:
                    break

            mask = (timestamps >= current_time) & (timestamps < current_time + self.temporal_window)
            delayed_mask = (timestamps >= current_time - self.delay) & (
                    timestamps < current_time - self.delay + self.temporal_window)

            frame_events = current_events[mask]
            delayed_frame_events = delayed_events[(delayed_events[:, 2] >= current_time - self.delay) & (
                    delayed_events[:, 2] < current_time - self.delay + self.temporal_window)]

            current_frames = self.preprocess_events(frame_events)  # Shape: (batch_size, 2, height, width)
            delayed_frames = self.preprocess_events(delayed_frame_events)  # Same shape

            num_events_current = np.count_nonzero(current_frames)
            num_events_delayed = np.count_nonzero(delayed_frames)
            print(f"Current frame events: {num_events_current}, Delayed frame events: {num_events_delayed}")

            frames = np.concatenate([current_frames, delayed_frames], axis=1)  # Now shape: (batch_size, 4, height, width)
            frames = torch.tensor(frames, dtype=torch.float32).to(self.device)
            yield frames

            delayed_events = np.concatenate((delayed_events, current_events[mask]), axis=0)
            delayed_events = delayed_events[delayed_events[:, 2] >= current_time - self.delay]
            current_events = current_events[~mask]
            timestamps = current_events[:, 2]  # Update timestamps correctly
            current_time += self.temporal_window
            if current_time > max_time and current_events.size == 0:
                break
        print(f"Number of events in current time window: {np.sum(mask)}")
        print(f"Number of events in delayed time window: {len(delayed_frame_events)}")

    def __len__(self):
        return 1000000  # Placeholder

    def __getitem__(self, idx):
        raise NotImplementedError("Use create_frames_generator() to iterate through the dataset.")


def custom_lif_step(input_tensor, state, p):
    # Unpack the state
    v, i = state.v, state.i

    # Compute synaptic input current
    i_new = i + p.tau_syn_inv * (input_tensor - i)

    # Update membrane potential
    v_new = v + p.tau_mem_inv * (i_new - v)

    # Check for spikes (when membrane potential crosses threshold)
    z_new = (v_new >= p.v_th).to(v_new.dtype)

    # Store pre-reset membrane potential
    v_before_reset = v_new.clone()

    # Apply reset
    v_new = torch.where(z_new > 0, p.v_reset, v_new)

    # Return new state and spike output
    new_state = LIFState(z=z_new, v=v_new, i=i_new)
    return z_new, new_state, v_before_reset


class LateralInhibitionLIFCell(nn.Module):
    def __init__(self, p=LIFParameters(tau_syn_inv=0.5, tau_mem_inv=0.5, v_th=1.0, v_reset=0.0), inhibition_strength=-5.0):
        super().__init__()
        self.p = p
        self.inhibition_strength = inhibition_strength

    def forward(self, x, state):
        if state is None:
            batch_size = x.size(0)
            neuron_count = x.size(1)
            state = LIFState(
                z=torch.zeros(batch_size, neuron_count, device=x.device),
                v=torch.zeros(batch_size, neuron_count, device=x.device),
                i=torch.zeros(batch_size, neuron_count, device=x.device),
            )

        # Forward through custom LIF function
        z, new_state, v_before_reset = custom_lif_step(x, state, self.p)

        # Lateral inhibition logic per batch item
        batch_size = z.size(0)
        new_v = new_state.v.clone()

        for b in range(batch_size):
            z_b = z[b]
            if torch.any(z_b > 0):
                spiked_neurons = torch.nonzero(z_b > 0).squeeze()
                if spiked_neurons.numel() > 1:
                    # Get membrane potentials at the time of spike
                    v_spiked = v_before_reset[b, spiked_neurons]
                    # Find the neuron with the highest membrane potential
                    max_potential, max_index = torch.max(v_spiked, dim=0)
                    winner_idx = spiked_neurons[max_index].item()
                else:
                    winner_idx = spiked_neurons.item()
                # Apply inhibition to other neurons
                inhibition_mask = torch.ones_like(z_b, dtype=torch.bool)
                inhibition_mask[winner_idx] = False
                new_v[b][inhibition_mask] = self.inhibition_strength

        # Update the state with the modified membrane potentials
        new_state = LIFState(
            z=new_state.z,
            v=new_v,
            i=new_state.i,
        )

        return z, new_state

    def reset(self):
        pass


class SNN(nn.Module):
    def __init__(self, input_shape, device):
        super(SNN, self).__init__()
        self.flatten = nn.Flatten()
        input_size = input_shape[1] * input_shape[2] * input_shape[3]  # Adjusted for batch size
        self.fc = nn.Linear(input_size, 4, bias=False)
        self.lif_neurons = LateralInhibitionLIFCell()
        self.device = device
        self.to(device)

    def forward(self, x, state):
        batch_size = x.size(0)
        x = self.flatten(x)
        x = self.fc(x)
        z, new_state = self.lif_neurons(x, state)
        return z, new_state

    def reset(self):
        pass  # No internal state to reset in the LIF cell


def plot_weights(weights, input_shape=(10, 10), num_channels=2, save_path="weights"):
    num_neurons = weights.shape[0]
    num_features_per_channel = input_shape[1] * input_shape[2]

    fig, axs = plt.subplots(num_neurons, input_shape[0], figsize=(input_shape[0] * 5, num_neurons * 5))
    for neuron_idx in range(num_neurons):
        for channel_idx in range(input_shape[0]):
            start_idx = channel_idx * num_features_per_channel
            end_idx = start_idx + num_features_per_channel
            neuron_weights = weights[neuron_idx, start_idx:end_idx].view(input_shape[1], input_shape[2])

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
    S, batch_size, width, height = 1, 9, 346, 260  # Adjusted batch_size to 9
    lr, w_min, w_max = 0.0008, 0.0, 0.3

    # Calculate the correct input size for the fully connected layer
    input_shape = (4, 2, 10, 10)  # Adjusted Channels, Height, Width
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    net = SNN(input_shape, device=device)
    nn.init.uniform_(net.fc.weight.data, 0.1, 0.3)
    net.fc.weight.data.clamp_(w_min, w_max)

    # Define weight limits
    w_min = torch.tensor(0.0)
    w_max = torch.tensor(0.3)

    # Initialize STDP parameters
    stdp_params = STDPParameters(
        a_pre=torch.tensor(1.0),  # Contribution of presynaptic spikes to trace
        a_post=torch.tensor(-1.0),  # Contribution of postsynaptic spikes to trace (negative for depression)
        tau_pre_inv=torch.tensor(1 / 5.0),  # Inverse of presynaptic time constant (1 / tau_pre)
        tau_post_inv=torch.tensor(1 / 5.0),  # Inverse of postsynaptic time constant (1 / tau_post)
        eta_plus=torch.tensor(0.005),  # Learning rate for potentiation
        eta_minus=torch.tensor(0.005),  # Learning rate for depression
        w_min=w_min,
        w_max=w_max,
        stdp_algorithm='additive',  # Choose 'additive' or other algorithm as needed
        mu=torch.tensor(0.0),
        hardbound=True
    )

    # Load dataset
    file_path = 'data/indoor_flying1_data.hdf5'
    max_events = 100000
    dataset = EventDataset(
        file_path,
        max_events=None,
        temporal_window=0.01,  # 10 ms window for temporal resolution
        delay=0.02,
        start_time=1504645177.42 + 6,
        end_time=1504645177.42 + 15,
        device=device)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)

    # Training loop
    print("TRAINING")
    for s in range(1):
        print(s)
        frame_gen = dataset.create_frames_generator()
        neuron_state = None  # Initialize neuron state
        # Initialize t_pre and t_post tensors
        t_pre = torch.zeros_like(net.fc.weight.data)
        t_post = torch.zeros((batch_size, net.fc.out_features), device=device)

        # Create STDPState instance
        stdp_state = STDPState(
            t_pre=t_pre,
            t_post=t_post
        )
        for idx, combined_input in enumerate(frame_gen):
            print("time step (10ms) ", idx)
            combined_input = combined_input.to(device)
            # Forward pass
            z, neuron_state = net(combined_input, neuron_state)
            print(f"Membrane potentials at time {idx}: {neuron_state.v}")
            print(f"Spikes at time {idx}: {z}")

            # Reshape inputs and outputs for STDP
            z_pre = combined_input.view(combined_input.size(0), -1)
            z_post = z

            # Apply STDP update
            net.fc.weight.data, stdp_state = stdp_step_linear(
                w=net.fc.weight.data,
                z_pre=z_pre,
                z_post=z_post,
                state_stdp=stdp_state,
                p_stdp=stdp_params,
                dt=1.0
            )

            # Clamp weights
            net.fc.weight.data.clamp_(w_min, w_max)

            # Release memory
            del combined_input
            torch.cuda.empty_cache()

        net.reset()
    plot_weights(net.fc.weight.data, input_shape=(input_shape[1], input_shape[2], input_shape[3]), save_path="weights_final10x10")
