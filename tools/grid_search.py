import torch
import torch.nn as nn
from sklearn.model_selection import ParameterGrid
from spikingjelly.activation_based import neuron, layer, learning, functional
from spikingjelly.activation_based.base import MemoryModule
import random
from torch.utils.data import Dataset, DataLoader
import numpy as np
import hdf5plugin
import h5py


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU setups
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class EventDataset(Dataset):
    def __init__(self, file_path, height=720, width=1280, chunk_size=100000,
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

        # Calculate aspect ratio based on FoV
        fov_horizontal = 90  # degrees
        fov_vertical = 65  # degrees
        self.aspect_ratio = fov_horizontal / fov_vertical

        # Determine new dimensions based on aspect ratio
        self.new_width = round(self.height * self.aspect_ratio)
        self.new_height = self.height

        print(f"New dimensions of main frame: width {self.new_width}, height {self.new_height}")

        # Calculate the size of 1° of visual angle in pixels
        pixels_per_degree_horizontal = self.new_width / fov_horizontal
        pixels_per_degree_vertical = self.new_height / fov_vertical

        # Set the receptive field size to cover 1 degree in both dimensions
        self.rf_size = int(min(pixels_per_degree_horizontal, pixels_per_degree_vertical))

        print(f"Size of receptive field (pixels per degree): {self.rf_size}")

        # Center coordinates for the receptive field
        self.center_x = self.new_width // 2
        self.center_y = self.new_height // 2

        # Define offsets for 9 receptive fields
        self.rf_offsets = [
            (-1, -1), (0, -1), (1, -1),
            (-1, 0),  (0, 0),  (1, 0),
            (-1, 1),  (0, 1),  (1, 1)
        ]

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

                        # Filter events by start_time and end_time
                        if self.start_time is not None:
                            start_idx = np.searchsorted(t, self.start_time, side='left')
                        else:
                            start_idx = 0

                        if self.end_time is not None:
                            end_idx = np.searchsorted(t, self.end_time, side='right')
                        else:
                            end_idx = len(t)

                        events = np.column_stack(
                            (x[start_idx:end_idx], y[start_idx:end_idx], p[start_idx:end_idx], t[start_idx:end_idx]))
                        events_list.append(events)
            self.cached_events = np.concatenate(events_list, axis=0)
        else:
            print("Using cached events...")

        yield self.cached_events

    def preprocess_events(self, events):
        combined_on_frames = []
        combined_off_frames = []

        # Calculate scaling factors
        scale_x = self.new_width / self.width
        scale_y = self.new_height / self.height

        # Adjust the coordinates based on scaling factors and round to nearest integers
        events[:, 0] = np.round(events[:, 0] * scale_x).astype(int)
        events[:, 1] = np.round(events[:, 1] * scale_y).astype(int)

        # Ensure events are within bounds of the entire frame
        events[:, 0] = np.clip(events[:, 0], 0, self.new_width - 1)
        events[:, 1] = np.clip(events[:, 1], 0, self.new_height - 1)

        # Precompute indices
        rf_indices = []
        for dx, dy in self.rf_offsets:
            x_min = self.center_x - (self.rf_size // 2) + dx * self.rf_size
            x_max = x_min + self.rf_size
            y_min = self.center_y - (self.rf_size // 2) + dy * self.rf_size
            y_max = y_min + self.rf_size
            rf_indices.append((x_min, x_max, y_min, y_max))

        for i, (x_min, x_max, y_min, y_max) in enumerate(rf_indices):
            on_frame = np.zeros((self.rf_size, self.rf_size), dtype=np.float32)
            off_frame = np.zeros((self.rf_size, self.rf_size), dtype=np.float32)

            for event in events:
                x, y, polarity, timestamp = int(event[0]), int(event[1]), int(event[2]), event[3]
                if x_min <= x < x_max and y_min <= y < y_max:
                    x_rf = x - x_min
                    y_rf = y - y_min
                    if polarity == 1:
                        on_frame[y_rf, x_rf] = 1
                    else:
                        off_frame[y_rf, x_rf] = 1

            combined_on_frames.append(on_frame)
            combined_off_frames.append(off_frame)

        return np.array(combined_on_frames), np.array(combined_off_frames)

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

            current_frames_on, current_frames_off = self.preprocess_events(frame_events)
            delayed_frames_on, delayed_frames_off = self.preprocess_events(delayed_frame_events)

            frames = np.stack([current_frames_on, current_frames_off, delayed_frames_on, delayed_frames_off], axis=1)
            frames = torch.tensor(frames, dtype=torch.float32).to(self.device)  # Move frames to the specified device
            yield frames

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
    def __init__(self, tau=10.0, v_threshold=10.0, v_reset=0.0, inhibition_strength=-10.0):
        super().__init__(tau=tau, v_threshold=v_threshold, v_reset=v_reset)
        self.inhibition_strength = inhibition_strength
        self.inhibited_neurons_mask = None
        self.previous_v = None

    def forward(self, x):
        batch_size = x.size(0)

        # Ensure self.v is a tensor
        if not isinstance(self.v, torch.Tensor):
            self.v = torch.zeros(x.size(0), x.size(1)).to(x.device)

        # Initialize previous_v if it's the first call and self.v is already a tensor
        if self.previous_v is None or self.previous_v.shape != self.v.shape:
            self.previous_v = torch.zeros_like(self.v).to(x.device)

        current_spikes = super().forward(x)  # Get current spikes from LIF dynamics

        for b in range(batch_size):
            if torch.any(current_spikes[b] > 0):
                spiked_neurons = torch.where(current_spikes[b] > 0)[0]
                if len(spiked_neurons) > 1:
                    max_potentials = self.previous_v[b, spiked_neurons]
                    max_potential_indices = (max_potentials == torch.max(max_potentials)).nonzero(as_tuple=True)[0]
                    if len(max_potential_indices) > 1:
                        winner_idx = spiked_neurons[
                            max_potential_indices[torch.randint(len(max_potential_indices), (1,))]].item()
                    else:
                        winner_idx = spiked_neurons[max_potential_indices[0]].item()
                else:
                    winner_idx = spiked_neurons[0].item()

                inhibited_neurons_mask = torch.ones_like(current_spikes[b], dtype=torch.bool)
                inhibited_neurons_mask[winner_idx] = False
                self.v[b, inhibited_neurons_mask] = self.inhibition_strength

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
        input_size = input_shape[1] * input_shape[2] * input_shape[3]  # Correct input size calculation
        self.fc = nn.Linear(input_size, 4, bias=False)  # Ensure the input size matches here
        self.lif_neurons = LateralInhibitionLIFNode(tau=10.0, v_threshold=10.0)
        self.to(device)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc(x)
        x = self.lif_neurons(x)
        return x

    def reset(self):
        super().reset()  # Reset inherited from MemoryModule
        self.lif_neurons.reset()


# Define the evaluation function
def evaluate_pattern_matching(params, file_path, device):
    input_shape = (9, 4, 11, 11)
    net = SNN(input_shape, device=device)
    net.lif_neurons.enable_inhibition()
    nn.init.uniform_(net.fc.weight.data, 0.001, 0.1)
    net.lif_neurons.tau = params['tau']
    net.lif_neurons.v_threshold = params['v_threshold']
    net.lif_neurons.inhibition_strength = params['inhibition_strength']
    optimizer = torch.optim.Adam(net.parameters(), lr=params['learning_rate'])

    learner = learning.STDPLearner(
        step_mode='s', synapse=net.fc, sn=net.lif_neurons,
        tau_pre=params['tau_pre'], tau_post=params['tau_post'],
        f_pre=lambda x: torch.clamp(x, 0.0, 0.3), f_post=lambda x: torch.clamp(x, 0.0, 0.3),
    )

    dataset = EventDataset(
        file_path, max_events=None, temporal_window=10e3, delay=20e3,
        start_time=25e6, end_time=26e6, device=device
    )
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)

    for epoch in range(params['epochs']):
        net.train()
        optimizer.zero_grad()
        frame_gen = dataset.create_frames_generator()
        for idx, combined_input in enumerate(frame_gen):
            combined_input = combined_input.to(device, non_blocking=True)
            output = net(combined_input)
            learner.step(on_grad=True)
            optimizer.step()
            net.fc.weight.data.clamp_(0.0, 0.3)
            del combined_input
            torch.cuda.empty_cache()
        net.reset()
        functional.reset_net(net)
        learner.reset()

    # Extract weights after training
    weight_data = net.fc.weight.data.cpu().numpy()
    pattern_score = check_for_motion_patterns(weight_data)

    return pattern_score


def check_for_motion_patterns(weights, rf_size=11):
    """
    Check the learned weights for patterns corresponding to different motion directions.
    Args:
        weights (np.ndarray): Weight matrix for each neuron, shape (num_neurons, input_size).
        rf_size (int): Size of the receptive field.
    Returns:
        float: Combined diversity and selectivity score.
    """
    num_neurons, input_size = weights.shape
    num_channels = 4  # Expected number of channels: ON, OFF, delayed ON, delayed OFF
    assert input_size == num_channels * rf_size * rf_size, "Unexpected input size"

    direction_counts = {'left': 0, 'right': 0, 'up': 0, 'down': 0}
    total_score = 0

    for neuron_idx in range(num_neurons):
        neuron_weights = weights[neuron_idx]
        if neuron_weights.ndim != 1:
            raise ValueError(f"Expected 1D array for neuron_weights, got {neuron_weights.ndim}D array instead.")

        # Reshape the weights from 1D to 3D (channels, height, width)
        neuron_weights = neuron_weights.reshape(num_channels, rf_size, rf_size)

        for channel_idx in range(num_channels):
            channel_weights = neuron_weights[channel_idx]
            if channel_weights.ndim != 2:
                print(f"Unexpected dimension for channel_weights: {channel_weights.ndim}. Data: {channel_weights}")
                raise ValueError(f"Expected 2D array for channel_weights, got {channel_weights.ndim}D array instead.")

            print(f"Channel {channel_idx} weights shape: {channel_weights.shape}")

            left_side = channel_weights[:, :rf_size // 3]
            right_side = channel_weights[:, -rf_size // 3:]
            top_side = channel_weights[:rf_size // 3, :]
            bottom_side = channel_weights[-rf_size // 3:, :]

            neuron_direction_scores = {
                'left': np.sum(right_side) - np.sum(left_side),
                'right': np.sum(left_side) - np.sum(right_side),
                'up': np.sum(bottom_side) - np.sum(top_side),
                'down': np.sum(top_side) - np.sum(bottom_side)
            }

            # Determine the direction with the highest score for this neuron and channel
            preferred_direction = max(neuron_direction_scores, key=neuron_direction_scores.get)
            if neuron_direction_scores[preferred_direction] > 0:
                direction_counts[preferred_direction] += 1
                total_score += neuron_direction_scores[preferred_direction]

    # Diversity penalty: Penalize if any one direction dominates
    diversity_penalty = sum(max(count - num_neurons / 4, 0) for count in direction_counts.values())
    total_score -= diversity_penalty

    return total_score


if __name__ == '__main__':
    # Define the parameter grid
    param_grid = {
        'tau': [10.0, 20.0, 30.0],
        'v_threshold': [10.0, 20.0, 30.0],
        'inhibition_strength': [-10.0, -20.0, -30.0],
        'learning_rate': [0.0001, 0.001, 0.01],
        'tau_pre': [5.0, 10.0, 15.0],
        'tau_post': [5.0, 10.0, 15.0],
        'epochs': [1, 3]
    }

    file_path = 'data/running-easy-events_right.h5'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # Grid search loop
    best_score = float('-inf')
    best_params = None
    seed = 8
    for params in ParameterGrid(param_grid):
        set_seed(seed)
        pattern_score = evaluate_pattern_matching(params, file_path, device)
        print(f"Params: {params}, Pattern Score: {pattern_score}")
        if pattern_score > best_score:
            best_score = pattern_score
            best_params = params

    print(f"Best Params: {best_params}, Best Pattern Score: {best_score}")
