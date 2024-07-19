# Compare the original frame (1280x720) with the adjusted frame (997x720) at a specific time in the event recording
# Event date: https://tumevent-vi.vision.in.tum.de/running-easy/running-easy-events_right.h5

import numpy as np
import matplotlib.pyplot as plt
import hdf5plugin
import h5py
from torch.utils.data import Dataset
import torch


class EventDataset(Dataset):
    def __init__(self, file_path, height=720, width=1280, chunk_size=100000,
                 max_events=None, start_time=None, end_time=None, device=torch.device('cpu')):
        self.file_path = file_path
        self.height = height
        self.width = width
        self.chunk_size = chunk_size
        self.max_events = max_events
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

        return self.cached_events

    def adjust_coordinates(self, events):
        # Calculate scaling factors
        scale_x = self.new_width / self.width
        scale_y = self.new_height / self.height

        # Adjust the coordinates based on scaling factors and round to nearest integers
        adjusted_x = np.round(events[:, 0] * scale_x).astype(int)
        adjusted_y = np.round(events[:, 1] * scale_y).astype(int)

        return adjusted_x, adjusted_y


# Load the dataset and filter events between 25 and 25.01 seconds
file_path = 'data/running-easy-events_right.h5'
start_time = 26000  # 26 seconds in milliseconds
end_time = 27000  # 27 seconds in milliseconds

# Initialize the dataset
dataset = EventDataset(file_path, start_time=start_time, end_time=end_time)

# Load and filter events
events = dataset.load_events_in_chunks()

# Debug: Check if filtered events are empty
print(f"Number of filtered events: {len(events)}")
if len(events) == 0:
    print("No events found in the specified time range.")

# Adjust coordinates of filtered events
adjusted_x, adjusted_y = dataset.adjust_coordinates(events)

# Separate the events based on polarity
on_events = events[events[:, 2] == 1]
off_events = events[events[:, 2] == 0]

adjusted_on_x, adjusted_on_y = dataset.adjust_coordinates(on_events)
adjusted_off_x, adjusted_off_y = dataset.adjust_coordinates(off_events)

# Plot original vs adjusted coordinates for the filtered events
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(on_events[:, 0], on_events[:, 1], s=2, c='r', label='ON events', alpha=0.6)
plt.scatter(off_events[:, 0], off_events[:, 1], s=2, c='b', label='OFF events', alpha=0.6)
plt.title('Original Event Coordinates from 26s to 27s')
plt.xlabel('x')
plt.ylabel('y')
plt.xlim(0, dataset.width)
plt.ylim(0, dataset.height)
plt.legend()
plt.gca().set_aspect('equal', adjustable='box')  # Keep aspect ratio equal

plt.subplot(1, 2, 2)
plt.scatter(adjusted_on_x, adjusted_on_y, s=2, c='r', label='ON events', alpha=0.6)
plt.scatter(adjusted_off_x, adjusted_off_y, s=2, c='b', label='OFF events', alpha=0.6)
plt.title('Adjusted Event Coordinates from 26s to 27s')
plt.xlabel('x')
plt.ylabel('y')
plt.xlim(0, dataset.new_width)
plt.ylim(0, dataset.new_height)
plt.legend()
plt.gca().set_aspect('equal', adjustable='box')  # Keep aspect ratio equal

plt.tight_layout()
plt.show()
