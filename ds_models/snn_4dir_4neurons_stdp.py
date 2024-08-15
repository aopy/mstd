"""
Spiking Neural Network (SNN) Model using STDP Learning with Moving Bars Stimulus and Lateral Inhibition

This script implements a Spiking Neural Network (SNN) model trained using Spike-Timing-Dependent Plasticity (STDP)
to classify the direction of moving bars stimulus. The SNN comprises Leaky Integrate-and-Fire (LIF) neurons with
lateral inhibition, where the first neuron that spikes is declared the winner and inhibits the other neurons.

Key Features:
1. **Four Directions/Four Neurons**: The network is trained to recognize four directions of a moving bar:
   left-to-right, right-to-left, top-to-bottom, and bottom-to-top. There are four neurons, each expected to become
   selective to one direction after training.
2. **STDP Learning**: The model uses Spike-Timing-Dependent Plasticity (STDP) for learning.
3. **Lateral Inhibition**: During stimulus presentation, the first neuron that spikes is declared the winner and
   inhibits the other neurons.
4. **Single Linear Layer**: The model uses a single linear layer to process the input stimuli.
5. **Stimulus Generation**: The `create_moving_bars_stimulus_with_delay_and_labels` function generates moving bars
   stimulus as tensors, with each frame having a delayed version to imitate synaptic delay, thereby facilitating
   motion direction selectivity.
6. **Unsupervised Learning**: No labels are used as the training is unsupervised.
"""


import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, layer, learning, functional
from spikingjelly.activation_based.base import MemoryModule
import random
from matplotlib import pyplot as plt

# Seed for reproducibility
# random.seed(5)
# torch.manual_seed(5)

direction_choice = ''
right_count = 0
left_count = 0
le = 0
ri = 0
up = 0
dn = 0
n0 = 0
n1 = 0
n2 = 0
n3 = 0


class LateralInhibitionLIFNode(neuron.LIFNode):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.inhibition_enabled = True
        self.winner_idx = None
        self.inhibited_neurons_mask = None  # Tracks which neurons are inhibited
        self.previous_v = None  # To store the previous membrane potentials
        self.first_spike_has_occurred = False  # Indicates if the first spike in the stimulus has occurred

    def forward(self, x):
        global n0, n1, n2, n3
        # Initialize previous_v if it's the first call and self.v is already a tensor
        if self.previous_v is None and isinstance(self.v, torch.Tensor):
            self.previous_v = torch.zeros_like(self.v)

        current_spikes = super().forward(x)  # Get current spikes from LIF dynamics
        # print("original spikes ", current_spikes)
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
        if self.winner_idx == 0:
            n0 += 1
        elif self.winner_idx == 1:
            n1 += 1
        elif self.winner_idx == 2:
            n2 += 1
        elif self.winner_idx == 3:
            n3 += 1

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
        self.fc = nn.Linear(input_size, 4, bias=False)
        # self.lif_neurons = LateralInhibitionLIFNode(tau=2.0, v_threshold=4.0)
        self.lif_neurons = LateralInhibitionLIFNode(tau=2.0, v_threshold=5.0)
        # self.lif_neurons = LateralInhibitionLIFNode(tau=10.0, v_threshold=1.0)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc(x)
        x = self.lif_neurons(x)
        return x

    def reset(self):
        super().reset()  # Reset inherited from MemoryModule
        self.lif_neurons.reset()


direction_counter = 0  # Add a counter at the global level


def create_moving_bars_stimulus_with_delay_and_labels(batch_size, width, height, bar_width, time_step, synaptic_delay=1, direction=""):
    global direction_choice, ri, le, up, dn
    global direction_counter

    # Define directions in a fixed order
    directions = ["up", "down", "right", "left"]

    # moving bars stimulus with synaptic delay and labels
    current_stimulus = torch.zeros(batch_size, height, width)
    delayed_stimulus = torch.zeros(batch_size, height, width)

    # Determine the direction based on the counter
    if time_step == 0:
        direction_choice = directions[direction_counter % 4]
        direction_counter += 1  # Increment the counter to change direction in the next cycle

    if 1 <= time_step <= 10:
        if direction_choice == "down":
            current_position = min(height - 1, time_step-1)
            current_stimulus[:, current_position:current_position + bar_width, :] = 1
        elif direction_choice == "up":
            current_position = max(0, height - time_step+1 - 1)
            current_stimulus[:, current_position:current_position + bar_width, :] = 1
        elif direction_choice == "right":
            current_position = min(width - 1, time_step-1)
            current_stimulus[:, :, current_position:current_position + bar_width] = 1
        elif direction_choice == "left":
            current_position = max(0, width - time_step + 1 - 1)
            current_stimulus[:, :, current_position:current_position + bar_width] = 1

    if 2 <= time_step <= 11:
        if direction_choice == "down":
            delayed_position = min(height - 1, time_step-2)
            delayed_stimulus[:, delayed_position:delayed_position + bar_width, :] = 1
        elif direction_choice == "up":
            delayed_position = max(0, height - time_step+2 - 1)
            delayed_stimulus[:, delayed_position:delayed_position + bar_width, :] = 1
        elif direction_choice == "right":
            delayed_position = min(width - 1, time_step-2)
            delayed_stimulus[:, :, delayed_position:delayed_position + bar_width] = 1
        elif direction_choice == "left":
            delayed_position = max(0, width - time_step + 2 - 1)
            delayed_stimulus[:, :, delayed_position:delayed_position + bar_width] = 1

    label = torch.zeros(4, dtype=torch.float32)
    label[directions.index(direction_choice)] = 1  # Set the corresponding label index to 1
    if direction_choice == 'right':
        ri += 1
    elif direction_choice == 'left':
        le += 1
    elif direction_choice == 'up':
        up += 1
    elif direction_choice == 'down':
        dn += 1

    return torch.stack([current_stimulus, delayed_stimulus], dim=1), label


def plot_weights(weights, input_shape=(10, 10), num_channels=2):
    # We expect 'weights' to be of shape [num_neurons, num_features]
    num_neurons = weights.shape[0]
    num_features_per_channel = input_shape[0] * input_shape[1]

    fig, axs = plt.subplots(num_neurons, num_channels, figsize=(num_channels * 5, num_neurons * 5))
    for neuron_idx in range(num_neurons):
        for channel_idx in range(num_channels):
            # Calculate the start and end index for each channel's weights
            start_idx = channel_idx * num_features_per_channel
            end_idx = start_idx + num_features_per_channel

            # Reshape the weights for the current neuron and channel
            neuron_weights = weights[neuron_idx, start_idx:end_idx].view(input_shape)

            # Select the appropriate subplot
            ax = axs[neuron_idx, channel_idx] if num_neurons > 1 else axs[channel_idx]

            # Plot the weights
            im = ax.imshow(neuron_weights.detach().numpy(), cmap='viridis', origin='upper')
            ax.set_title(f'Neuron {neuron_idx + 1}, Channel {channel_idx + 1}')
            ax.axis('off')
            plt.colorbar(im, ax=ax)

    plt.tight_layout()
    # plt.show()


if __name__ == '__main__':
    # Network parameters
    N_in, N_out = 200, 4

    # S, batch_size, width, height, bar_width = 410, 1, 10, 10, 1  # 40 39

    # S, batch_size, width, height, bar_width = 20000, 1, 10, 10, 1  # 40 39 # try running many more times
    S, batch_size, width, height, bar_width = 2000, 1, 10, 10, 1

    # lr, w_min, w_max = 0.0004, 0.0, 0.3  # good one, also 0.0005, 0.0007, 0.0009
    lr, w_min, w_max = 0.0008, 0.0, 0.3  # best so far
    # lr, w_min, w_max = 0.02, 0.0, 0.3

    # th = 1.0
    # th = 4.0
    th = 5.0

    net = MySpikingNetwork(input_size=200)
    net.lif_neurons.enable_inhibition()
    # net.lif_neurons.disable_inhibition()

    # nn.init.uniform_(net.fc.weight.data, 0.4, 0.5)
    # nn.init.uniform_(net.fc.weight.data, 0.1, 0.5)
    # nn.init.constant_(net.fc.weight.data, 0.5)
    nn.init.constant_(net.fc.weight.data, 0.3)
    # nn.init.constant_(net.fc.weight.data, 0.25)

    # nn.init.constant_(net.fc.weight.data, 0.29)
    # nn.init.constant_(net.fc.weight.data, 2.5)

    # torch.nn.init.normal_(net[1].weight.data, mean=0, std=0.01)
    # torch.nn.init.uniform_(net[1].weight.data, a=0.1, b=0.2)

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    learner = learning.STDPLearner(
        step_mode='s', synapse=net.fc, sn=net.lif_neurons,  # synapse=net[1], sn=net[2],

        # tau_pre=5.5, tau_post=5.5,  # good one
        # tau_pre=5.3, tau_post=5.3,  # two neurons spike twice
        # tau_pre=5.2, tau_post=5.2,  # one neuron spikes twice
        tau_pre=5.0, tau_post=5.0,  # one neuron spikes twice
        # tau_pre=4.4, tau_post=4.4,  # no spikes
        # tau_pre=5.9, tau_post=5.9,  # four neurons spikes twice
        # tau_pre=6.1, tau_post=6.1,  # four neurons spikes twice
        # tau_pre=7.0, tau_post=7.0,

        # f_pre=f_pre, f_post=f_post,
        # f_pre=lambda x: torch.clamp(x, 0.0, 0.3), f_post=lambda x: torch.clamp(x, 0.0, 0.25),  # good one
        # f_pre=lambda x: torch.clamp(x, 0.0, 0.3), f_post=lambda x: torch.clamp(x, 0.0, 0.27), # same as 0.25
        # f_pre=lambda x: torch.clamp(x, 0.0, 0.3), f_post=lambda x: torch.clamp(x, 0.0, 0.18),
        f_pre=lambda x: torch.clamp(x, 0.0, 0.3), f_post=lambda x: torch.clamp(x, 0.0, 0.25),
    )

    p = 0
    l = 0
    # Training loop
    print("TRAINING")
    for s in range(S):
        if s % 100 == 0:
            print(s)
        optimizer.zero_grad()
        # model.reset()
        for time_step in range(13):
            # print(time_step)
            combined_input, _ = create_moving_bars_stimulus_with_delay_and_labels(
                batch_size=batch_size, width=width, height=height,
                bar_width=bar_width, time_step=time_step,  # direction="right",
            )
            # print(direction_choice)
            # print(combined_input)
            output = net(combined_input)
            # output = model(combined_input)

            # print("output spikes ", output)
            # print("output membrane potentials ", net.lif_neurons.v)
            # import pdb;pdb.set_trace()
            if output[0][0] == 1:
                p = p + 1
            elif output[0][1] == 1:
                l = l + 1

            learner.step(on_grad=True)
            optimizer.step()
            # net[1].weight.data.clamp_(w_min, w_max)
            net.fc.weight.data.clamp_(w_min, w_max)
            # net.reset()
        # print(direction_choice)
        net.reset()
        functional.reset_net(net)

    # Visualize final weights
    # plot_weights(net[3].weight.data, input_shape=(10, 10), num_channels=2)

    # plot_weights(net[1].weight.data, input_shape=(10, 10), num_channels=2)
    plot_weights(net.fc.weight.data, input_shape=(10, 10), num_channels=2)
    # print("count of neuron index 0 spikes ", p)
    # print("count of neuron index 1 spikes ", l)
    print("count of neuron 0 winner ", n0)
    print("count of neuron 1 winner ", n1)
    print("count of neuron 2 winner ", n2)
    print("count of neuron 3 winner ", n3)
    print("count of right direction ", ri)
    print("count of left direction ", le)
    print("count of up direction ", up)
    print("count of down direction ", dn)

    print("TESTING---------->")

    net.eval()
    net.lif_neurons.disable_inhibition()
    test_stimuli = ['up', 'down', 'right', 'left']
    # test_stimuli = ['left', 'right']
    membrane_potentials = {direction: torch.zeros(4, 13) for direction in test_stimuli}
    spike_times_per_neuron_per_stimulus = {direction: [[] for _ in range(N_out)] for direction in test_stimuli}
    response = {direction: torch.zeros(N_out, 13) for direction in test_stimuli}

    membrane_potentials2 = {direction: [] for direction in test_stimuli}
    spikes = {direction: [] for direction in test_stimuli}

    with torch.no_grad():
        for d in test_stimuli:
            print("d ", d)
            for i in range(13):
                print("i ", i)
                # Create the moving bars stimulus
                combined_input, label = create_moving_bars_stimulus_with_delay_and_labels(batch_size=1, width=10, height=10,
                                                                                          bar_width=1, time_step=i, direction=d)
                output = net(combined_input)
                # output = model(combined_input)
                print("output ", output)
                # mp = net[2].v
                mp = net.lif_neurons.v
                # mp = model.lif_neurons.v
                print("mps ", mp)
                membrane_potentials[d][:, i] = mp
                # print(
                #     f"Time step {i}, Output: {output.squeeze().item()}, Membrane Potential: {mp.item()}")
                # Record spike times whenever a neuron spikes
                for neuron_idx in range(N_out):
                    if output[0][neuron_idx] == 1:  # Assuming output is a binary spike train
                        spike_times_per_neuron_per_stimulus[d][neuron_idx].append(i)  # Append the time step
                # import pdb;pdb.set_trace()
                response[d][:, i] = (output > 0).float()
                if d not in membrane_potentials2:
                    membrane_potentials2[d] = [mp]
                    spikes[d] = [output]
                else:
                    membrane_potentials2[d].append(mp)
                    spikes[d].append(output)
            # net.reset()
            functional.reset_net(net)
    for direction in membrane_potentials:
        membrane_potentials2[direction] = torch.stack(membrane_potentials2[direction])
        spikes[direction] = torch.stack(spikes[direction])

    if test_stimuli:

        fig, axs = plt.subplots(len(test_stimuli), 1, figsize=(10, 9 * len(test_stimuli)))
        threshold = th  # Define the threshold
        colors = ['blue', 'orange', 'red', 'green']  # Colors for Neuron 1 and Neuron 2

        for i, direction in enumerate(test_stimuli):
            for neuron_index in range(N_out):
                # Plot membrane potential
                axs[i].plot(membrane_potentials[direction][neuron_index], label=f'Neuron {neuron_index + 1}',
                            color=colors[neuron_index])
                # Plot threshold line
                axs[i].axhline(y=threshold, color='r', linestyle='--', label='Threshold' if neuron_index == 0 else "")
                # Mark spikes (assuming 'output' contains the spike information)
                spike_times = [t for t, spike in enumerate(response[direction][neuron_index]) if spike > 0]
                for t in spike_times:
                    axs[i].axvline(x=t, color=colors[neuron_index], linestyle=':',
                                   label=f'Neuron {neuron_index + 1} Spike' if t == spike_times[0] else "")

            axs[i].set_title(f'Membrane Potentials for Stimulus: {direction.capitalize()}')
            axs[i].set_xlabel('Time Step')
            axs[i].set_ylabel('Membrane Potential')
            axs[i].legend()

        plt.tight_layout()
        plt.show()
