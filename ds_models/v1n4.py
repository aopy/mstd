"""
Spiking Neural Network (SNN) Model using Backpropagation with Moving Bars Stimulus

This script implements a Spiking Neural Network (SNN) model trained using backpropagation to classify
the direction of moving bars stimulus. The SNN comprises Leaky Integrate-and-Fire (LIF) neurons with an
infinite threshold. The training is performed using the AdamW optimizer and the cross-entropy loss function.

Key Features:
1. **Four Directions/Four Neurons**: The network is trained to recognize four directions of a moving bar:
   left-to-right, right-to-left, top-to-bottom, and bottom-to-top. There are four neurons, each expected to
   become selective (i.e. have high membrane potential) to one direction after training.
2. **Single Linear Layer**: The model uses a single linear layer to process the input stimuli.
3. **Stimulus Generation**: The `create_moving_bars_stimulus_with_delay_and_labels` function generates
   moving bars stimulus as tensors and corresponding labels, with each frame having a delayed version to imitate
   synaptic delay,
   thereby facilitating motion direction selectivity.
4. **Training Process**: After each stimulus presentation, the maximum membrane potentials of LIF neurons
   are used in the cross-entropy loss calculation for backpropagation.
5. **Accuracy Calculation**: The training and testing accuracy is calculated to evaluate the performance
   of the model.
6. **Weight Visualization**: The weights of the neurons are visualized as heatmaps.
"""


import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, layer, learning, functional
import torch.nn.functional as F
from matplotlib import pyplot as plt
import numpy as np
import random
# from plasticity import STDPLearner

random.seed(1)
# torch.manual_seed(0)

direction_choice = 'right'


class CustomLinear(nn.Linear):
    def forward(self, input):
        # Get the absolute values of the weights
        abs_weights = torch.abs(self.weight)
        # Apply the linear transformation with the absolute weights
        return nn.functional.linear(input, abs_weights, self.bias)


def create_moving_bars_stimulus_with_delay_and_labels(batch_size, width, height, bar_width, time_step, synaptic_delay=1, direction=""):
    # Probability of moving left, right, up, or down
    p_directions = [0.25, 0.25, 0.25, 0.25]

    # Initialize the stimulus matrices
    current_stimulus = torch.zeros(batch_size, height, width)
    delayed_stimulus = torch.zeros(batch_size, height, width)

    global direction_choice
    # Define the directions
    directions = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
    if time_step == 0:
        if direction:
            direction_choice = direction
            print("manual direction ", direction)
        else:
            print("random ")
            # Choose a random direction
            direction_choice = random.choices(["up", "down", "left", "right"], weights=p_directions)[0]
            print("random direction ", direction_choice)

    # Update the position based on the chosen direction and time step
    if 1 <= time_step <= 10:

        # Update the position based on the chosen direction and time step
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
        # Update the position based on the chosen direction and time step
        if direction_choice == "down":

            delayed_position = min(height - 1, time_step-2)
            delayed_stimulus[:, delayed_position:delayed_position + bar_width, :] = 1
        elif direction_choice == "up":

            delayed_position = max(0, height - time_step+1)
            delayed_stimulus[:, delayed_position:delayed_position + bar_width, :] = 1
        elif direction_choice == "right":

            delayed_position = min(width - 1, time_step-1 - synaptic_delay)
            delayed_stimulus[:, :, delayed_position:delayed_position + bar_width] = 1
        elif direction_choice == "left":
            delayed_position = max(0, (width - time_step + 2 - 1))
            # delayed_position = max(0, width - time_step + 1 - 1 - synaptic_delay)
            delayed_stimulus[:, :, delayed_position:delayed_position + bar_width] = 1

    # Create the label based on the chosen direction
    if direction_choice == "up":
        label = torch.tensor([1, 0, 0, 0], dtype=torch.float32)
    elif direction_choice == "down":
        label = torch.tensor([0, 1, 0, 0], dtype=torch.float32)
    elif direction_choice == "left":
        label = torch.tensor([0, 0, 1, 0], dtype=torch.float32)
    else:
        label = torch.tensor([0, 0, 0, 1], dtype=torch.float32)

    return torch.stack([current_stimulus, delayed_stimulus], dim=1), label


if __name__ == '__main__':
    # w_min, w_max = -1., 1.
    w_min, w_max = 0, 1.
    # tau_pre, tau_post = 6., 6.  # too much 20
    tau_pre, tau_post = 20., 20.
    # tau_pre, tau_post = 15.0, 15.0
    # tau_pre, tau_post = 2., 2.
    N_in = 10 * 10  # corresponding to image hight and width (pixels)
    N_out = 4   # corresponding to neurons for each direction of the moving bar (left, right, up, down)
    S = 2000
    # S = 10
    batch_size = 1
    width = 10
    height = 10
    lr = 0.1  # the good one
    # lr = 0.01  # (adam)
    # lr = 0.001

    # pl = PositiveLinear(in_features=2 * N_in, out_features=N_out)

    loss_values = []

    net = nn.Sequential(
        nn.Flatten(),  # Flatten the input
        # nn.Linear(2 * N_in, N_out, bias=False),
        # pl,
        CustomLinear(2 * N_in, N_out, bias=False),
        neuron.LIFNode(tau=10.0, v_threshold=float('inf'))  # infinite threshold
    )
    # initializes the weights of the linear layer

    # nn.init.uniform_(net[1].weight.data, -0.1, 0.1)
    # nn.init.normal_(net[1].weight.data, mean=0.1)
    nn.init.uniform_(net[1].weight.data, 0.0, 1.0)

    optimizer = torch.optim.AdamW(net.parameters(), lr=lr)

    input_size = N_in
    output_size = N_out
    bar_width = 1

    out_spike = []
    trace_pre = []
    trace_post = []
    weight = []
    potential = []

    # Collect spike times
    spike_times = []

    # Initialize an empty list to store weight data over time
    weights_over_time = []
    weight_history = []

    # List to store the selected weight value at each time step
    selected_weight_history = []

    print("TRAINING---------->")
    total_correct = 0
    total_correctx = 0
    total_samples = 0
    # accumulated_labels = []
    accumulated_labels = torch.tensor([])
    # accumulated_potentials = torch.zeros(2, 10)
    # accumulated_potentials = torch.zeros(2, 9)
    accumulated_potentials = torch.zeros(4, 13)
    # potentials_history = torch.zeros(S, 13, 4)
    potentials_history = torch.zeros(S, N_out)
    stimulus_labels = []
    print("!! accumulated_potentials ", accumulated_potentials)
    correct2 = 0
    for s in range(S):
        print("sample ", s)
        optimizer.zero_grad()
        for i in range(13):
            print("i ", i)
            # Create the moving bars stimulus
            combined_input, label = create_moving_bars_stimulus_with_delay_and_labels(batch_size=1, width=10, height=10,
                                                                                      bar_width=1, time_step=i)
            print(combined_input)
            output = net(combined_input)
            mp = net[2].v
            print("mps ", mp)
            # accumulated_potentials[:, i] = mp.squeeze()
            accumulated_potentials[:, i] = mp
            print("accumulated_potentials ", accumulated_potentials)
            # label = label.long()
            print("direction_choice ", direction_choice)
            print("label ", label)

        max_values, max_indices = torch.max(accumulated_potentials, dim=1)
        indices_tensor = torch.arange(accumulated_potentials.shape[0])
        highest_values_tensor = accumulated_potentials[indices_tensor, max_indices]
        print("highest_values_tensor ", highest_values_tensor)
        print("label ", label)

        loss = F.cross_entropy(highest_values_tensor, label)
        print("direction_choice ", direction_choice)
        # accumulated_labels = torch.tensor([])
        print("loss ", loss)

        # potentials_history[:, s] = accumulated_potentials
        # potentials_history[s] = accumulated_potentials
        potentials_history[s] = highest_values_tensor
        stimulus_labels.append(direction_choice)

        # import pdb;pdb.set_trace()
        # predicted_label = torch.argmax(accumulated_potentials, dim=0)
        active_neuron = torch.argmax(accumulated_potentials)
        # predicted_label = torch.argmax(accumulated_potentials, dim=1)
        print("active_neuron ", active_neuron)
        if active_neuron < 13:  # up
            active_neuron = torch.tensor([1, 0, 0, 0], dtype=torch.float32)
        elif 13 <= active_neuron < 26:  # down
            active_neuron = torch.tensor([0, 1, 0, 0], dtype=torch.float32)
        elif 26 <= active_neuron < 39:  # left
            active_neuron = torch.tensor([0, 0, 1, 0], dtype=torch.float32)
        else:  # right
            active_neuron = torch.tensor([0, 0, 0, 1], dtype=torch.float32)
        print("active_neuron ", active_neuron)
        # print("overall_label ", overall_label)
        # import pdb;pdb.set_trace()
        accuracy = torch.sum(active_neuron[0] == label[0]).item() / 10 * 100.0
        accuracy3 = torch.sum(active_neuron[0] == label[0]).item() * 100.0
        # accuracy3 = torch.sum(active_neuron[0] == label[0]).item() * 100.0
        # torch.equal(active_neuron, label)
        # accuracy = torch.sum(predicted_label == overall_label).item() / 10 * 100.0
        print("accuracy ", accuracy)
        print("accuracy ", accuracy3)
        correct = active_neuron == label
        # correct = predicted_label == overall_label
        print("correct ", correct)
        total_correct += correct.sum().item()
        print("total_correct ", total_correct)

        if torch.equal(active_neuron, label):
            print("active_neuron ", active_neuron)
            print("label ", label)
            total_correctx += 1
            print("total_correctx ", total_correctx)

        correct2 += (active_neuron[0] == label[0]).sum()
        print("correct2 ", correct2)
        print("total_correct ", total_correct)

        # Store the weights for later analysis
        weight_history.append(net[1].weight.data.clone())

        loss_values.append(loss.item())
        loss.backward(retain_graph=True)
        optimizer.step()
        weight.append(net[1].weight.data.clone().numpy().flatten())
        if s == S-1:
            print(s)
            # final_weights = net[1].weight.data.clone().numpy().flatten()
            final_weights = net[1].weight.data.clone()
            print("final_weights ", final_weights)
        functional.reset_net(net)
        # accumulated_potentials = torch.zeros(1, 2)
        # accumulated_potentials = torch.zeros(2, 9)
        accumulated_potentials = torch.zeros(4, 13)

    # average_accuracy = (total_correct / (10 * S)) * 100.0
    average_accuracy = (total_correct / 2 / S) * 100.0

    accuracy2 = 100 * correct2 / S
    print("Accuracy2 = {}".format(accuracy2))
    print(f'Average Accuracy over {S} samples: {average_accuracy:.2f}%')

    accuracyx = 100 * total_correctx / S
    print("AccuracyX = {}".format(accuracyx))

    # accuracy_training = total_correct / T
    # print(f'Training Overall Accuracy: {accuracy_training * 100:.2f}%')

    # Transpose the data to have time on the x-axis
    # Plot the final weights as a heatmap
    # final_weights = final_weights2[-1]

    # final_weights = weight_history[-1]

    # print("final_weights shape ", final_weights.shape)

    # final_weights_reshaped = final_weights.view(2, 4, 10, 10)

    # Reshape the weights for visualization
    # final_weights_reshaped = final_weights.view(2, 2, 10, 10)

    # Reshape the weights for visualization
    final_weights_reshaped = final_weights.view(4, 2, 100)

    # Create a 4x2 subplot grid
    fig, axs = plt.subplots(4, 2)

    # Plot the weights for each neuron
    for neuron_index in range(4):
        # Swap the indices to switch the places of the frames
        abs_weights1 = torch.abs(final_weights_reshaped[neuron_index, 1])  # Frame 2
        abs_weights2 = torch.abs(final_weights_reshaped[neuron_index, 0])  # Frame 1

        ax1 = axs[neuron_index, 0]  # Frame 2
        ax2 = axs[neuron_index, 1]  # Frame 1

        # Plot Frame 1 (synaptic delay)
        im1 = ax1.imshow(abs_weights1.detach().numpy().reshape(10, 10), cmap='viridis', origin='upper')
        ax1.set_title(f'Weights for Neuron {neuron_index + 1} Frame 1 (Synaptic Delay)')
        ax1.axis('off')

        # Plot Frame 2
        im2 = ax2.imshow(abs_weights2.detach().numpy().reshape(10, 10), cmap='viridis', origin='upper')
        ax2.set_title(f'Weights for Neuron {neuron_index + 1} Frame 2')
        ax2.axis('off')

    # Add a colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im1, cax=cbar_ax)

    plt.show()

    plt.plot(loss_values, label='Training Loss')
    plt.xlabel('Samples')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    print("TESTING---------->")
    net.eval()
    R = 4
    accumulated_potentials = torch.zeros(4, 13)
    total_correctz = 0
    # test_stimuli = ['up', 'down', 'left', 'right']
    test_stimuli = ['right', 'up', 'left', 'down']
    # membrane_potentials = {direction: [] for direction in test_stimuli}
    membrane_potentials = {direction: torch.zeros(4, 13) for direction in test_stimuli}
    with torch.no_grad():
        # for r in range(R):
        for d in test_stimuli:
            # print("sample ", r)
            print("d ", d)
            # optimizer.zero_grad()
            for i in range(13):
                print("i ", i)
                # Create the moving bars stimulus
                combined_input, label = create_moving_bars_stimulus_with_delay_and_labels(batch_size=1, width=10, height=10,
                                                                                          bar_width=1, time_step=i, direction=d)
                print("input ", combined_input)
                print("label ", label)
                # print(combined_input)

                output = net(combined_input)
                mp = net[2].v
                print("mps ", mp)
                membrane_potentials[d][:, i] = mp
                # accumulated_potentials[:, i] = mp
                # membrane_potentials[d][:, i] = mp
                print("accumulated_potentials ", membrane_potentials[d])
                print("direction_choice ", direction_choice)

            # membrane_potentials[d] = accumulated_potentials.detach().numpy().T
            # membrane_potentials[d] = accumulated_potentials.detach().numpy()
            print("membrane_potentials ", membrane_potentials)

            max_values, max_indices = torch.max(membrane_potentials[d], dim=1)
            indices_tensor = torch.arange(membrane_potentials[d].shape[0])
            highest_values_tensor = membrane_potentials[d][indices_tensor, max_indices]
            print("highest_values_tensor ", highest_values_tensor)
            print("label ", label)

            print("direction_choice ", direction_choice)

            # predicted_label = torch.argmax(accumulated_potentials, dim=0)
            active_neuron = torch.argmax(membrane_potentials[d])
            # predicted_label = torch.argmax(accumulated_potentials, dim=1)
            print("active_neuron ", active_neuron)
            if active_neuron < 13:  # up
                active_neuron = torch.tensor([1, 0, 0, 0], dtype=torch.float32)
            elif 13 <= active_neuron < 26:  # down
                active_neuron = torch.tensor([0, 1, 0, 0], dtype=torch.float32)
            elif 26 <= active_neuron < 39:  # left
                active_neuron = torch.tensor([0, 0, 1, 0], dtype=torch.float32)
            else:  # right
                active_neuron = torch.tensor([0, 0, 0, 1], dtype=torch.float32)
            print("active_neuron ", active_neuron)
            # print("overall_label ", overall_label)
            # import pdb;pdb.set_trace()

            if torch.equal(active_neuron, label):
                print("active_neuron ", active_neuron)
                print("label ", label)
                total_correctz += 1
                print("total_correctz ", total_correctz)

            # Store the weights for later analysis
            # weight_history.append(net[1].weight.data.clone())

            # weight.append(net[1].weight.data.clone().numpy().flatten())
            # if r == R-1:
            #    print(r)
            #    final_weights = net[1].weight.data.clone().numpy().flatten()
            #    final_weights = net[1].weight.data.clone()
            #    print("final_weights ", final_weights)
            functional.reset_net(net)
            # accumulated_potentials = torch.zeros(1, 2)
            # accumulated_potentials = torch.zeros(2, 9)
            # accumulated_potentials = torch.zeros(2, 13)

    accuracyz = 100 * total_correctz / R
    print("AccuracyZ = {}".format(accuracyz))

    # Plot the time course of the membrane potentials for each stimulus direction
    # fig, axs = plt.subplots(len(test_stimuli), 1, figsize=(10, 6 * len(test_stimuli)))
    fig, axs = plt.subplots(len(test_stimuli), 1, figsize=(10, 9 * len(test_stimuli)))

    for i, direction in enumerate(test_stimuli):
        for neuron_index in range(N_out):
            axs[i].plot(membrane_potentials[direction][neuron_index], label=f'Neuron {neuron_index + 1}')

        axs[i].set_title(f'Membrane Potentials for Stimulus: {direction.capitalize()}')
        axs[i].set_xlabel('Time Step')
        axs[i].set_ylabel('Membrane Potential')
        axs[i].legend()

    plt.tight_layout()
    plt.show()
