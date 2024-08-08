"""
Spiking Neural Network (SNN) Model using Backpropagation with Moving Bars Stimulus

This script implements a Spiking Neural Network (SNN) model trained using backpropagation to classify
the direction of moving bars stimulus. The SNN comprises Leaky Integrate-and-Fire (LIF) neurons with an
infinite threshold. The training is performed using the AdamW optimizer and the cross-entropy loss function.

Key Features:
1. **Two Directions/Two Neurons**: The network is trained to recognize two directions of a moving bar:
   left-to-right and right-to-left. There are two neurons, each expected to become selective (i.e. have high membrane
   potential) to one direction after training.
2. **Single Linear Layer**: The model uses a single linear layer to process the input stimuli.
3. **Stimulus Generation**: The `create_moving_bars_stimulus_with_delay_and_labels` function generates
   moving bars stimulus as tensors and corresponding labels, with each frame having a delayed version to imitate synaptic delay,
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
from plasticity import STDPLearner

random.seed(1)
# torch.manual_seed(0)

direction_choice = 'left_to_right'


def create_moving_bars_stimulus_with_delay_and_labels(batch_size, width, height, bar_width, time_step, synaptic_delay=1):
    p_left = 0.5  # probability of moving to the left
    # moving bars stimulus with synaptic delay and labels
    current_stimulus = torch.zeros(batch_size, height, width)
    delayed_stimulus = torch.zeros(batch_size, height, width)

    # time_step = time_step % width

    global direction_choice

    # Check if a direction change is needed
    # if time_step % width == 0:
    # if (time_step % width == 0) or (time_step == T - 1):
    if time_step == 0:
        # Randomly choose a new direction
        # direction_choice = random.choice(['left_to_right', 'right_to_left'])
        # Randomly choose the direction based on probability p_left
        direction_choice = 'left_to_right' if random.random() < p_left else 'right_to_left'
        print("!!!! direction_choice ", direction_choice)

    if 1 <= time_step <= 10:
        # Calculate the position of the bar based on the time step and direction choice
        if direction_choice == 'left_to_right':
            # Bar moving from left to right
            # current_position = min(time_step, width - bar_width)
            current_position = min(time_step-1, width - bar_width)
        elif direction_choice == 'right_to_left':
            # Bar moving from right to left, and waiting at the end
            # current_position = max(0, (width - time_step - 1))
            current_position = max(0, (width - time_step + 1 - 1))
        else:
            raise ValueError("Invalid direction choice. Use 'left_to_right' or 'right_to_left'.")
        # import pdb;pdb.set_trace()
        current_stimulus[:, :, current_position: current_position + bar_width] = 1

    if 2 <= time_step <= 11:
        if direction_choice == 'left_to_right':
            delayed_position = min(time_step-2, width - bar_width)
        elif direction_choice == 'right_to_left':
            delayed_position = max(0, (width - time_step + 2 - 1))
        delayed_stimulus[:, :, delayed_position: delayed_position + bar_width] = 1

    combinedinput = torch.stack([current_stimulus, delayed_stimulus], dim=1)

    # Determine label based on the relative position of the bars
    # label = torch.tensor(1.0 if direction_choice == 'left_to_right' else 0.0, dtype=torch.float32)
    label = torch.tensor([0, 1] if direction_choice == 'left_to_right' else [1, 0], dtype=torch.float32)

    return combinedinput, label


if __name__ == '__main__':
    # w_min, w_max = -1., 1.
    w_min, w_max = 0, 1.
    # tau_pre, tau_post = 6., 6.  # too much 20
    tau_pre, tau_post = 20., 20.
    # tau_pre, tau_post = 15.0, 15.0
    # tau_pre, tau_post = 2., 2.
    N_in = 10 * 10  # corresponding to image hight and width (pixels)
    N_out = 2   # corresponding to neurons for each direction of the moving bar (left, right, up, down)
    S = 1000
    batch_size = 1
    width = 10
    height = 10
    # lr = 0.1  # the good one
    lr = 0.01  # (adam)

    loss_values = []

    net = nn.Sequential(
        nn.Flatten(),  # Flatten the input
        nn.Linear(2 * N_in, N_out, bias=False),
        neuron.LIFNode(tau=10.0, v_threshold=float('inf'))  # infinite threshold
    )
    # initializes the weights of the linear layer
    # nn.init.constant_(net[1].weight.data, 0.4)
    # nn.init.normal_(net[1].weight.data, mean=0.4)
    # nn.init.normal_(net[1].weight.data, mean=0.4, std=0.01)
    nn.init.uniform_(net[1].weight.data, -0.1, 0.1)
    # torch.nn.init.uniform_(net[1].weight.data, -1., 1.)
    # nn.init.uniform_(net[1].weight.data, 0.1, 0.4)
    # nn.init.xavier_uniform_(net[1].weight)
    # nn.init.uniform_(net[1].weight, a=-0.1, b=0.1)
    # initializes the stochastic gradient descent (SGD) optimizer for training the snn
    # Define the loss function and optimizer
    # optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.)
    # optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr)

    # momentum (factor): a technique that helps accelerate SGD in the relevant direction and dampens oscillations.
    # momentum value of 0 means that there is nlearner.step(on_grad=True)o momentum applied in this case

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
    total_samples = 0
    # accumulated_labels = []
    accumulated_labels = torch.tensor([])
    # accumulated_potentials = torch.zeros(2, 10)
    # accumulated_potentials = torch.zeros(2, 9)
    accumulated_potentials = torch.zeros(2, 13)
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
        # import pdb;pdb.set_trace()
        # predicted_label = torch.argmax(accumulated_potentials, dim=0)
        active_neuron = torch.argmax(accumulated_potentials)
        # predicted_label = torch.argmax(accumulated_potentials, dim=1)
        print("active_neuron ", active_neuron)
        if active_neuron < 13:
            active_neuron = torch.tensor([1, 0])
        else:
            active_neuron = torch.tensor([0, 1])
        print("active_neuron ", active_neuron)
        # print("overall_label ", overall_label)
        # import pdb;pdb.set_trace()
        accuracy = torch.sum(active_neuron[0] == label[0]).item() / 10 * 100.0
        accuracy3 = torch.sum(active_neuron[0] == label[0]).item() * 100.0
        # accuracy = torch.sum(predicted_label == overall_label).item() / 10 * 100.0
        print("accuracy ", accuracy)
        print("accuracy ", accuracy3)
        correct = active_neuron == label
        # correct = predicted_label == overall_label
        print("correct ", correct)
        total_correct += correct.sum().item()
        print("total_correct ", total_correct)

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
        accumulated_potentials = torch.zeros(2, 13)

    # average_accuracy = (total_correct / (10 * S)) * 100.0
    average_accuracy = (total_correct / 2 / S) * 100.0

    accuracy2 = 100 * correct2 / S
    print("Accuracy2 = {}".format(accuracy2))
    print(f'Average Accuracy over {S} samples: {average_accuracy:.2f}%')

    # accuracy_training = total_correct / T
    # print(f'Training Overall Accuracy: {accuracy_training * 100:.2f}%')

    # Transpose the data to have time on the x-axis
    # Plot the final weights as a heatmap
    # final_weights = final_weights2[-1]

    # final_weights = weight_history[-1]

    final_weights_reshaped = final_weights.view(2, 2, 10, 10)

    # Create a 2x2 subplot grid
    fig, axs = plt.subplots(2, 2)

    # Plot the weights for each neuron
    for neuron_index in range(2):
        for frame_index in range(2):
            ax = axs[neuron_index, frame_index]
            im = ax.imshow(final_weights_reshaped[neuron_index, frame_index].detach().numpy(), cmap='viridis',
                           origin='upper')
            ax.set_title(f'Weights for Neuron {neuron_index + 1} Frame {frame_index + 1}')
            ax.axis('off')

    # Add a colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    plt.show()

    plt.plot(loss_values, label='Training Loss')
    plt.xlabel('Samples')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    print("TESTING---------->")
    net.eval()
    R = 0
    all_predictions = torch.tensor([])
    all_true_labels = torch.tensor([])
    correct = 0
    total_correct = 0
    total_samples = 0
    total_time_steps = 0
    for r in range(R):
        print("sample ", r)
        optimizer.zero_grad()
        for i in range(13):
            print("i ", i)
            # Create the moving bars stimulus
            combined_input, label = create_moving_bars_stimulus_with_delay_and_labels(batch_size=1, width=10, height=10,
                                                                                      bar_width=1, time_step=i)
            # print(combined_input)
            output = net(combined_input)
            mp = net[2].v
            print("mps ", mp)
            accumulated_potentials[:, i] = mp
            print("accumulated_potentials ", accumulated_potentials)
            print("direction_choice ", direction_choice)
            print("label ", label)

        max_values, max_indices = torch.max(accumulated_potentials, dim=1)
        indices_tensor = torch.arange(accumulated_potentials.shape[0])
        highest_values_tensor = accumulated_potentials[indices_tensor, max_indices]
        print("highest_values_tensor ", highest_values_tensor)
        print("label ", label)

        print("direction_choice ", direction_choice)

        print("loss ", loss)
        # predicted_label = torch.argmax(accumulated_potentials, dim=0)
        active_neuron = torch.argmax(accumulated_potentials)
        # predicted_label = torch.argmax(accumulated_potentials, dim=1)
        print("active_neuron ", active_neuron)
        if active_neuron < 13:
            active_neuron = torch.tensor([1, 0])
        else:
            active_neuron = torch.tensor([0, 1])
        print("active_neuron ", active_neuron)
        # print("overall_label ", overall_label)
        # import pdb;pdb.set_trace()
        accuracy = torch.sum(active_neuron[0] == label[0]).item() / 10 * 100.0
        accuracy3 = torch.sum(active_neuron[0] == label[0]).item() * 100.0
        # accuracy = torch.sum(predicted_label == overall_label).item() / 10 * 100.0
        print("accuracy ", accuracy)
        print("accuracy ", accuracy3)
        correct = active_neuron == label
        # correct = predicted_label == overall_label
        print("correct ", correct)
        total_correct += correct.sum().item()
        print("total_correct ", total_correct)

        correct2 += (active_neuron[0] == label[0]).sum()
        print("correct2 ", correct2)
        print("total_correct ", total_correct)

        # Store the weights for later analysis
        weight_history.append(net[1].weight.data.clone())

        weight.append(net[1].weight.data.clone().numpy().flatten())
        if r == R-1:
            print(r)
            # final_weights = net[1].weight.data.clone().numpy().flatten()
            final_weights = net[1].weight.data.clone()
            print("final_weights ", final_weights)
        functional.reset_net(net)
        # accumulated_potentials = torch.zeros(1, 2)
        # accumulated_potentials = torch.zeros(2, 9)
        accumulated_potentials = torch.zeros(2, 13)
    if R:
        average_accuracy = (total_correct / 2 / R) * 100.0

        accuracy2 = 100 * correct2 / R
        print("Accuracy2 = {}".format(accuracy2))
        print(f'Average Accuracy over {S} samples: {average_accuracy:.2f}%')

