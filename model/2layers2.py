import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, layer, learning
from matplotlib import pyplot as plt


def f_weight(x):
    # weight update functions for the pre/post-synaptic traces
    # ensure that weight updates in the STDP learning process are bounded within a specific range,
    # preventing them from becoming too large or too small.
    # The non-negative values of the weights reflect the fact that thalamic connections to V1 are excitatory in nature
    return torch.clamp(x, 0, 1.)  # x, -1, 1


def create_moving_bars_stimulus(batch_size, width, height, bar_width, time_step, total_time_steps):
    # moving bars stimulus where a vertical line of 1s moves at each time step
    stimulus = torch.zeros(batch_size, height, width)  # 1 for batch
    # Calculate the position of the bar based on the time step
    if time_step < total_time_steps/2:
        current_position = (time_step % (width - bar_width + 1))  # bar moving from left to right
    else:
        current_position = width - bar_width - (time_step % (width - bar_width + 1))  # from right to left
    # Set the bar at the calculated position
    stimulus[:, :, current_position: current_position + bar_width] = 1
    # print("current_position ", current_position)
    # print("stimulus ", stimulus)
    return stimulus


torch.manual_seed(0)

if __name__ == '__main__':
    # w_min, w_max = -1., 1.
    w_min, w_max = 0, 1.
    # tau_pre, tau_post = 2., 2.
    tau_pre, tau_post = 6., 6.
    # N_in = 8
    # N_in = 28 * 28  # corresponding to image hight and width (pixels)
    N_in = 10 * 10  # corresponding to image hight and width (pixels)
    N_out = 4   # corresponding to neurons for each 4 directions of the moving bar (left, right, up, down)
    # T = 128  # time steps
    T = 10
    batch_size = 2  # not used
    lr = 0.01  # learning rate (hyperparameter that controls the size of the step taken during optimization)

    net = nn.Sequential(
        torch.nn.Flatten(),  # Flatten the input
        layer.Linear(N_in, N_out, bias=False),  # layers LGN and V1
        neuron.LIFNode(tau=8.)  # Leaky integrate and Fire neuron model (tau=2.)
    )

    # initializes the weights of the linear layer
    nn.init.constant_(net[1].weight.data, 0.4)
    #  initializes the stochastic gradient descent (SGD) optimizer for training the snn
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.)
    # momentum (factor): a technique that helps accelerate SGD in the relevant direction and dampens oscillations.
    # momentum value of 0 means that there is no momentum applied in this case

    input_size = N_in
    output_size = N_out
    bar_width = 1  # 2

    # spike-timing-dependent plasticity (STDP) learning
    learner = learning.STDPLearner(step_mode='s', synapse=net[1], sn=net[2],
                                   tau_pre=tau_pre, tau_post=tau_post,
                                   f_pre=f_weight, f_post=f_weight)
    # 1 time steps corresponds to 10 ms
    # step_mode='s' -> updates are applied at every time step
    # synapse (connection weights) to be updated (linear layer)
    # sn=net[2]: specifies the spiking neuron model to be used (LIF)
    # tau_pre=tau_pre and tau_post=tau_post: time constants for the pre/post-synaptic traces
    # they determine how fast the traces decay over time
    # f_pre=f_weight and f_post=f_weight: weight update functions for the pre/post-synaptic traces

    out_spike = []
    trace_pre = []
    trace_post = []
    weight = []

    for t in range(T*2):  # iterates over a specified number of time steps
        # for t in range(T):
        # print("t ", t)
        optimizer.zero_grad()  # clear the gradients of all optimized tensors

        # Create the moving bars stimulus
        # moving_bars_stimulus = create_moving_bars_stimulus(1, 28, 28, bar_width, t)
        moving_bars_stimulus = create_moving_bars_stimulus(1, 10, 10, bar_width, t, 2*T)
        # print("moving_bars_stimulus ", moving_bars_stimulus)

        # Feeds stimulus (moving_bars_stimulus) through the snn to obtain the output spikes at each neuron
        out_spike.append(net(moving_bars_stimulus))
        # print("out_spike ", out_spike)
        learner.step(on_grad=True)  # advance the STDP learning process for one time step
        optimizer.step()  # update the model parameters (e.g weights) based on the computed gradients
        net[1].weight.data.clamp_(w_min, w_max)  # Update index to 1 to access the linear layer
        weight.append(net[1].weight.data.clone())
        # to keep track of how the weights change over time
        trace_pre.append(learner.trace_pre)  # appends the current pre-synaptic trace to the trace_pre list.
        trace_post.append(learner.trace_post)  # appends the current post-synaptic trace to the trace_post list
        # used for monitoring how the pre/post-synaptic trace evolves during training

    # creating subplots for plotting
    fig, axes = plt.subplots(4, 1, figsize=(8, 10))

    # plotting the input stimulus
    axes[0].imshow(moving_bars_stimulus.squeeze(), cmap='gray', aspect='auto')
    axes[0].set_title('Input Stimulus')
    axes[0].set_xlabel('Time Steps')
    axes[0].set_ylabel('Neurons')

    # plotting the output spikes
    axes[1].imshow(torch.stack(out_spike).squeeze().detach().numpy(), cmap='gray', aspect='auto', vmin=0, vmax=1)
    axes[1].set_title('Output Spikes')
    axes[1].set_xlabel('Time Steps')
    axes[1].set_ylabel('Neurons')

    # plotting the pre-synaptic trace
    axes[2].plot(torch.stack(trace_pre).squeeze().detach().numpy().T)
    axes[2].set_title('Pre-synaptic Trace')
    axes[2].set_xlabel('Time Steps')
    axes[2].set_ylabel('Neurons')

    # plotting the post-synaptic trace
    axes[3].plot(torch.stack(trace_post).squeeze().detach().numpy().T)
    axes[3].set_title('Post-synaptic Trace')
    axes[3].set_xlabel('Time Steps')
    axes[3].set_ylabel('Neurons')

    plt.tight_layout()
    plt.show()

    # collecting the output spikes
    out_spike = torch.stack(out_spike).squeeze().detach().numpy()

    # print output spikes for each neuron in each time step
    for neuron_idx in range(out_spike.shape[1]):
        print(f'Neuron {neuron_idx + 1} Output Spikes:')
        print(out_spike[:, neuron_idx])
        print('\n')
