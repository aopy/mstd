import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, layer, learning
from matplotlib import pyplot as plt


def f_weight(x):
    # weight update functions for the pre/post-synaptic traces
    # ensure that weight updates in the STDP learning process are bounded within a specific range,
    # preventing them from becoming too large or too small.
    return torch.clamp(x, 0, 1.)  # x, -1, 1


def create_moving_bars_stimulus(width, height, bar_width):
    stimulus = torch.zeros(height, width)
    stimulus[:, :bar_width] = 1  # bar that starts from the left and moves towards the right
    # stimulus[:, -bar_width:] = 1 # bar that starts from the right and moves towards the left
    return stimulus


torch.manual_seed(0)

if __name__ == '__main__':
    w_min, w_max = -1., 1.
    tau_pre, tau_post = 2., 2.
    # N_in = 8
    N_in = 28 * 28  # coresponding to image hight and width (pixels)
    N_out = 4   # corresponding to neurons for each 4 directions of the moving bar (left, right, up, down)
    # T = 128  # time steps
    T = 28
    batch_size = 2  # not used
    lr = 0.01  # learning rate (hyperparameter that controls the size of the step taken during optimization)

    net = nn.Sequential(
        torch.nn.Flatten(),  # Flatten the input
        layer.Linear(N_in, N_out, bias=False),  # layers LGN and V1
        neuron.LIFNode(tau=2.)  # Leaky integrate and Fire neuron model
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
    # duration = T
    # speed = 1.0

    # Create the moving bars stimulus
    # moving_bars_stimulus = create_moving_bars_stimulus(input_size, output_size, bar_width)
    moving_bars_stimulus = create_moving_bars_stimulus(28, 28, bar_width)

    # Get the height and width of the stimulus
    # height, width = moving_bars_stimulus.shape

    # Modify the input size in the linear layer
    # net[1] = layer.Linear(height * width, N_out, bias=False)

    # spike-timing-dependent plasticity (STDP) learning
    learner = learning.STDPLearner(step_mode='s', synapse=net[1], sn=net[2],  # Updated indices
                                   tau_pre=tau_pre, tau_post=tau_post,
                                   f_pre=f_weight, f_post=f_weight)
    # 1 time steps corresponds to 10 ms
    # step_mode='s' -> updates are applied at every time step
    # synapse (connection weights) to be updated (linear layer)
    # sn=net[2]: specifies the spiking neuron model to be used (LIF)
    # tau_pre=tau_pre and tau_post=tau_post: time constants for the pre/post-synaptic traces
    # they determine how fast the traces decay over time
    # f_pre=f_weight and f_post=f_weight: weight update functions for the pre/post-synaptic traces
    # in this case, f_weight is a function that clamps the input between -1 and 1

    out_spike = []
    trace_pre = []
    trace_post = []
    weight = []

    for t in range(T):  # iterates over a specified number of time steps
        optimizer.zero_grad()  # clear the gradients of all optimized tensors

        # Use the moving bars stimulus here
        in_spike = moving_bars_stimulus.view(1, -1)  # reshape to (1, num_inputs) - still necessary?
        # print("in_spike ", in_spike)
        # Feeds stimulus (in_spike) through the snn to obtain the output spikes at each neuron
        out_spike.append(net(in_spike))
        # out_spike.append(net(moving_bars_stimulus))
        # print("out_spike ", out_spike)
        learner.step(on_grad=True)  # advance the STDP learning process for one time step
        optimizer.step()  # update the model parameters (e.g weights) based on the computed gradients
        net[1].weight.data.clamp_(w_min, w_max)  # Update index to 1 to access the linear layer
        weight.append(net[1].weight.data.clone())
        # to keep track of how the weights change over time
        trace_pre.append(learner.trace_pre)  # appends the current pre-synaptic trace to the trace_pre list.
        trace_post.append(learner.trace_post)  # appends the current post-synaptic trace to the trace_post list
        # used for monitoring how the pre/post-synaptic trace evolves during training

    out_spike = torch.stack(out_spike)  # [T, batch_size, N_out]
    trace_pre = torch.stack(trace_pre)  # [T, batch_size, N_in]
    trace_post = torch.stack(trace_post)  # [T, batch_size, N_out]
    weight = torch.stack(weight)  # [T, N_out, N_in]

    t = torch.arange(0, len(out_spike)).float()  # Use len(out_spike) to get the correct size
    # creating a time tensor t with values representing each time step in the simulation
    # print("t ", t)

    # in_spike = moving_bars_stimulus[:, 0, 0]
    out_spike = out_spike[:, 0, :]
    # [T, batch_size, N_out] where T: number of time steps, batch_size: number of samples in a batch,
    # N_out: number of output neurons. [:, 0, :] extracts the spikes of all 8 neurons
    # for each time step for the first batch sample
    trace_pre = trace_pre[:, 0, :]
    trace_post = trace_post[:, 0, :]
    weight = weight[:, 0, :]
    # extract the weights connecting all output neurons to all input neurons for each time step

    # Plotting
    # Store the input and output spikes for plotting
    input_spikes = moving_bars_stimulus.view(-1).numpy()
    output_spikes = out_spike.detach().numpy()  # Use detach() to remove the gradient information

    # Plotting
    plt.figure(figsize=(10, 6))

    # Plot input spikes
    input_spike_times = torch.nonzero(torch.from_numpy(input_spikes)).numpy()
    plt.subplot(2, 1, 1)
    plt.eventplot(input_spike_times, colors='b', lineoffsets=0)
    plt.title('Input Spikes')
    plt.xlabel('Time Step')
    plt.ylabel('Neuron Index')

    # Plot output spikes
    plt.subplot(2, 1, 2)
    for neuron_idx in range(N_out):
        spike_times = torch.nonzero(torch.from_numpy(output_spikes[:, neuron_idx])).numpy()
        plt.eventplot(spike_times, colors='r', lineoffsets=neuron_idx, label=f'Neuron {neuron_idx + 1}')

    plt.title('Output Spikes')
    plt.xlabel('Time Step')
    plt.ylabel('Neuron Index')
    plt.legend()

    plt.tight_layout()
    plt.show()
