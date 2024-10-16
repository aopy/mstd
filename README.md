**MSTD*** is a computational neuroscience project aimed at modeling motion and depth processing in the primate visual cortex.

It utilizes spiking neural networks (SNNs) based on the leaky integrate-and-fire and adaptive exponential integrate-and-fire neuron models.

**Stimuli**: The project includes artificial stimuli (moving bars in directions up, down, left, right) found in the "ds_models" directory and event camera recordings ([TUM-VIE](https://cvg.cit.tum.de/data/datasets/visual-inertial-event-dataset) and [MVSEC](https://daniilidis-group.github.io/mvsec/) datasets) in the "of_models" and "v_models" directories.

**Learning**: The project employs _Spike-Timing-Dependent Plasticity (STDP)_ and _backpropagation_ to achieve selectivity for motion directions and optic flow patterns.

**Software**: The models are implemented using deep learning libraries such as [PyTorch](https://github.com/pytorch/pytorch) and [Norse](https://github.com/norse/norse), which provide tools for constructing and simulating spiking neural networks.

**Hardware**: The models are capable of running on both CPU and GPU, with CUDA support if available, to enhance computational efficiency and performance.

MSTD stands for dorsal medial superior temporal.

