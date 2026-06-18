Convolutional
Neural Networks

Lab 2

Marco Garosi

University of Trento

Agenda

Some theory

• Normalization strategies

• AlexNet

Lab session

• Dive into the lab!

Project assignment

• Code skeleton & evaluation

Some theory

Batch normalizzation

Before 2015, training neural network posed different challenges than

today. Specifically:

1. The initialization nightmare

2. Snail-paced learning

The initialization nightmare

Weights had to be initialized with painstaking precision, such as using

Xavier or He initialization.

Starting with the wrong weights could mean wasting an entire training

run.

https://en.wikipedia.org/wiki/Weight_initialization

Snail-paced learning

Because activations shifted layer by layer (i.e., their effects

compounded), learning rates had to be kept very small.

But a small learning rate means convergence takes a very long time to

be reached.

Batch norm

In 2015, Sergey Ioffe and Christian Szegedy introduced batch

normalization: https://arxiv.org/abs/1502.03167

They proposed a radical idea: don’t just normalize the inputs to the

network, normalize the inputs to every single layer

Batch norm – How it works

For a given mini-batch of size m,

batch norm calculates the mean and

variance for each feature channel,

normalizes the activations, and then

applies learnable scale (gamma) and

shift (beta).

Batch norm – Why it works

1. Faster convergence: you can increase the learning rate!

2. Forgiving initialization: the network is highly resilient to the initial

weights, so no need to carefully choose the “starting point” anymore

3.

Implicit regularization: reduces the need for dropout

4. Smooth optimization landscape: gradients are more predictive and

stable (https://arxiv.org/abs/1805.11604)

Batch norm – Shortcomings

However, Batch Norm…

1. Breaks the independence assumption: predictions for x1 depend on x2, x3, …,

xm in the batch

2. The micro-batch problem: with larger inputs, we had to reduce the batch size

to fit in VRAM. But a smaller batch size hurts BN, because its statistics depend

on the batch itself!

3. Train-test discrepancy: the way it normalizes inputs changes between train

and test time, leading to subtle bugs

4. Doesn’t like sequential data: RNNs and Transformers deal with variable-length

data, and applying BN to this kind of data is both complex and inefficient

Today’s normalization
strategies

Modern deep learning relies on:

1. Layer normalization (LayerNorm)

2. Group normalization (GroupNorm)

3. Root mean square normalization (RMSNorm)

LayerNorm

• Applied to: Transformers, LLMs

• How it works: instead of normalizing one feature across a whole batch, it

normalizes all features for a single sample

• Why it works: it’s independent of batch size. It works perfectly with varying

sequence lengths.

GroupNorm

• Applied to: Diffusion models

• How it works: it groups the channels of a single image together and normalizes

withing those groups

• Why it works: it bridges the gap between BatchNorm and LayerNorm.

Independent of batch size

RMSNorm

• Applied to: SOTA LLMs

• How it works: it’s a simplified LayerNorm, which throws away the mean-

centering part of LayerNorm (as researchers discovered that wasn’t doing much)

• Why it works: same stability & performance as LayerNorm, mathematically

cheaper, can provide speedup in LLM training and inference

Lab Session

Convolutional
Neural Networks

Lab 2

Marco Garosi

University of Trento

