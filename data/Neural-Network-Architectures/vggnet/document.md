# VGGNet

## Architecture
The key architectural principle of VGG models is the consistent use of small  ${\displaystyle 3\times 3}$  convolutional filters throughout the network. This contrasts with earlier CNN architectures that employed larger filters, such as  ${\displaystyle 11\times 11}$  in AlexNet.

For example, two  ${\textstyle 3\times 3}$  convolutions stacked together has the same receptive field pixels as a single  ${\textstyle 5\times 5}$  convolution, but the latter uses  ${\textstyle \left(25\cdot c^{2}\right)}$  parameters, while the former uses  ${\textstyle \left(18\cdot c^{2}\right)}$  parameters (where  ${\displaystyle c}$  is the number of channels). The original publication showed that deep and narrow CNN significantly outperform their shallow and wide counterparts.

The VGG series of models are deep neural networks composed of generic modules:

1. Convolutional modules:  ${\displaystyle 3\times 3}$  convolutional layers with stride 1, followed by ReLU activations.
2. Max-pooling layers: After some convolutional modules, max-pooling layers with a  ${\displaystyle 2\times 2}$  filter and a stride of 2 to downsample the feature maps. It halves both width and height, but keeps the number of channels.
3. Fully connected layers: Three fully connected layers at the end of the network, with sizes 4096-4096-1000. The last one has 1000 channels corresponding to the 1000 classes in ImageNet.
4. Softmax layer: A softmax layer outputs the probability distribution over the classes.


The VGG family includes various configurations with different depths, denoted by the letter "VGG" followed by the number of weight layers. The most common ones are VGG-16 (13 convolutional layers + 3 fully connected layers) and VGG-19 (16 + 3), denoted as configurations D and E in the original paper.

As an example, the 16 convolutional layers of VGG-19 are structured as follows: ${\displaystyle {\begin{aligned}&3\to 64\to 64&\xrightarrow {\text{downsample}} \\&64\to 128\to 128&\xrightarrow {\text{downsample}} \\&128\to 256\to 256\to 256\to 256&\xrightarrow {\text{downsample}} \\&256\to 512\to 512\to 512\to 512&\xrightarrow {\text{downsample}} \\&512\to 512\to 512\to 512\to 512&\xrightarrow {\text{downsample}} \end{aligned}}}$ where the arrow  ${\displaystyle c_{1}\to c_{2}}$  means a 3x3 convolution with  ${\displaystyle c_{1}}$  input channels and  ${\displaystyle c_{2}}$  output channels and stride 1, followed by ReLU activation. The  ${\displaystyle \xrightarrow {\text{downsample}} }$  means a down-sampling layer by 2x2 maxpooling with stride 2.

| Name   | Number of convolutional layers | Number of fully connected layers | Parameter count |
| ------ | ------------------------------ | -------------------------------- | --------------- |
| VGG-16 | 13                             | 3                                | 138M            |
| VGG-19 | 16                             | 3                                | 144M            |



## Training
The original VGG models were implemented in a version of C++ Caffe, modified for multi-GPU training and evaluation with data parallelism. On a system equipped with 4 NVIDIA Titan Black GPUs, training a single net took 2–3 weeks depending on the architecture.