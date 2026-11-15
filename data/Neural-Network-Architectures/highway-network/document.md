# Highway network

## Model
The model has two gates in addition to the  ${\displaystyle H(W_{H},x)}$  gate: the transform gate  ${\displaystyle T(W_{T},x)}$  and the carry gate  ${\displaystyle C(W_{C},x)}$ . The latter two gates are non-linear transfer functions (specifically sigmoid by convention). The function  ${\displaystyle H}$  can be any desired transfer function.

The carry gate is defined as:

 ${\displaystyle C(W_{C},x)=1-T(W_{T},x)}$ 

while the transform gate is just a gate with a sigmoid transfer function.


## Structure
The structure of a hidden layer in the Highway Network follows the equation:

 ${\displaystyle {\begin{aligned}y=H(x,W_{H})\cdot T(x,W_{T})+x\cdot C(x,W_{C})\\=H(x,W_{H})\cdot T(x,W_{T})+x\cdot (1-T(x,W_{T}))\end{aligned}}}$ 


## Related work
Sepp Hochreiter analyzed the vanishing gradient problem in 1991 and attributed to it the reason why deep learning did not work well.
To overcome this problem, Long Short-Term Memory (LSTM) recurrent neural networks
have residual connections with a weight of 1.0 in every LSTM cell (called the constant error carrousel) to compute  ${\textstyle y_{t+1}=F(x_{t})+x_{t}}$ . During backpropagation through time, this becomes the  residual formula  ${\textstyle y=F(x)+x}$  for feedforward neural networks. This enables training very deep recurrent neural networks with a very long time span t. A later LSTM version published in 2000 modulates the identity LSTM connections by so-called "forget gates" such that their weights are not fixed to 1.0 but can be learned. In experiments, the forget gates were initialized with positive bias weights, thus being opened, addressing the vanishing gradient problem.
As long as the forget gates of the 2000 LSTM are open, it behaves like the 1997 LSTM.

The Highway Network of May 2015
applies these principles to feedforward neural networks.
It was reported to be "the first very deep feedforward network with hundreds of layers". 
It is like a 2000 LSTM with forget gates unfolded in time, while the later Residual Nets have no equivalent of forget gates and are like the unfolded original 1997 LSTM.
If the skip connections in Highway Networks are "without gates," or if their gates are kept open (activation 1.0), they become Residual Networks.

The residual connection is a special case of the "short-cut connection" or "skip connection" by Rosenblatt (1961) and Lang & Witbrock (1988) which has the form  ${\displaystyle x\mapsto F(x)+Ax}$ . Here the randomly initialized weight matrix A does not have to be the identity mapping. Every residual connection is a skip connection, but almost all skip connections are not residual connections.

The original Highway Network paper not only introduced the basic principle for very deep feedforward networks, but also included experimental results with 20, 50, and 100 layers networks, and mentioned ongoing experiments with up to 900 layers. Networks with 50 or 100 layers had lower training error than their plain network counterparts, but no lower training error than their 20 layers counterpart (on the MNIST dataset, Figure 1 in ). No improvement on test accuracy was reported with networks deeper than 19 layers (on the CIFAR-10 dataset; Table 1 in ). The ResNet paper, however, provided strong experimental evidence of the benefits of going deeper than 20 layers. It argued that the identity mapping without modulation is crucial and mentioned that modulation in the skip connection can still lead to vanishing signals in forward and backward propagation (Section 3 in ). This is also  why the forget gates of the 2000 LSTM were initially opened through positive bias weights: as long as the gates are open, it behaves like the 1997 LSTM. Similarly, a Highway Net whose gates are opened through strongly positive bias weights behaves like a ResNet. The skip connections used in modern neural networks (e.g., Transformers) are dominantly identity mappings.