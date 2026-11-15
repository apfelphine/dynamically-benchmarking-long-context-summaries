# Deep belief network

## Training
The training method for RBMs proposed by Geoffrey Hinton for use with training "Product of Experts" models is called contrastive divergence (CD). CD provides an approximation to the maximum likelihood method that would ideally be applied for learning the weights. In training a single RBM, weight updates are performed with gradient descent via the following equation:  ${\displaystyle w_{ij}(t+1)=w_{ij}(t)+\eta {\frac {\partial \log(p(v))}{\partial w_{ij}}}}$ 

where,  ${\displaystyle p(v)}$  is the probability of a visible vector, which is given by  ${\displaystyle p(v)={\frac {1}{Z}}\sum _{h}e^{-E(v,h)}}$ .  ${\displaystyle Z}$  is the partition function (used for normalizing) and  ${\displaystyle E(v,h)}$  is the energy function assigned to the state of the network. A lower energy indicates the network is in a more "desirable" configuration. The gradient  ${\displaystyle {\frac {\partial \log(p(v))}{\partial w_{ij}}}}$  has the simple form  ${\displaystyle \langle v_{i}h_{j}\rangle _{\text{data}}-\langle v_{i}h_{j}\rangle _{\text{model}}}$  where  ${\displaystyle \langle \cdots \rangle _{p}}$  represent averages with respect to distribution  ${\displaystyle p}$ . The issue arises in sampling  ${\displaystyle \langle v_{i}h_{j}\rangle _{\text{model}}}$  because this requires extended alternating Gibbs sampling. CD replaces this step by running alternating Gibbs sampling for  ${\displaystyle n}$  steps (values of  ${\displaystyle n=1}$  perform well). After  ${\displaystyle n}$  steps, the data are sampled and that sample is used in place of  ${\displaystyle \langle v_{i}h_{j}\rangle _{\text{model}}}$ . The CD procedure works as follows:

1. Initialize the visible units to a training vector.
2. Update the hidden units in parallel given the visible units:  ${\displaystyle p(h_{j}=1\mid {\textbf {V}})=\sigma (b_{j}+\sum _{i}v_{i}w_{ij})}$ .  ${\displaystyle \sigma }$  is the sigmoid function and  ${\displaystyle b_{j}}$  is the bias of  ${\displaystyle h_{j}}$ .
3. Update the visible units in parallel given the hidden units:  ${\displaystyle p(v_{i}=1\mid {\textbf {H}})=\sigma (a_{i}+\sum _{j}h_{j}w_{ij})}$ .  ${\displaystyle a_{i}}$  is the bias of  ${\displaystyle v_{i}}$ . This is called the "reconstruction" step.
4. Re-update the hidden units in parallel given the reconstructed visible units using the same equation as in step 2.
5. Perform the weight update:  ${\displaystyle \Delta w_{ij}\propto \langle v_{i}h_{j}\rangle _{\text{data}}-\langle v_{i}h_{j}\rangle _{\text{reconstruction}}}$ .


Once an RBM is trained, another RBM is "stacked" atop it, taking its input from the final trained layer. The new visible layer is initialized to a training vector, and values for the units in the already-trained layers are assigned using the current weights and biases. The new RBM is then trained with the procedure above. This whole process is repeated until the desired stopping criterion is met.

Although the approximation of CD to maximum likelihood is crude (does not follow the gradient of any function), it is empirically effective.