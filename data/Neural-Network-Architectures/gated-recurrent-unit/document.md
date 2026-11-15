# Gated recurrent unit

## Architecture
There are several variations on the full gated unit, with gating done using the previous hidden state and the bias in various combinations, and a simplified form called minimal gated unit.

In the following, the operator  ${\displaystyle \odot }$  denotes the Hadamard product.


### Fully gated unit
Initially, for  ${\displaystyle t=0}$ , the output vector is  ${\displaystyle h_{0}=0}$ . 

 ${\displaystyle {\begin{aligned}z_{t}&=\sigma (W_{z}x_{t}+U_{z}h_{t-1}+b_{z})\\r_{t}&=\sigma (W_{r}x_{t}+U_{r}h_{t-1}+b_{r})\\{\hat {h}}_{t}&=\phi (W_{h}x_{t}+U_{h}(r_{t}\odot h_{t-1})+b_{h})\\h_{t}&=(1-z_{t})\odot h_{t-1}+z_{t}\odot {\hat {h}}_{t}\end{aligned}}}$ 

Variables ( ${\displaystyle d}$  denotes the number of input features and  ${\displaystyle e}$  the number of output features):

- ${\displaystyle x_{t}\in \mathbb {R} ^{d}}$ : input vector
- ${\displaystyle h_{t}\in \mathbb {R} ^{e}}$ : output vector
- ${\displaystyle {\hat {h}}_{t}\in \mathbb {R} ^{e}}$ : candidate activation vector
- ${\displaystyle z_{t}\in (0,1)^{e}}$ : update gate vector
- ${\displaystyle r_{t}\in (0,1)^{e}}$ : reset gate vector
- ${\displaystyle W\in \mathbb {R} ^{e\times d}}$ ,  ${\displaystyle U\in \mathbb {R} ^{e\times e}}$  and  ${\displaystyle b\in \mathbb {R} ^{e}}$ : parameter matrices and vector which need to be learned during training


Activation functions

- ${\displaystyle \sigma }$ : The original is a logistic function.
- ${\displaystyle \phi }$ : The original is a hyperbolic tangent.


Alternative activation functions are possible, provided that  ${\displaystyle \sigma (x)\in [0,1]}$ .

Alternate forms can be created by changing  ${\displaystyle z_{t}}$  and  ${\displaystyle r_{t}}$ 

- Type 1: each gate depends only on the previous hidden state and the bias.
 ${\displaystyle {\begin{aligned}z_{t}&=\sigma (U_{z}h_{t-1}+b_{z})\\r_{t}&=\sigma (U_{r}h_{t-1}+b_{r})\\\end{aligned}}}$
- Type 2: each gate depends only on  the previous hidden state.
 ${\displaystyle {\begin{aligned}z_{t}&=\sigma (U_{z}h_{t-1})\\r_{t}&=\sigma (U_{r}h_{t-1})\\\end{aligned}}}$
- Type 3: each gate is computed using only the bias.
 ${\displaystyle {\begin{aligned}z_{t}&=\sigma (b_{z})\\r_{t}&=\sigma (b_{r})\\\end{aligned}}}$



### Minimal gated unit
The minimal gated unit (MGU) is similar to the fully gated unit, except the update and reset gate vector is merged into a forget gate. This also implies that the equation for the output vector must be changed:

 ${\displaystyle {\begin{aligned}f_{t}&=\sigma (W_{f}x_{t}+U_{f}h_{t-1}+b_{f})\\{\hat {h}}_{t}&=\phi (W_{h}x_{t}+U_{h}(f_{t}\odot h_{t-1})+b_{h})\\h_{t}&=(1-f_{t})\odot h_{t-1}+f_{t}\odot {\hat {h}}_{t}\end{aligned}}}$ 

Variables

- ${\displaystyle x_{t}}$ : input vector
- ${\displaystyle h_{t}}$ : output vector
- ${\displaystyle {\hat {h}}_{t}}$ : candidate activation vector
- ${\displaystyle f_{t}}$ : forget vector
- ${\displaystyle W}$ ,  ${\displaystyle U}$  and  ${\displaystyle b}$ : parameter matrices and vector



### Light gated recurrent unit
The light gated recurrent unit (LiGRU) removes the reset gate altogether, replaces tanh with the ReLU activation, and applies batch normalization (BN):

 ${\displaystyle {\begin{aligned}z_{t}&=\sigma (\operatorname {BN} (W_{z}x_{t})+U_{z}h_{t-1})\\{\tilde {h}}_{t}&=\operatorname {ReLU} (\operatorname {BN} (W_{h}x_{t})+U_{h}h_{t-1})\\h_{t}&=z_{t}\odot h_{t-1}+(1-z_{t})\odot {\tilde {h}}_{t}\end{aligned}}}$ 

LiGRU has been studied from a Bayesian perspective. This analysis yielded a variant called light Bayesian recurrent unit (LiBRU), which showed slight improvements over the LiGRU on speech recognition tasks.