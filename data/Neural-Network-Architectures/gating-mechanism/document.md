# Gating mechanism

## RNNs
Gating mechanisms are the centerpiece of long short-term memory (LSTM). They were proposed to mitigate the vanishing gradient problem often encountered by regular RNNs.

An LSTM unit contains three gates:

- An input gate, which controls the flow of new information into the memory cell
- A forget gate, which controls how much information is retained from the previous time step
- An output gate, which controls how much information is passed to the next layer.


The equations for LSTM are:

 ${\displaystyle {\begin{aligned}\mathbf {I} _{t}&=\sigma (\mathbf {X} _{t}\mathbf {W} _{xi}+\mathbf {H} _{t-1}\mathbf {W} _{hi}+\mathbf {b} _{i})\\\mathbf {F} _{t}&=\sigma (\mathbf {X} _{t}\mathbf {W} _{xf}+\mathbf {H} _{t-1}\mathbf {W} _{hf}+\mathbf {b} _{f})\\\mathbf {O} _{t}&=\sigma (\mathbf {X} _{t}\mathbf {W} _{xo}+\mathbf {H} _{t-1}\mathbf {W} _{ho}+\mathbf {b} _{o})\\{\tilde {\mathbf {C} }}_{t}&=\tanh(\mathbf {X} _{t}\mathbf {W} _{xc}+\mathbf {H} _{t-1}\mathbf {W} _{hc}+\mathbf {b} _{c})\\\mathbf {C} _{t}&=\mathbf {F} _{t}\odot \mathbf {C} _{t-1}+\mathbf {I} _{t}\odot {\tilde {\mathbf {C} }}_{t}\\\mathbf {H} _{t}&=\mathbf {O} _{t}\odot \tanh(\mathbf {C} _{t})\end{aligned}}}$ 



Here,  ${\displaystyle \odot }$  represents elementwise multiplication. 

- LSTM architecture, with gates


The gated recurrent unit (GRU) simplifies the LSTM. Compared to the LSTM, the GRU has just two gates: a reset gate and an update gate. GRU also merges the cell state and hidden state. The reset gate roughly corresponds to the forget gate, and the update gate roughly corresponds to the input gate. The output gate is removed.

There are several variants of GRU. One particular variant has these equations:

 ${\displaystyle {\begin{aligned}\mathbf {R} _{t}&=\sigma (\mathbf {X} _{t}\mathbf {W} _{xr}+\mathbf {H} _{t-1}\mathbf {W} _{hr}+\mathbf {b} _{r})\\\mathbf {Z} _{t}&=\sigma (\mathbf {X} _{t}\mathbf {W} _{xz}+\mathbf {H} _{t-1}\mathbf {W} _{hz}+\mathbf {b} _{z})\\{\tilde {\mathbf {H} }}_{t}&=\tanh(\mathbf {X} _{t}\mathbf {W} _{xh}+(\mathbf {R} _{t}\odot \mathbf {H} _{t-1})\mathbf {W} _{hh}+\mathbf {b} _{h})\\\mathbf {H} _{t}&=\mathbf {Z} _{t}\odot \mathbf {H} _{t-1}+(1-\mathbf {Z} _{t})\odot {\tilde {\mathbf {H} }}_{t}\end{aligned}}}$ 

- Gated Recurrent Unit architecture, with gates



## Gated Linear Unit
Gated Linear Units (GLUs) adapt the gating mechanism for use in feedforward neural networks, often within transformer-based architectures. They are defined as:

 ${\displaystyle \mathrm {GLU} (a,b)=a\odot \sigma (b)}$ 

where  ${\displaystyle a,b}$  are the first and second inputs, respectively.  ${\displaystyle \sigma }$  represents the sigmoid activation function.

Replacing  ${\displaystyle \sigma }$  with other activation functions leads to variants of GLU:

 ${\displaystyle {\begin{aligned}\mathrm {ReGLU} (a,b)&=a\odot {\text{ReLU}}(b)\\\mathrm {GEGLU} (a,b)&=a\odot {\text{GELU}}(b)\\\mathrm {SwiGLU} (a,b,\beta )&=a\odot {\text{Swish}}_{\beta }(b)\end{aligned}}}$ 

where ReLU, GELU, and Swish are different activation functions.

In transformer models, such gating units are often used in the feedforward modules. For a single vector input, this results in:

 ${\displaystyle {\begin{aligned}\operatorname {GLU} (x,W,V,b,c)&=\sigma (xW+b)\odot (xV+c)\\\operatorname {Bilinear} (x,W,V,b,c)&=(xW+b)\odot (xV+c)\\\operatorname {ReGLU} (x,W,V,b,c)&=\max(0,xW+b)\odot (xV+c)\\\operatorname {GEGLU} (x,W,V,b,c)&=\operatorname {GELU} (xW+b)\odot (xV+c)\\\operatorname {SwiGLU} (x,W,V,b,c,\beta )&=\operatorname {Swish} _{\beta }(xW+b)\odot (xV+c)\end{aligned}}}$ 


## Other architectures
Gating mechanism is used in highway networks, which were designed by unrolling an LSTM.

Channel gating uses a gate to control the flow of information through different channels inside a convolutional neural network (CNN).