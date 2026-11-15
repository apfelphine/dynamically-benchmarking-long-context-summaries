## Recurrent Neural Networks (RNNs)
RNNs have historical roots in neuroscience, tracing back to early observations of recurrent loops in the cerebellum and conceptual models from the 1940s. Statistical mechanics also provided an origin, notably through the Ising model and its evolution into the Hopfield network. Modern RNNs are primarily based on LSTM and BRNN architectures.

RNN configurations include standard variants, stacked RNNs (deep RNNs), bidirectional RNNs (biRNNs), and encoder-decoder structures for sequence transduction. Abstractly, an RNN maps an input $x_t$ to an output $y_t$ while updating its internal "memory" hidden state $h_{t+1}$.

### Key Architectures and Training
Architectural variants include the Hopfield network, Elman networks, Jordan networks, and specialized forms like IndRNN. LSTM networks were developed to mitigate the vanishing gradient problem, allowing them to learn long-term dependencies. Bidirectional RNNs process input in both forward and reverse directions.

Training techniques involve teacher forcing to stabilize autoregression. Gradient descent is typically implemented via Backpropagation Through Time (BPTT), though computationally intensive methods like Real-Time Recurrent Learning (RTRL) also exist. Other training strategies include Connectionist Temporal Classification (CTC) and global optimization methods like genetic algorithms.

## Radial Basis Function (RBF) Networks
RBF networks typically feature three layers: input, a hidden layer with a non-linear RBF activation, and a linear output layer. The output is a weighted sum of radially symmetric functions dependent on the distance from hidden neuron centers.

RBF networks can be normalized, which has theoretical justification in stochastic data flow. Training generally proceeds in two steps: first, determining the hidden layer centers (often via k-means clustering), and second, fitting the linear model weights using least squares or gradient descent. RBF networks are effective for function approximation and interpolation.

## Hopfield Networks
Hopfield networks, a type of RNN, serve as associative memory systems. They use binary threshold units where interactions $w_{ij}$ are learned via Hebb's law, enabling the network to recall stored states from partial inputs. The network converges to local minima in an associated energy function.

The memory capacity of classical Hopfield networks scales linearly with the number of neurons, but modern Dense Associative Memories increase capacity by introducing higher-order interactions and stronger non-linearities. Training relies on Hebbian or Storkey learning rules, which are local and incremental.

## Convolutional Neural Networks (CNNs)
CNNs are designed to mimic the visual cortex by exploiting the spatial locality in data like images. Their core components include convolutional layers, pooling layers, and fully connected layers. Convolutional layers use shared weights (filters) that slide over the input to produce feature maps, drastically reducing parameter count.

Pooling layers (max or average) reduce spatial dimensions and grant local translational invariance. Advanced features include dilated convolutions to expand the receptive field without increasing parameters, and specialized regularization methods like Dropout and weight decay to prevent overfitting.

## Graph Neural Networks (GNNs)
GNNs process data structured as graphs. The architecture fundamentally involves permutation equivariant layers for local information aggregation (message passing) and global pooling for a fixed-size graph representation.

Message Passing Neural Networks (MPNNs) define how nodes update their representations by aggregating messages from neighbors. Graph Convolutional Networks (GCNs) formalize this as a spectral approximation. Graph Attention Networks (GATs) introduce learnable attention coefficients to weigh neighbor importance, while Gated Graph Sequence Neural Networks (GGS-NNs) extend this to sequence output using GRU cells.

## Transformer Architecture
Transformers revolutionized sequence modeling by abandoning the sequential recurrence of RNNs, allowing for full parallel computation. They rely on self-attention and feed-forward networks stacked into encoder and decoder layers.

Key components include tokenizers, embedding layers augmented with positional encodings (sinusoidal or RoPE), and the multi-head attention mechanism, which calculates relevance between tokens via scaled dot-product attention. Variations exist, such as encoder-only (BERT) for representation learning, decoder-only (GPT) for generation, and encoder-decoder (T5) for sequence transduction.

## Variational Autoencoders (VAEs)
VAEs are generative models that optimize the evidence lower bound (ELBO). They use an encoder to map input data $x$ to parameters of a variational posterior $q_{\phi}(z|x)$ in the latent space $z$, and a decoder to reconstruct $\hat{x}$ from $z$.

Optimization is achieved by maximizing the ELBO, which balances the reconstruction error (e.g., MSE) and the KL-divergence between the approximate posterior and a chosen prior $p_{\theta}(z)$. Variations like $\beta$-VAE and Conditional VAEs extend this framework for specific learning goals.

## Generative Adversarial Networks (GANs)
GANs define a zero-sum game between a Generator ($\mu_G$), which tries to match a reference distribution ($\mu_{ref}$), and a Discriminator ($D$), which tries to distinguish between real and generated samples.

Training stability is a major challenge; issues include mode collapse (where the generator produces only a few types of outputs) and vanishing gradients. Wasserstein GANs (WGANs) improve stability by replacing the JS divergence with the Wasserstein metric, using a Lipschitz-constrained critic. Variants like CycleGAN handle unpaired domain translation, while StyleGAN series focuses on high-resolution, controllable image synthesis.

## Vision Transformers (ViT)
ViT adapted the Transformer's architecture to computer vision. It processes images by dividing them into fixed-size, overlapping patches, treating these patches as tokens, and feeding them into an encoder-only Transformer. This allows the model to capture long-range semantic relationships between image regions.

Variants like Swin Transformer introduce CNN-like local windows and merging layers to increase efficiency, while Masked Autoencoders (MAE) employ self-supervision by reconstructing masked patches. DINO and Masked Autoencoders demonstrate powerful self-supervised pretraining techniques.

## Multimodal Models
Multimodal models integrate different data types, often by tokenizing each modality. For instance, in vision-language models (e.g., LLaVA), a language model is connected to a vision model via a linear layer. Image generation models like Parti utilize encoder-decoder Transformers, where the encoder processes text prompts to generate an image token representation decoded into an image.