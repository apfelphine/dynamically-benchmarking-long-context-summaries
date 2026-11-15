### **Radial Basis Function Networks (RBFN)**

#### **Architecture**
RBF networks consist of three layers: an input layer, a hidden layer with non-linear RBF activation functions, and a linear output layer. The output function is defined as:
\[
\varphi(\mathbf{x}) = \sum_{i=1}^{N} a_i \rho(|| \mathbf{x} - \mathbf{c}_i ||)
\]
where \(\rho\) is a Gaussian RBF function, \(N\) is the number of hidden neurons, \(\mathbf{c}_i\) are the centers, and \(a_i\) are weights. RBFs are locally sensitive, meaning their influence diminishes as input distance increases.

RBFNs are universal approximators on compact subsets of \(\mathbb{R}^n\), meaning they can approximate any continuous function with sufficient hidden neurons.

#### **Normalization**
Normalized RBFNs adjust the output by dividing the weighted sum of RBFs by the sum of RBFs themselves, creating normalized radial basis functions:
\[
\varphi(\mathbf{x}) = \sum_{i=1}^{N} a_i u(|| \mathbf{x} - \mathbf{c}_i ||)
\]
where \(u\) is the normalized RBF. This approach is theoretically justified for stochastic data flow, ensuring robustness in probabilistic modeling.

#### **Local Linear Models**
RBFNs can incorporate local linear models by extending the architecture to include linear terms in the hidden layer, enabling more complex feature interactions:
\[
\varphi(\mathbf{x}) = \sum_{i=1}^{N} \left(a_i + \mathbf{b}_i \cdot (\mathbf{x} - \mathbf{c}_i)\right) \rho(|| \mathbf{x} - \mathbf{c}_i ||)
\]
This allows for higher-order linear dependencies, improving model flexibility.

#### **Training**
RBFNs are trained in two steps:
1. **Center Selection**: Hidden layer centers (\(\mathbf{c}_i\)) are chosen via unsupervised methods like random sampling or k-means clustering.
2. **Linear Weights**: The hidden layer outputs are fitted to the target values using linear regression (e.g., least squares), optimizing the output layer weights.

Training can involve regularization (e.g., smoothness constraints) to balance accuracy and generalization.

#### **Interpolation & Function Approximation**
RBFNs can interpolate functions by using known points as centers. The weights are solved via linear algebra, ensuring exact reconstruction for distinct points. For function approximation, centers are randomly sampled or determined via clustering, followed by linear weight optimization.

#### **Applications**
RBFNs are versatile for tasks like time series prediction, anomaly detection, and pattern recognition, leveraging their ability to model complex, locally sensitive relationships.

---

### **Autoencoder**

#### **Architecture & Operation**
An autoencoder consists of an encoder and decoder, mapping input data to a compressed latent space and back. The encoder \(E_{\phi}\) maps \(\mathcal{X}\) to \(\mathcal{Z}\), and the decoder \(D_{\theta}\) reconstructs \(\mathcal{X}\) from \(\mathcal{Z}\). The goal is to minimize the reconstruction error between input and output.

#### **Variational Autoencoder (VAE)**
VAEs extend autoencoders by introducing probabilistic modeling. They use a variational posterior \(q_{\phi}(z|x)\) to approximate the true posterior \(p_{\theta}(z|x)\). The loss function combines reconstruction error and KL divergence:
\[
L_{\theta,\phi}(x) = \mathbb{E}_{z \sim q_{\phi}(\cdot |x)}[\ln p_{\theta}(x|z)] - D_{KL}(q_{\phi}(\cdot |x) || p_{\theta}(\cdot))
\]
VAEs use reparameterization tricks to enable gradient-based optimization.

#### **Variants**
- **Sparse Autoencoder (SAE)**: Encourages sparsity in latent representations to improve feature learning.
- **Denoising Autoencoder (DAE)**: Trains the model to reconstruct corrupted inputs, enhancing robustness.
- **Contractive Autoencoder (CAE)**: Adds contractive regularization to preserve feature stability under perturbations.
- **Minimum Description Length Autoencoder (MDL-AE)**: Optimizes for compact and interpretable latent representations.

#### **Depth & Training**
Autoencoders benefit from depth, reducing computational costs and improving compression. Training involves pretraining with deep belief networks (DBNs) and fine-tuning with backpropagation.

#### **Applications**
Autoencoders are widely used for dimensionality reduction, anomaly detection, image processing, and generative modeling, enabling tasks like denoising, compression, and unsupervised learning.

---

### **Recurrent Neural Network (RNN)**

#### **History & Configurations**
RNNs evolved from neuroscience-inspired models (e.g., Elman networks) to modern architectures like LSTM and GRU. Key configurations include:
- **Standard**: Processes sequences via hidden states, capturing temporal dependencies.
- **Stacked**: Multiple RNN layers for deeper learning.
- **Bidirectional**: Processes sequences in both directions, capturing context from past and future.
- **Encoder-Decoder**: Uses two RNNs for sequence-to-sequence tasks (e.g., machine translation).

#### **Training**
RNNs are trained using backpropagation through time (BPTT) or gradient descent variants like Real-Time Recurrent Learning (RTRL). Teacher forcing is used to improve training stability by masking future inputs during training.

#### **Applications**
RNNs excel in time-series prediction, speech recognition, natural language processing, and robotics, leveraging their ability to model sequential data.

---

### **Graph Neural Network (GNN)**

#### **Architecture**
GNNs implement permutation-equivariant message passing layers, local pooling, and global pooling. Message passing aggregates information from neighbors, while pooling reduces graph complexity. GNNs are limited by the Weisfeiler–Leman Graph Isomorphism Test, meaning they cannot distinguish certain graph structures.

#### **Message Passing Layers**
GNNs update node representations by aggregating messages from neighbors:
\[
\mathbf{h}_u = \phi(\mathbf{x}_u, \bigoplus_{v \in N_u} \psi(\mathbf{x}_u, \mathbf{x}_v, \mathbf{e}_{uv}))
\]
where \(\phi\) and \(\psi\) are differentiable functions, and \(\bigoplus\) is an aggregation operator.

#### **Variants**
- **Graph Convolutional Network (GCN)**: Uses spectral filtering for graph-structured data.
- **Graph Attention Network (GAT)**: Incorporates attention mechanisms to focus on relevant neighbors.
- **Gated Graph Sequence Neural Network (GGS-NN)**: Extends GNNs to output sequences via GRU cells.

#### **Applications**
GNNs are used in protein folding, social network analysis, cybersecurity, and computer vision, enabling tasks like node classification, link prediction, and anomaly detection.

---

### **Hopfield Network**

#### **Structure & Dynamics**
Hopfield networks are associative memory models with binary threshold units. They store patterns via Hebbian learning and retrieve them by minimizing an energy function:
\[
E = -\frac{1}{2} \sum_{i,j} w_{ij} s_i s_j - \sum_i \theta_i s_i
\]
where \(w_{ij}\) are connection weights, \(s_i\) are neuron states, and \(\theta_i\) are thresholds.

#### **Training & Capacity**
Training involves setting weights based on stored patterns. The network’s capacity is limited by the number of neurons, with a theoretical maximum of \(\approx \frac{n}{2 \log_2 n}\).

#### **Applications**
Hopfield networks are used in associative memory, optimization problems, and pattern recognition, leveraging their ability to store and retrieve patterns efficiently.

---

### **Class Activation Mapping (CAM)**

#### **Background**
CAM methods visualize class-discriminative regions in CNNs by highlighting pixels contributing most to class predictions. Key techniques include:
- **Class Model Visualization**: Synthesizes an artificial input image to activate target class neurons.
- **Saliency Maps**: Computes gradients of class scores to identify influential pixels.

#### **Base Versions**
- **CAM**: Uses global average pooling (GAP) to generate class-specific heatmaps.
- **Grad-CAM**: Generalizes CAM by computing gradients of class scores across feature maps, enabling visualization at intermediate layers.

#### **Variants**
- **Guided Grad-CAM**: Combines Grad-CAM with guided backpropagation for sharper visualizations.
- **Grad-CAM++**: Improves Grad-CAM by using higher-order gradients to focus on multiple instances.
- **Score-CAM**: Uses model confidence scores instead of gradients for more accurate heatmaps.

#### **Applications**
CAM methods are widely used in explainable AI, helping interpret CNNs’ decisions for tasks like image classification and object localization.

---

### **Transformer**

#### **History & Architecture**
Transformers introduced self-attention mechanisms, enabling parallel processing of sequences. Key components include:
- **Tokenization**: Converts text into tokens using subword tokenizers.
- **Embedding**: Maps tokens to dense vectors, incorporating positional encodings.
- **Encoder-Decoder**: Uses multi-head attention and feedforward layers for sequence modeling.

#### **Training**
Transformers are typically pretrained using self-supervised tasks (e.g., masked language modeling) and fine-tuned on downstream tasks. Training stability is improved with techniques like layer normalization and learning rate warmup.

#### **Variants**
- **RoPE**: Uses rotary positional encodings for better handling of variable-length sequences.
- **ALiBi**: Implements linear biases in attention to enable long-range dependencies.
- **Efficient Implementations**: FlashAttention and KV caching optimize attention computations.

#### **Applications**
Transformers revolutionized NLP, enabling tasks like machine translation, text generation, and multimodal learning. They are also applied in vision, speech, and generative modeling.

---

### **Vision Transformer (ViT)**

#### **Overview**
ViTs adapt transformers to computer vision by splitting images into patches and treating them as tokens. The architecture includes:
- **Patch Embedding**: Converts image patches into vectors.
- **Encoder**: Processes patches via self-attention, producing a global representation.

#### **Variants**
- **Masked Autoencoder (MAE)**: Uses ViTs to reconstruct masked patches, enabling unsupervised pretraining.
- **DINO**: Self-distills ViTs to learn robust representations.
- **Swin Transformer**: Uses shifted windows for efficient attention.

#### **Applications**
ViTs achieve state-of-the-art results in image classification, object detection, and generative modeling, often outperforming CNNs.

---

### **Generative Adversarial Network (GAN)**

#### **Definition**
GANs consist of a generator and discriminator, where the generator creates samples, and the discriminator distinguishes real from fake data. The game is defined by:
\[
L(\mu_G, D) = \mathbb{E}_{x \sim \mu_{ref}}[\ln D(x)] + \mathbb{E}_{x \sim \mu_G}[\ln (1-D(x))]
\]
where \(\mu_G\) is the generator distribution and \(D\) is the discriminator.

#### **Training & Evaluation**
GANs are trained via adversarial learning, often suffering from mode collapse. Evaluation metrics include Inception Score (IS) and Fréchet Inception Distance (FID). Variants like Wasserstein GAN address stability issues.

#### **Variants**
- **Conditional GAN (cGAN)**: Generates samples conditioned on labels.
- **CycleGAN**: Transforms images between domains without paired data.
- **StyleGAN**: Uses neural style transfer for high-quality image generation.

#### **Applications**
GANs are used in image synthesis, medical imaging, art generation, and adversarial learning, enabling tasks like image enhancement and style transfer.

---

### **Wasserstein GAN (WGAN)**
WGANs replace the GAN game with the Wasserstein distance, improving training stability by bounding the discriminator’s Lipschitz norm. The objective is:
\[
L_{WGAN}(\mu_G, D) = \mathbb{E}_{x \sim \mu_G}[D(x)] - \mathbb{E}_{x \sim \mu_{ref}}[D(x)]
\]
WGANs avoid mode collapse and provide better gradients for generator optimization. Training methods include weight clipping, spectral normalization, and gradient penalty.