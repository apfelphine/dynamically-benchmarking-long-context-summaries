### **Wasserstein GAN (WGAN)**

#### **Motivation**
The Wasserstein GAN introduces the **Wasserstein distance** as a criterion for measuring the dissimilarity between distributions, replacing the Jensen-Shannon divergence used in traditional GANs. This distance is more efficient to compute due to its dual representation theorem (Kantorovich-Rubenstein duality), allowing for better gradient flow and avoiding issues like mode collapse and vanishing gradients.

#### **Key Concepts**
- **Wasserstein Distance**: Measures the "work" needed to transform one distribution into another, offering a more stable and interpretable metric compared to Jensen-Shannon divergence.
- **Critic (Discriminator)**: Restricted to functions with bounded Lipschitz norm, ensuring stable gradient updates.
- **Generator**: Optimized to minimize the Wasserstein distance between its generated distribution and the reference distribution.

#### **Training Methods**
1. **Weight Clipping**: Bounds the Lipschitz norm by clipping weights to a fixed range.
2. **Spectral Normalization**: Normalizes weights to ensure a Lipschitz constant of 1, improving gradient stability.
3. **Gradient Penalty**: Penalizes deviations from the Lipschitz constraint by adding a gradient penalty term to the loss function.

#### **Comparison with GAN**
- **Gradient Flow**: Wasserstein GAN provides better gradient signals, reducing variance in gradient estimators.
- **Mode Collapse**: Avoids mode collapse due to its robust distance metric, making it more stable for complex datasets like ImageNet.

---

### **Generative Adversarial Network (GAN)**

#### **Definition and Game Theory**
- **GAN Game**: A zero-sum game between a generator (G) and a discriminator (D), where:
  - **Generator** aims to minimize the loss: ${\mathbb{E}}_{x\sim \mu_{G}}[\ln(1-D(x))]$.
  - **Discriminator** aims to maximize the loss: ${\mathbb{E}}_{x\sim \mu_{ref}}[\ln D(x)] + {\mathbb{E}}_{x\sim \mu_{G}}[\ln(1-D(x))]$.
- **Optimal Discriminator**: Computes the Jensen-Shannon divergence, ensuring the discriminator outputs a probability close to 0.5 at equilibrium.

#### **Training Challenges**
- **Unstable Convergence**: Requires careful tuning of learning rates and techniques like **two-time-scale update rule (TTUR)**.
- **Mode Collapse**: Occurs when the generator focuses on a subset of the data, failing to generalize.
- **Vanishing Gradients**: Discriminator may become too accurate, making gradient updates ineffective.

#### **Variants**
- **Conditional GAN**: Adds conditional labels to generate specific outputs.
- **Wasserstein GAN (WGAN)**: Uses the Wasserstein distance for more stable training.
- **InfoGAN**: Introduces mutual information to disentangle latent factors.
- **CycleGAN**: Transforms images between domains without paired data.

---

### **Vision Transformer (ViT)**

#### **Overview**
- **Transformer Architecture**: Uses self-attention to capture long-range dependencies in images, replacing convolutional layers.
- **Patch Embedding**: Splits images into fixed-size patches, converting them into token embeddings.
- **Positional Encoding**: Adds positional information to embeddings to retain spatial context.

#### **Applications**
- **Image Classification**: Achieves state-of-the-art performance by leveraging global attention.
- **Object Detection**: Extends ViT for tasks requiring spatial localization.
- **Self-Supervised Learning**: Uses masked autoencoders (MAE) for unsupervised pretraining.

#### **Variants**
- **Masked Autoencoder (MAE)**: Trains ViT by masking patches and reconstructing them.
- **DINO**: Self-distillation method for robust feature learning.
- **Swin Transformer**: Uses shifted windows for efficient attention.
- **TimeSformer**: Extends ViT to video data with space-time attention.

---

### **Convolutional Neural Network (CNN)**

#### **Architecture**
- **Convolutional Layers**: Apply filters to extract features, reducing parameters compared to fully connected networks.
- **Pooling Layers**: Downsample feature maps to reduce dimensionality and enhance translation invariance.
- **Fully Connected Layers**: Perform final classification or regression.

#### **Key Features**
- **Local Connectivity**: Each neuron connects only to a small receptive field, preserving spatial structure.
- **Shared Weights**: Reduces parameters by sharing weights across spatial locations.
- **Translation Invariance**: Pooling and convolution layers make CNNs robust to spatial shifts.

#### **Applications**
- **Image Recognition**: Dominates tasks like object classification and detection.
- **Video Analysis**: Combines spatial and temporal features for action recognition.
- **Natural Language Processing**: Used in tasks requiring sequence-to-sequence modeling.

---

### **Variational Autoencoder (VAE)**

#### **Architecture**
- **Encoder**: Maps input data to a latent space distribution.
- **Decoder**: Reconstructs input data from latent representations.
- **Latent Space**: Composed of a mixture of distributions (e.g., Gaussian) for probabilistic modeling.

#### **Training**
- **Evidence Lower Bound (ELBO)**: Optimizes the reconstruction error and KL divergence between approximate and true posterior distributions.
- **Reparameterization Trick**: Allows gradient-based optimization by sampling latent variables from a distribution.

#### **Variants**
- **Sparse Autoencoder**: Encourages sparsity in latent representations.
- **Denoising Autoencoder**: Learns robust representations by reconstructing noisy inputs.
- **Contractive Autoencoder**: Penalizes sensitivity to input perturbations for better generalization.

---

### **Radial Basis Function Network (RBFN)**

#### **Architecture**
- **Input Layer**: Processes raw data.
- **Hidden Layer**: Uses radial basis functions (e.g., Gaussian) for local approximation.
- **Output Layer**: Linear combination of hidden layer outputs.

#### **Training**
- **Interpolation**: Centers are chosen from training data, and weights are solved via linear algebra.
- **Function Approximation**: Optimizes weights to minimize error using gradient descent or pseudoinverse methods.

#### **Applications**
- **Time Series Prediction**: Effective for chaotic systems like the logistic map.
- **Function Approximation**: Universal approximator for continuous functions on compact domains.

---

### **Hopfield Network**
#### **Structure**
- **Binary Threshold Units**: Each neuron has two states (+1 or -1).
- **Symmetric Connections**: Weights are symmetric, ensuring stable dynamics.

#### **Training**
- **Hebbian Learning**: Updates weights based on simultaneous activations of neurons.
- **Spurious Patterns**: Can converge to incorrect patterns, especially with limited memory capacity.

#### **Applications**
- **Associative Memory**: Retrieves stored patterns from partial inputs.
- **Optimization**: Solves combinatorial problems like the traveling salesman problem.

---

### **Graph Neural Network (GNN)**
#### **Architecture**
- **Message Passing**: Aggregates information from neighbors to update node representations.
- **Local Pooling**: Reduces graph size via downsampling.
- **Global Pooling**: Provides fixed-size graph representations.

#### **Variants**
- **Graph Convolutional Network (GCN)**: Uses spectral filtering for local interactions.
- **Graph Attention Network (GAT)**: Applies attention mechanisms to focus on relevant nodes.
- **Bidirectional Graph Sequence Neural Network (GGS-NN)**: Extends GNNs for sequential data.

#### **Applications**
- **Protein Folding**: Predicts protein structures using molecular graphs.
- **Social Networks**: Recommender systems and fraud detection.

---
### **Recurrent Neural Network (RNN)**
#### **Architecture**
- **Hidden State**: Maintains memory across time steps.
- **Bidirectional RNNs**: Processes sequences in both directions for richer context.

#### **Training**
- **Teacher Forcing**: Uses ground truth labels for training.
- **Backpropagation Through Time (BPTT)**: Computes gradients over time.

#### **Variants**
- **Long Short-Term Memory (LSTM)**: Mitigates vanishing gradients with gating mechanisms.
- **Gated Recurrent Unit (GRU)**: Simplified LSTM variant.

#### **Applications**
- **Time Series Prediction**: Forecasts future values.
- **Natural Language Processing**: Sequence modeling and translation.

---
### **Autoencoder**
#### **Purpose**
- **Dimensionality Reduction**: Compresses data into a lower-dimensional latent space.
- **Feature Learning**: Extracts meaningful representations for downstream tasks.

#### **Variations**
- **Sparse Autoencoder**: Encourages sparsity in latent representations.
- **Denoising Autoencoder**: Learns robust features by reconstructing noisy inputs.
- **Contractive Autoencoder**: Penalizes sensitivity to input perturbations.

#### **Applications**
- **Anomaly Detection**: Flags unusual patterns by high reconstruction error.
- **Image Compression**: Reduces image size while preserving quality.

---
### **Key Takeaways**
- **WGAN** improves GAN stability with Wasserstein distance and Lipschitz constraints.
- **ViT** leverages self-attention for image understanding, outperforming CNNs in many tasks.
- **CNNs** excel in spatial data processing due to local connectivity and shared weights.
- **VAEs** enable probabilistic modeling and generative tasks via latent space distributions.
- **RBFNs** offer local approximation with radial basis functions.
- **Hopfield Networks** specialize in associative memory and optimization.
- **GNNs** handle graph-structured data with message passing and attention.
- **RNNs** process sequential data, with LSTMs/GRUs addressing gradient issues.