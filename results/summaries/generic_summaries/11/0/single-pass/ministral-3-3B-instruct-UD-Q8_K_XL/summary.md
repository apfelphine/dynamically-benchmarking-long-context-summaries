### **Generative Adversarial Network (GAN)**

#### **Definition**
A GAN is a zero-sum game between two neural networks: a **generator** and a **discriminator**. The generator aims to create data resembling a reference distribution, while the discriminator distinguishes between real and generated data. The adversarial training process refines the generator to produce increasingly realistic samples.

- **Mathematical Formulation**: The game is defined over a probability space, with the generator optimizing to minimize a loss function, while the discriminator maximizes it. The generator’s goal is to approximate the reference distribution, and the discriminator’s goal is to distinguish between real and generated data.
- **Practical Implementation**: The generator creates samples from a latent space, while the discriminator evaluates them. Training involves iterative updates where the generator improves its ability to fool the discriminator, and the discriminator refines its ability to distinguish real from fake data.

#### **Relation to Other Methods**
GANs are **implicit generative models**, unlike alternatives like flow-based models or variational autoencoders (VAEs), which explicitly model likelihood functions. GANs can generate complete samples in one pass, unlike autoregressive models, which require multiple passes.

#### **Mathematical Properties**
- **Optimal Discriminator**: The discriminator’s optimal strategy is to compute the Jensen–Shannon divergence between the generator and reference distributions, ensuring it maximizes its ability to distinguish them.
- **Equilibrium**: The GAN game has a unique equilibrium where the generator perfectly mimics the reference distribution, and the discriminator outputs a fixed probability (typically 0.5), indicating no distinguishable difference.

#### **Training and Evaluation**
- **Instability**: GANs often suffer from unstable convergence due to the adversarial nature of training. Techniques like curriculum learning, two-time-scale update rules, and Wasserstein GANs (WGAN) help stabilize training.
- **Mode Collapse**: A common issue is mode collapse, where the generator fails to capture diverse modes of the data distribution, leading to poor generalization.
- **Evaluation Metrics**: GANs are evaluated using metrics like Inception Score (IS) and Fréchet Inception Distance (FID), which measure sample diversity and similarity to real data.

#### **Variants**
- **Conditional GANs**: Allow conditional generation based on additional information (e.g., class labels).
- **Wasserstein GAN (WGAN)**: Uses the Wasserstein distance to improve gradient stability and avoid mode collapse.
- **CycleGAN**: Enables domain translation without paired data by enforcing cycle consistency.
- **StyleGAN**: Uses progressive growth and style mixing to generate high-quality images with fine details.

---

### **Vision Transformer (ViT)**

#### **Overview**
The Vision Transformer (ViT) replaces convolutional layers with self-attention mechanisms, breaking images into patches and processing them as tokens in a transformer architecture. ViT achieves state-of-the-art performance in image classification by capturing long-range dependencies between patches.

- **Key Components**: ViT splits images into patches, embeds them with positional encodings, and processes them through transformer encoder layers. The final output is decoded into a classification probability distribution.

#### **Variants**
- **Masked Autoencoder (MAE)**: Uses a masked autoencoder approach to train ViT by reconstructing unmasked patches, improving efficiency and robustness.
- **DINO**: A self-distillation method that trains ViT by aligning representations between different image crops, enhancing feature learning.
- **Swin Transformer**: Uses sliding windows and hierarchical pooling to improve efficiency and performance in object detection tasks.

#### **Comparison with CNNs**
ViT is more sensitive to hyperparameters and training stability but can handle larger datasets and more complex tasks. CNNs are more data-efficient and less sensitive to input distortions, while ViT excels in capturing long-range dependencies.

#### **Applications**
ViT is used in image classification, object detection, video analysis, and generative tasks like image synthesis. It has also been adapted for medical imaging and autonomous driving.

---

### **Recurrent Neural Network (RNN)**

#### **Architecture**
RNNs process sequential data by maintaining a hidden state that captures temporal dependencies. They include:
- **Standard RNN**: Processes sequences by updating hidden states iteratively.
- **Stacked RNN**: Combines multiple RNNs to capture hierarchical temporal patterns.
- **Bidirectional RNN (BiRNN)**: Uses two RNNs to process sequences in both directions, capturing past and future context.
- **LSTM/GRU**: Specialized RNNs with gates to mitigate vanishing/exploding gradients.

#### **Training Methods**
- **Teacher Forcing**: Uses ground truth labels for training sequences to improve stability.
- **Gradient Descent**: Uses backpropagation through time (BPTT) or real-time recurrent learning (RTRL) to update weights.
- **Connectionist Temporal Classification (CTC)**: A loss function for sequence modeling tasks with variable timing.

#### **Applications**
RNNs are used in machine translation, speech recognition, time series forecasting, and natural language processing. They are also applied in robotics, bioinformatics, and anomaly detection.

---

### **Variational Autoencoder (VAE)**

#### **Architecture**
A VAE consists of an encoder that maps input data to a latent space and a decoder that reconstructs the input from latent representations. It uses variational inference to approximate the posterior distribution of latent variables.

- **Key Components**: Encoder (parameterizes latent distribution), decoder (reconstructs input), and KL divergence term to ensure the latent distribution matches a prior.

#### **Variations**
- **Sparse Autoencoder (SAE)**: Encourages sparsity in latent representations to improve feature learning.
- **Contractive Autoencoder (CAE)**: Adds a contractive regularization term to make the encoder robust to input perturbations.
- **Denoising Autoencoder (DAE)**: Trains the autoencoder to reconstruct noisy inputs, improving robustness.

#### **Applications**
VAEs are used for dimensionality reduction, anomaly detection, image denoising, and generative modeling. They are also applied in drug discovery, popularity prediction, and machine translation.

---

### **Radial Basis Function Network (RBFN)**

#### **Architecture**
RBFNs consist of an input layer, a hidden layer with radial basis functions (RBFs), and an output layer. RBFs are typically Gaussian functions centered at specific points in the input space.

- **Key Components**: Input layer (real-valued vectors), hidden layer (RBFs with centers and widths), and output layer (linear combination of RBF outputs).

#### **Training**
RBFNs are trained using two-step methods: first, centers are determined (e.g., via clustering), and then weights are optimized using gradient descent or pseudoinverse solutions.

#### **Applications**
RBFNs are used in function approximation, time series prediction, and control theory. They are also applied in pattern recognition and interpolation tasks.

---

### **Graph Neural Network (GNN)**

#### **Architecture**
GNNs process graph-structured data by aggregating information from neighboring nodes. They include:
- **Message Passing Layers**: Update node representations by aggregating messages from neighbors.
- **Local Pooling**: Downsample graphs to increase receptive fields.
- **Global Pooling**: Provide fixed-size representations of the entire graph.

#### **Variants**
- **Graph Convolutional Network (GCN)**: Uses spectral filtering to approximate graph convolutions.
- **Graph Attention Network (GAT)**: Incorporates attention mechanisms to focus on important neighbors.
- **Gated Graph Sequence Neural Network (GGS-NN)**: Extends GNNs to output sequences.

#### **Applications**
GNNs are used in protein folding, social network analysis, cybersecurity, and computer vision. They are also applied in combinatorial optimization, drug discovery, and recommendation systems.

---

### **Autoencoder**

#### **Architecture**
An autoencoder consists of an encoder that compresses input data into a latent representation and a decoder that reconstructs the input from the latent code. It is trained to minimize reconstruction error.

- **Key Components**: Encoder (reduces dimensionality), decoder (reconstructs input), and reconstruction loss (e.g., L2 loss).

#### **Variations**
- **Variational Autoencoder (VAE)**: Uses probabilistic latent variables to model data distributions.
- **Sparse Autoencoder (SAE)**: Encourages sparsity in latent representations.
- **Denoising Autoencoder (DAE)**: Trains to reconstruct noisy inputs.

#### **Applications**
Autoencoders are used for dimensionality reduction, anomaly detection, image processing, and generative modeling. They are also applied in drug discovery, popularity prediction, and machine translation.