### **Convolutional Neural Network (CNN)**

#### **Architecture**
A CNN consists of an input layer, hidden layers (including convolutional, pooling, fully connected, and normalization layers), and an output layer. Convolutional layers apply learnable filters to input data, generating feature maps that capture spatial hierarchies. Pooling layers reduce dimensionality while retaining key features, and fully connected layers perform classification. CNNs leverage shared weights and local connectivity to efficiently process spatial data, reducing computational complexity compared to fully connected networks.

#### **Key Features**
- **Convolutional Layers**: Use shared filters to detect features (edges, textures) at different scales, reducing parameters via weight sharing.
- **Pooling Layers**: Downsample feature maps (e.g., max pooling) to reduce spatial dimensions, enhancing translation invariance.
- **Receptive Fields**: Each neuron’s receptive field grows with layer depth, enabling hierarchical feature abstraction.
- **Dilated Convolutions**: Expand receptive fields without increasing parameters, improving long-range dependencies.

#### **Applications**
CNNs excel in image recognition, object detection, and video analysis. They are foundational in computer vision tasks like facial recognition, medical imaging, and autonomous driving, often outperforming traditional MLPs due to their spatial locality and hierarchical feature learning.

---

### **Hopfield Network**
A Hopfield network is a recurrent neural network designed for associative memory, storing patterns as stable states. Its architecture consists of binary threshold units with symmetric, learned weights, enabling content-addressable memory.

#### **Training & Dynamics**
- **Learning Rules**: Weights are updated via Hebbian learning, where simultaneous activations strengthen connections.
- **Convergence**: Networks converge to stable states (attractors) via energy minimization, ensuring robustness to input perturbations.
- **Capacity**: Limited by the number of neurons (e.g., ~138 patterns per 1000 neurons), with modern variants (dense associative memories) achieving exponential storage capacity.

#### **Applications**
Hopfield networks model human associative memory, solve optimization problems (e.g., traveling salesman), and are used in pattern recognition and cognitive simulations.

---

### **Transformer**
Transformers revolutionize sequence modeling by replacing recurrent connections with self-attention mechanisms, enabling parallel processing.

#### **Architecture**
- **Tokenization & Embeddings**: Input text is split into tokens (e.g., words/subwords) and embedded into dense vectors.
- **Self-Attention**: Captures long-range dependencies by weighting token relationships dynamically.
- **Positional Encoding**: Adds positional context to embeddings, enabling order-aware processing.
- **Encoder-Decoder**: Encoders compress input sequences; decoders generate outputs autoregressively.

#### **Training**
- **Pretrain-Finetune**: Models are pretrained on large unlabeled data (e.g., language modeling) and fine-tuned for specific tasks.
- **Loss Functions**: Variants like masked language modeling or next-sentence prediction guide training.

#### **Applications**
Transformers dominate NLP (e.g., BERT, GPT-3), machine translation, and multimodal tasks (e.g., vision-language models). They also extend to video, speech, and generative AI.

---

### **Recurrent Neural Network (RNN)**
RNNs process sequential data by maintaining hidden states, enabling temporal dependencies.

#### **Variants**
- **LSTM/GRU**: Mitigate vanishing gradients via gating mechanisms, enabling long-term memory.
- **Bidirectional RNNs**: Process sequences in both directions, capturing past and future context.
- **Encoder-Decoder**: Combines RNNs for sequence-to-sequence tasks (e.g., machine translation).

#### **Training**
- **Backpropagation Through Time (BPTT)**: Computes gradients across time steps, though computationally expensive.
- **Teacher Forcing**: Uses ground truth labels during training to stabilize learning.

#### **Applications**
RNNs excel in time-series forecasting, speech recognition, and natural language processing (e.g., grammar learning, text generation).

---

### **Generative Adversarial Network (GAN)**
GANs consist of a generator and discriminator, trained in a minimax game to produce realistic data.

#### **Training Challenges**
- **Mode Collapse**: Generator may focus on a subset of data, ignoring others.
- **Wasserstein GAN**: Addresses instability by using Lipschitz-constrained discriminators.

#### **Applications**
GANs generate images (e.g., faces, art), simulate physics (e.g., particle collisions), and enhance medical imaging. Variants like CycleGAN enable domain translation without paired data.

---

### **Radial Basis Function Network (RBF)**
RBF networks approximate functions using Gaussian "basis functions" centered at learned parameters.

#### **Architecture**
- **Three Layers**: Input → Hidden (Gaussian activation) → Output (linear).
- **Normalization**: Encodes data into a latent space for improved generalization.

#### **Training**
- **Two-Phase**: First, centers are determined (e.g., via clustering); second, weights are optimized via least squares.

#### **Applications**
RBF networks model chaotic systems, time-series prediction, and control theory.

---

### **Variational Autoencoder (VAE)**
VAEs combine autoencoding with probabilistic modeling, using a prior and variational posterior to generate data.

#### **Key Components**
- **Encoder**: Maps data to a latent distribution (e.g., Gaussian).
- **Decoder**: Reconstructs data from latent samples.
- **ELBO Loss**: Balances reconstruction error and KL divergence between approximate and true posteriors.

#### **Applications**
VAEs generate synthetic data (e.g., images, molecules), compress data, and enable unsupervised learning.

---

### **Vision Transformer (ViT)**
ViTs replace CNNs’ convolutional filters with self-attention, treating images as sequences of patches.

#### **Architecture**
- **Patch Embeddings**: Images split into patches, each converted to a vector.
- **Self-Attention**: Processes patches in parallel, capturing global relationships.
- **Positional Encoding**: Adds positional context to embeddings.

#### **Variants**
- **Masked Autoencoder (MAE)**: Trains ViTs by masking patches, improving efficiency.
- **DINO**: Self-distillation method for pretraining ViTs without labels.

#### **Applications**
ViTs achieve state-of-the-art results in image classification, segmentation, and generative tasks.

---
### **Graph Neural Network (GNN)**
GNNs model relationships between nodes in graph-structured data using message passing.

#### **Core Layers**
- **Message Passing**: Nodes update representations via neighbor interactions.
- **Local Pooling**: Aggregates node features (e.g., k-nearest neighbors).
- **Global Pooling**: Produces fixed-size graph representations (e.g., sum/mean).

#### **Variants**
- **Graph Convolutional Networks (GCN)**: Uses spectral filtering.
- **Graph Attention Networks (GAT)**: Learns attention weights for node interactions.

#### **Applications**
GNNs solve problems in chemistry (e.g., drug discovery), social networks, and cybersecurity.

---
### **Class Activation Mapping (CAM)**
CAM visualizes which regions in an image contribute most to a CNN’s class prediction.

#### **Methods**
- **CAM**: Uses global average pooling to highlight discriminative regions.
- **Grad-CAM**: Computes gradients of class scores w.r.t. feature maps, emphasizing influential pixels.

#### **Applications**
CAM aids interpretability in medical imaging, autonomous driving, and explainable AI.

---
### **Wasserstein GAN**
Wasserstein GANs use the Wasserstein distance to improve stability and avoid mode collapse.

#### **Key Improvements**
- **Lipschitz-Constrained Discriminator**: Ensures discriminator gradients are bounded.
- **Gradient Penalty**: Penalizes discriminator gradients to enforce Lipschitz continuity.

#### **Applications**
Wasserstein GANs generate high-quality images and simulate physical phenomena.