### **Convolutional Neural Networks (CNNs)**
- **Architecture**: Composed of input, hidden (convolutional, pooling, normalization), and output layers. Convolutional layers apply filters to detect features via dot products, producing feature maps. Pooling reduces spatial dimensions, while fully connected layers classify images.
- **Key Features**: 
  - **Local connectivity** and **shared weights** reduce parameters and enable translational invariance.
  - **Receptive fields** define the region of input each neuron considers, expanding through layers.
- **History**: Inspired by Hubel & Wiesel’s visual cortex research. Fukushima’s Neocognitron introduced shared weights and pooling. Modern CNNs use backpropagation for training.
- **Applications**: Image recognition, object detection, video analysis, drug discovery, and game AI.
- **Advantages**: Efficient for spatial data, reduces overfitting via parameter sharing, and achieves high accuracy in vision tasks.

### **Hopfield Networks**
- **Structure**: Recurrent neural networks with binary neurons and symmetric weights. Stores memories via Hebbian learning.
- **Function**: Acts as an associative memory, where partial inputs are used to retrieve stored patterns.
- **Key Properties**: 
  - Converges to stable states (attractors) with energy minimization.
  - Exhibits spurious patterns and limited capacity.
- **Modern Variants**: Dense Associative Memories (Dense AMs) with higher capacity via nonlinear energy functions and improved dynamics.
- **Applications**: Optimization problems, memory modeling, and pattern recognition.

### **Transformers**
- **Architecture**: Uses self-attention and feedforward layers. Processes sequences in parallel, enabling efficient long-range dependency modeling.
- **Key Components**: 
  - **Positional encoding** provides sequence order information.
  - **Multi-head attention** allows parallel processing of different feature relationships.
- **Training**: Uses pretraining (e.g., masked language modeling) and fine-tuning on specific tasks.
- **Applications**: Natural language processing (NLP), machine translation, speech recognition, and multimodal tasks (e.g., vision, audio).
- **Efficiency**: Enables faster inference and better performance in long sequences compared to RNNs.

### **Recurrent Neural Networks (RNNs)**
- **Architecture**: Processes sequences step-by-step. Maintains hidden state to capture temporal dependencies.
- **Types**: 
  - **LSTM** and **GRU** address vanishing gradients via gating mechanisms.
  - **Bidirectional RNNs** process sequences in both directions.
- **Applications**: Speech recognition, machine translation, time series prediction, and sequence generation.
- **Limitations**: Sequential processing limits parallelization and efficiency.

### **Generative Adversarial Networks (GANs)**
- **Mechanism**: Two networks (generator and discriminator) in a zero-sum game. Generator creates realistic data; discriminator evaluates it.
- **Variants**: 
  - **Conditional GANs** generate data based on labels.
  - **Wasserstein GAN (WGAN)** improves training stability via Wasserstein distance.
- **Applications**: Image generation, style transfer, medical imaging, and synthetic data creation.
- **Challenges**: Mode collapse and unstable convergence.

### **Autoencoders**
- **Architecture**: Encodes input into a latent space and decodes it back. Includes sparse, denoising, and variational variants.
- **Types**:
  - **VAE**: Uses probabilistic latent space with KL divergence for disentangled representations.
  - **Sparse Autoencoders**: Encourage sparse activations for better feature learning.
  - **Denoising Autoencoders**: Improve robustness by reconstructing from corrupted inputs.
- **Applications**: Dimensionality reduction, anomaly detection, image denoising, and feature learning.

### **Vision Transformers (ViTs)**
- **Architecture**: Replaces CNNs with self-attention on image patches. Uses patch embeddings and positional encoding.
- **Variants**: 
  - **MAE (Masked Autoencoder)**: Self-supervised pretraining with masked patches.
  - **DINO**: Self-distillation for robust feature learning.
  - **Swin Transformer**: Uses shifted windows for efficient attention.
- **Applications**: Image classification, object detection, video understanding, and image generation.

### **Graph Neural Networks (GNNs)**
- **Architecture**: Processes graph-structured data via message passing between nodes.
- **Layers**: 
  - **Message passing** aggregates information from neighbors.
  - **Pooling** reduces graph size; **global pooling** provides fixed-size output.
- **Applications**: Protein folding, social network analysis, cybersecurity, and recommendation systems.

### **Class Activation Mapping (CAM)**
- **Purpose**: Generates heatmaps to highlight discriminative regions in CNN predictions.
- **Methods**:
  - **CAM**: Uses global average pooling and class-specific weights.
  - **Grad-CAM**: Computes gradients of class scores with respect to feature maps.
- **Fine-tuned Versions**: Guided Grad-CAM, Grad-CAM++, Score-CAM, and Layer-CAM improve localization and resolution.

### **Variational Autoencoders (VAEs)**
- **Architecture**: Models data as a mixture of Gaussians; learns latent variables via variational inference.
- **Loss Function**: Evidence Lower Bound (ELBO) balances reconstruction and KL divergence.
- **Applications**: Generative modeling, disentangled representations, and data synthesis.