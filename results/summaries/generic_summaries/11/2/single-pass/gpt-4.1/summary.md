# Summary of Neural Network Architectures

---

## Convolutional Neural Network (CNN)

**Architecture & Building Blocks:**  
CNNs comprise input, hidden (convolutional, pooling, normalization), and output layers. Convolutional layers extract local features via learnable filters, pooling layers reduce spatial dimensions, and fully connected layers make final predictions. Neurons process restricted regions (receptive fields), and parameter sharing reduces model size and boosts generalization. Advanced techniques include depthwise separable convolutions and dilated convolutions for receptive field enlargement.

**Training & Regularization:**  
CNNs are trained via backpropagation using loss functions suited to the task (e.g., softmax for classification). Regularization methods such as dropout, weight decay, early stopping, and data augmentation combat overfitting.

**Distinctive Features:**  
CNNs leverage local connectivity, weight sharing, and pooling for efficient hierarchical feature extraction, enabling scalability and robustness, especially for image data.

**Applications:**  
Widely used in image and video recognition, natural language processing, anomaly detection, drug discovery, games (Go, checkers), and time series forecasting. GPU acceleration has been central to recent breakthroughs.

---

## Recurrent Neural Network (RNN)

**Architecture & Variants:**  
RNNs process sequential data with hidden state vectors that capture temporal dependencies. Common variants include Elman and Jordan networks, LSTM (for long-range memory), GRU (simplified LSTM), and bidirectional and stacked RNNs. They can be combined in encoder-decoder configurations for tasks like machine translation.

**Training:**  
Trained via backpropagation through time (BPTT) or real-time recurrent learning; teacher forcing is used for efficient sequence modeling. Issues with vanishing/exploding gradients are mitigated with LSTM/GRU.

**Applications:**  
Used in machine translation, speech recognition/synthesis, anomaly detection, time series modeling, music/rhythm learning, handwriting, action recognition, and more.

---

## Hopfield Network

**Structure & Properties:**  
Hopfield networks are recurrent nets serving as content-addressable associative memory. Units are binary (or continuous), interconnected with symmetric weights, and update asynchronously or synchronously to reach attractor states (energy minima).

**Training:**  
Weights are set via Hebbian learning or the more advanced Storkey rule for higher capacity and reduced error rates. Learning is local and incremental.

**Applications & Extensions:**  
Model associative/human memory, optimization problems, and form the basis for dense associative memories (modern Hopfield networks), which generalize classical models to super-linear storage capacities.

---

## Transformer

**Architecture:**  
Transformers eschew recurrence for multihead self-attention, enabling parallel processing of tokens and contextual feature extraction. Consists of tokenization, embedding, encoder/decoder layers, residual connections, and positional encodings.

**Training:**  
Pretrained on large corpora via masked or autoregressive language modeling, then fine-tuned for tasks. Layer normalization and learning rate warmup stabilize training. Efficient inference uses techniques like KV caching and FlashAttention.

**Variants & Extensions:**  
Adapted for multimodality (vision, audio, etc.), sub-quadratic designs for long inputs, and efficient implementations. Key models include BERT (encoder-only) and GPT (decoder-only).

**Applications:**  
Dominates NLP (translation, summarization, search, coding, speech) and is increasingly used in vision, protein folding, and games.

---

## Generative Adversarial Network (GAN)

**Definition:**  
GANs pit a generator (producing synthetic data) against a discriminator (distinguishing real from fake). The objective is a minimax game based on divergences (Jensen–Shannon in classical GAN, Wasserstein in WGAN).

**Training & Challenges:**  
Unstable convergence, mode collapse, and vanishing gradients are practical issues, addressed by techniques like TTUR and architectural/loss function variants.

**Variants:**  
Conditional GANs, WGAN, CycleGAN, InfoGAN, BiGAN, and the StyleGAN family extend capabilities (conditional generation, improved stability, style transfer).

**Applications:**  
Image generation, scientific simulations, privacy-preserving medical imaging, fake media detection, creativity/art, molecular discovery, and more.

---

## Radial Basis Function (RBF) Network

**Architecture:**  
Consists of input, hidden (RBF-activated), and output layers. Functions approximate any continuous mapping via weighted sums of radial basis functions centered on data points.

**Training:**  
Centers determined by clustering or selection; output weights solved via least-squares minimization or pseudoinverse; regularization may be applied.

**Applications:**  
Function approximation, time series prediction, interpolation, and system control, particularly for non-linear/chaotic data.

---

## Autoencoder & Variational Autoencoder (VAE)

**Architecture:**  
Autoencoders consist of encoder and decoder networks mapping data to/from a lower-dimensional latent space for reconstruction. VAEs combine this with probabilistic modeling and variational inference for generative tasks.

**Training:**  
Optimized via reconstruction loss and, in VAEs, KL divergence or other statistical distances to regularize latent distributions. Variants include sparse, denoising, contractive, minimum description length, and concrete autoencoders.

**Applications:**  
Dimensionality reduction, anomaly detection, lossy compression, information retrieval, drug discovery, popular content prediction, and as components in larger generative models.

---

## Vision Transformer (ViT)

**Architecture:**  
Adapts the Transformer to images by operating on sequences of patched embeddings. Processes images similarly to text, using self-attention for global context.

**Variants:**  
Includes masked autoencoders, DINO (self-distillation), Swin Transformer (localized attention), TimeSformer (video), ViT-VQGAN (vector quantized generative modeling), and methods for efficient pooling and multi-layer fusion.

**Applications:**  
Image classification, segmentation, generation (as GAN/diffusion backbone), anomaly detection, medical imaging, climate prediction, and more.

---

## Graph Neural Network (GNN)

**Architecture & Layers:**  
GNNs process non-Euclidean structured data (graphs) via permutation-equivariant message passing, local pooling, and global pooling layers. Variants include GCN, GAT (attention-based), and GGS-NN (sequential output).

**Training & Expressivity:**  
Message passing aggregates local graph information; stacking increases receptive field but too many layers lead to oversmoothing/squashing. GNNs are bounded in expressivity by graph isomorphism tests.

**Applications:**  
Protein folding (AlphaFold), social/recommender networks, combinatorial optimization, cybersecurity (anomaly detection), water distribution, computer vision (patch graphs), and text analysis.

---

## Class Activation Mapping (CAM) & Variants

**Purpose:**  
CAM and Grad-CAM visualize which regions of an input contributed to a neural network's prediction, especially for CNNs. CAM uses architectural modifications (GAP layer), while Grad-CAM uses gradient-based weighting for flexibility.

**Variants:**  
Guided Grad-CAM, Grad-CAM++, Score-CAM, and LayerCAM offer improved localization, class-specificity, and independence from architectural restrictions.

---

## Wasserstein GAN

**Overview:**  
Wasserstein GAN modifies the original GAN objective by measuring generator-discriminator divergence using the Wasserstein (Earth Mover’s) distance, leading to improved gradient flow and stability. The Lipschitz constraint on the discriminator is enforced via weight clipping, spectral normalization, or gradient penalty.

---

## Overall Connections  
- CNNs, RNNs, Transformers, and GNNs all constitute foundational neural network architectures for different data types and tasks.
- Autoencoders and GANs serve as core generative modeling methods, with VAEs unifying probabilistic generative and inference approaches.
- Modern architectures increasingly combine elements from multiple paradigms to achieve superior expressivity, scalability, and performance across vision, language, graphs, and generative tasks.