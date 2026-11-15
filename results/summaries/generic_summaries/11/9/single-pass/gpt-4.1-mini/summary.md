# Summary

## Class Activation Mapping (CAM)

Class activation mapping techniques provide visual explanations of CNN decisions by highlighting class-discriminative input regions. Early works introduced **class model visualization** (generating idealized input patterns activating a target class) and **image-specific saliency maps** (using gradients of class scores w.r.t input pixels to identify important image areas). **Guided backpropagation** further refines interpretability by passing only positive gradients through ReLU units, resulting in sharper visualizations.

The original **CAM** method requires CNN architectures where the final convolutional layer is succeeded by a global average pooling (GAP) layer and replaces fully connected layers, enabling class-specific heatmaps by weighting feature maps with learned linear classifier weights. However, CAM is limited to such architectures and needs retraining.

**Grad-CAM** generalizes CAM by computing gradients of class scores relative to arbitrary convolutional layers’ feature maps, enabling application to any differentiable CNN without architectural changes or retraining. It produces coarse localization heatmaps using weighted feature maps, but can suffer from low resolution and focus only on the most discriminative regions.

Multiple **Grad-CAM variants** refine the method:  
- **Guided Grad-CAM** combines Grad-CAM with guided backpropagation for high-resolution explanations.  
- **Grad-CAM++** employs pixel-wise higher-order gradients for better multi-instance localization.  
- **Score-CAM** is gradient-free, weighting feature maps by increase in model confidence when masked, improving interpretability at the cost of speed.  
- **Layer-CAM** uses positive gradients on intermediate layers for precise localization.  

CAM and Grad-CAM remain foundational explainable AI tools for CNNs.

---

## Vision Transformer (ViT)

Transformers, originally developed for NLP (2017), were adapted to vision, leading to the **Vision Transformer (ViT)** in 2020, which splits images into patches, projects them into embeddings with positional encodings, then processes them with standard encoder-only transformers achieving state-of-the-art image classification.

Key components: patch embedding, positional encoding, transformer encoder layers, and classification head. Variants include:  
- **Pooling variants** replacing the [CLS] token with global average or multihead attention pooling.  
- **Masked Autoencoder (MAE)** for self-supervised learning by reconstructing masked patches, boosting data-efficiency and robustness, with derivatives tailored for medical imaging and video (e.g., SiamMAE).  
- **DINO**: self-distillation method for unsupervised ViT training with teacher-student models; improved by DINOv2 and DINOv3 with enhancements like image-text alignment and axial rotary positional embeddings.  
- **Swin Transformer** applies self-attention within shifted windows and hierarchical merging, achieving strong results in detection and segmentation.  
- **TimeSformer** uses factorized spatial and temporal attention for video tasks.  

ViTs differ from CNNs by explicitly modeling long-range relationships and being more sensitive to training regimes but more robust to distortions. They excel across classification, detection, video, anomaly detection, image synthesis, and large-scale modeling (e.g., weather prediction).

---

## Variational Autoencoder (VAE)

VAEs are generative models combining an **encoder** neural network producing parameters of a variational posterior distribution over latent variables, and a **decoder** generating data from latent samples. They optimize the **evidence lower bound (ELBO)**, balancing reconstruction fidelity and regularity of the latent distribution, typically via KL divergence to a prior.

Key concepts:  
- Encoder approximates intractable true posterior with learned variational distribution.  
- ELBO includes a reconstruction term (e.g., Gaussian likelihood) and KL divergence encouraging latent distributions close to a prior (often standard normal).  
- The **reparameterization trick** enables backpropagation through stochastic sampling by expressing latent variables as deterministic functions of noise and parameters.  
- Variants include β-VAE (disentanglement), Conditional VAE, sparse, denoising, and contractive autoencoders.  
- Alternative statistical distances (Wasserstein, MMD) replace KL divergence in advanced variants for improved performance.

VAEs allow amortized inference, efficient training and probabilistic generative modeling with interpretable latent spaces.

---

## Wasserstein GAN (WGAN)

Traditional GANs pit a generator against a discriminator trained to distinguish real from fake data, optimizing a minimax game related to Jensen-Shannon divergence, often suffering from training instability and mode collapse.

**WGAN** replaces Jensen-Shannon divergence with the **Wasserstein-1 metric (Earth Mover’s Distance)**, providing more meaningful gradients even when distributions have disjoint supports. This is achieved by constraining the discriminator (critic) to be Lipschitz continuous.

Core differences and methods:  
- Objective function reflects Wasserstein distance via Kantorovich-Rubinstein duality, requiring discriminator to be 1-Lipschitz.  
- Techniques to enforce Lipschitz constraint: weight clipping, spectral normalization, and gradient penalty.  
- WGAN provides more stable training, mitigates mode collapse, and offers better convergence properties.

Training alternates between generator updates minimizing Wasserstein distance and discriminator updates constrained by Lipschitz conditions.

---

## Graph Neural Networks (GNNs)

GNNs process graph-structured data by iteratively updating node representations via **permutation-equivariant message passing layers**, where each node aggregates feature messages from neighbors using differentiable functions.

Key layers:  
- **Message Passing:** Node features updated by aggregating transformed neighbor features and edge attributes.  
- **Local Pooling:** Graph coarsening/downsampling methods (e.g., top-k pooling, self-attention pooling) extend receptive fields and reduce graph size.  
- **Global Pooling (Readout):** Permutation-invariant aggregation (sum, mean, max) summarizing node representations into a global graph embedding.

Variants:  
- **Graph Convolutional Networks (GCN):** Spectral filtering approximations with normalized adjacency matrices.  
- **Graph Attention Networks (GAT):** Attention mechanisms compute normalized attention coefficients to weigh neighbor contributions.  
- **Gated Graph Sequence Neural Networks (GGS-NN):** Employ gated recurrent units to iteratively update node states.

GNNs have expressivity limitations related to the Weisfeiler–Leman test but are applied widely in social networks, biology (protein folding), combinatorial optimization, cybersecurity, water networks, vision, and NLP. Research is active on heterophilic graphs where standard homophily assumptions fail.

---

## Generative Adversarial Networks (GANs)

GANs consist of two neural networks—a **generator** producing synthetic samples from latent noise, and a **discriminator** distinguishing real from generated data—trained adversarially as a zero-sum game.

Key aspects:  
- The generator aims to match its distribution to the real data distribution; the discriminator aims to correctly classify real vs fake.  
- Training instability issues include **mode collapse**, **vanishing gradients**, and convergence difficulties; various improvements address these (e.g., Wasserstein GAN, TTUR).  
- Variants include: conditional GANs (for class-conditional generation), architectures like DCGAN, Self-Attention GAN, StyleGAN series, InfoGAN, BiGAN, and CycleGAN for domain translation without paired data.  
- Evaluated with metrics such as Inception Score, Fréchet Inception Distance, and Learned Perceptual Image Patch Similarity.  

Applications span image synthesis, medical imaging (synthetic MRI), fraud detection, art generation, data augmentation, and more. GANs raise ethical concerns due to potential misuse in deepfakes.

---

## Autoencoders

Autoencoders are neural networks trained to encode inputs into a typically lower-dimensional **latent space** via an **encoder**, and reconstruct inputs via a **decoder** aiming to minimize reconstruction error.

Variants enhance capabilities:  
- **Variational Autoencoders (VAE):** Probabilistic latent space with KL regularization.  
- **Sparse Autoencoders:** Encourage sparsity in latent representations.  
- **Denoising Autoencoders:** Train to reconstruct original inputs from corrupted versions, improving robustness.  
- **Contractive Autoencoders:** Regularize encoder sensitivity to inputs for robust representations.  
- **Concrete Autoencoders:** For discrete feature selection.  
- **Minimum Description Length Autoencoder:** Balances compression size and reconstruction error per information theory.

Applications include dimensionality reduction, anomaly detection, image denoising, drug discovery, NLP, and communication systems. Deep architectures and joint training improve performance.

---

## Transformer (Deep Learning)

Transformers are sequence-modeling architectures based solely on **attention mechanisms** rather than recurrence or convolution.

Key components:  
- **Multi-head self-attention:** Each token attends to all others, capturing global dependencies efficiently.  
- **Feedforward network layers:** Pointwise MLPs for nonlinear processing.  
- **Positional encodings:** Provide token order information via sinusoidal or learned embeddings.  
- **Residual connections and layer normalization:** Stabilize and accelerate training.

Variants: encoder-only (e.g., BERT), decoder-only (e.g., GPT), and encoder-decoder (original Transformer, T5). Masked and causal attention enable autoregressive or masked language modeling.

Training typically uses large-scale self-supervised pretraining (e.g., masked-token prediction) followed by fine-tuning on downstream tasks.

Efficient implementations (KV caching, FlashAttention) and variants (Reformer, sparse attention) address scalability. Transformers underpin advances in NLP, vision (Vision Transformers), multimodal models, protein folding, and more.

---

## Radial Basis Function (RBF) Networks

RBF networks are three-layer architectures with:  
- An input layer  
- A hidden layer of neurons with radial basis (typically Gaussian) activation functions centered on learnable points  
- A linear output layer

They approximate functions as weighted sums of radial basis functions of input distance to centers.

Normalized RBFs use weighted averages for smooth interpolation and probabilistic interpretation.

Training involves choosing centers (e.g., clustering), widths, and fitting linear output weights via least squares or gradient descent.

Applications include function approximation, time series prediction (e.g., logistic map chaotic sequences), control, and interpolation.

---

## Convolutional Neural Networks (CNNs)

CNNs exploit local spatial correlations via convolutional layers applying learnable filters over local receptive fields, combined with pooling layers to downsample, enabling hierarchical feature extraction.

Key architectural features:  
- **Local connectivity:** neurons connected only to local patches.  
- **Shared weights (filters):** same filter applied across spatial locations, enforcing translational equivariance.  
- **Pooling:** max or average pooling for downsampling and local invariance.  
- **Nonlinear activations:** typically ReLU.

Historical development traces to neuroscience studies (Hubel and Wiesel), Fukushima’s Neocognitron, and advances in backpropagation for training. GPU implementations enabled large-scale CNNs like LeNet, AlexNet, GoogLeNet.

Used extensively in image recognition, video analysis, NLP (semantic parsing), anomaly detection, drug discovery, and game playing. Regularization methods (dropout, data augmentation) improve generalization.

---

## Recurrent Neural Networks (RNNs)

RNNs process sequential data by maintaining hidden states that encode past information. Standard RNNs suffer from vanishing/exploding gradients; advanced variants include:

- **LSTM:** memory cells with gates managing information flow, enabling learning long-range dependencies.  
- **GRU:** simplified gated units similar to LSTM with comparable performance.  
- **Bidirectional RNNs:** process sequences forward and backward for richer context.  

Configurations include stacked layers, encoder-decoder (seq2seq) models, and specialized variants (PixelRNN).

Training uses backpropagation through time (BPTT), teacher forcing, and global optimization methods. Applications are widespread in language modeling, speech recognition, robotics, and time series prediction.

---

## Hopfield Networks

Hopfield networks are recurrent neural networks functioning as content-addressable associative memories characterized by:  
- Binary or continuous neuron states updated asynchronously or synchronously.  
- Symmetric weight matrices learned via local, incremental rules (Hebbian or Storkey learning).  
- An **energy function** that monotonically decreases during state updates, guaranteeing convergence to stable attractor states representing stored patterns.

Limitations include low storage capacity (~0.14× number of neurons), presence of spurious states, and retrieval errors.

**Modern Hopfield networks (Dense Associative Memories)** enhance capacity via higher-order interactions and sharper energy landscapes, achieving super-linear or exponential storage capacity.

Hopfield networks model aspects of human memory and have applications in optimization and associative learning.

---

# End of Summary