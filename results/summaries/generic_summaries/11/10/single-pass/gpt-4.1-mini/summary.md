# Vision Transformer (ViT)

Vision Transformers adapt the Transformer architecture, originally developed for natural language processing, to computer vision tasks. Unlike CNNs that use convolutional kernels, ViTs process images by dividing them into patches, embedding each patch, adding positional encodings, and feeding the resulting vectors into a Transformer encoder. The basic ViT architecture achieves state-of-the-art image classification by modeling relationships between image patches through self-attention.

Variants of ViT improve efficiency and adapt to specific domains, such as the Swin Transformer, which utilizes shifted windows for localized attention mimicking CNN receptive fields. Other developments include Masked Autoencoders (MAE) for self-supervised learning by reconstructing masked patches, and DINO for self-distillation without labels. ViT models have become foundational components in diverse tasks including image classification, object detection, video understanding, and image generation.

# Wasserstein GAN (WGAN)

Wasserstein GANs modify the original GAN training by replacing the Jensen–Shannon divergence with the Wasserstein (Earth Mover’s) distance as the measure of distribution similarity. This change leads to a more stable training process with better gradient feedback for the generator, alleviating issues like vanishing gradients and mode collapse common in standard GANs.

The discriminator (critic) in WGANs is constrained to be Lipschitz-continuous, enforced via techniques like weight clipping, spectral normalization, or gradient penalties. By optimizing the Wasserstein distance, WGANs push the generator to approximate the real data distribution more effectively, improving training stability and output quality.

# Class Activation Mapping (CAM)

Class Activation Mapping techniques visually explain CNN predictions by highlighting image regions most influential for class decisions. The original CAM method requires specific CNN architectures with global average pooling layers and uses weighted feature maps to create class-specific heatmaps. Grad-CAM generalizes this approach by leveraging gradients flowing into any convolutional layer, enabling application to a broad range of CNNs without architectural modifications.

Refinements such as Guided Grad-CAM, Grad-CAM++, Score-CAM, and Layer-CAM enhance resolution, localization accuracy, and interpretability, often combining gradient and activation information or replacing gradients with model confidence scores for improved explanations.

# Generative Adversarial Network (GAN)

GANs are frameworks composed of two neural networks—the generator creates synthetic data, and the discriminator evaluates its authenticity relative to real data. The networks engage in a zero-sum game, where the generator aims to fool the discriminator, and the discriminator tries to distinguish real from fake data. Training proceeds via alternating optimization of both players’ objectives.

Variants include Conditional GANs that condition generation on labels, adversarial autoencoders, InfoGANs maximizing mutual information for interpretable latent codes, and bidirectional GANs combining encoding and decoding. GANs can suffer from training challenges like mode collapse and vanishing gradients, often addressed through architectural and loss function improvements such as Wasserstein GAN. GANs have widespread applications across image synthesis, medical imaging, scientific modeling, and more.

# Variational Autoencoder (VAE)

VAEs are probabilistic generative models combining neural networks with variational inference, aiming to model complex data distributions via latent variables. They consist of an encoder network approximating the posterior distribution over latent variables given data and a decoder network generating data samples from latent codes. Training maximizes the evidence lower bound (ELBO), balancing data reconstruction accuracy and regularization enforcing latent distributions close to a prior.

The reparameterization trick enables efficient gradient-based optimization by allowing derivatives to propagate through stochastic sampling. Variants of VAEs implement different priors, noise models, and statistical distances to improve performance and interpretability.

# Transformer (Deep Learning)

Transformers are deep learning models primarily built on self-attention mechanisms that allow parallel processing of sequences, overcoming limitations of recurrent networks. Originally designed for natural language processing tasks like machine translation, the canonical Transformer architecture uses encoder and decoder stacks composed of multi-head self-attention and feedforward layers, combined with positional encodings.

Transformers have fueled advances in language modeling (e.g., BERT, GPT), and have been successfully extended to modalities such as vision (Vision Transformers), speech, proteins, and multimodal data. Variations include encoder-only, decoder-only, and encoder-decoder models. Architectural and training innovations address efficiency, scale, and long-context processing, including methods like FlashAttention, KV caching, and sub-quadratic attention.

# Graph Neural Network (GNN)

GNNs are neural networks designed to operate on graph-structured data. They typically involve message-passing layers where nodes update representations by aggregating information from neighbors, local and global pooling layers for graph coarsening and readout, ensuring permutation equivariance and invariance.

Variants such as Graph Convolutional Networks (GCNs) use spectral filters, and Graph Attention Networks (GAT) integrate attention mechanisms to weigh neighbor contributions. GNNs can capture complex relational structures but have expressiveness limitations tied to graph isomorphism tests. They are widely applied in protein folding, social network analysis, combinatorial optimization, cybersecurity, and computer vision.

# Convolutional Neural Network (CNN)

CNNs are specialized neural networks for grid-like data, such as images, exploiting local connectivity, shared weights (filters), and spatial hierarchies to learn translation-equivariant features. Core components include convolutional layers, pooling layers for spatial downsampling, fully connected layers for classification, and nonlinear activation functions like ReLU.

CNNs gained prominence due to their efficiency in handling high-dimensional inputs with fewer parameters than fully connected networks. They evolved from neuroscience-inspired models and have been accelerated by GPU implementations. CNNs excel in image recognition, video analysis, natural language processing, anomaly detection, and drug discovery, often combined with other architectures for enhanced performance.

# Autoencoder

Autoencoders are neural networks trained to reconstruct input data through a bottleneck latent representation, effectively learning compressed data encodings. They consist of an encoder mapping inputs to a lower-dimensional latent space, and a decoder reconstructing the input from the latent code. The training minimizes reconstruction error.

Variations include sparse autoencoders enforcing sparse latent representations, denoising autoencoders trained to restore corrupted inputs, contractive autoencoders penalizing sensitivity to input variations, and variational autoencoders that introduce probabilistic latent variables. Autoencoders are applied to dimensionality reduction, anomaly detection, image processing, and generative modeling.

# Hopfield Network

Hopfield networks are recurrent neural networks functioning as content-addressable associative memories. They store patterns as energy minima in a symmetric weight matrix learned typically via Hebbian rules. Network states update asynchronously or synchronously, converging to stored patterns or spurious attractors.

Classical Hopfield networks have limited storage capacity scaling linearly with neuron count; modern Hopfield networks or Dense Associative Memories use higher-order interactions and non-linear energy functions to achieve exponential capacity. They have applications in optimization, modeling human memory, and serve as theoretical foundations for recurrent dynamics in neural computation.

# Radial Basis Function (RBF) Network

RBF networks are feedforward architectures with an input layer, a hidden layer of radial basis neurons using localized, radially symmetric functions (often Gaussians), and a linear output layer. They perform function approximation by representing outputs as weighted sums of radial basis functions centered on prototype points.

Training involves selecting centers (via sampling or clustering) and optimizing linear weights, often by least squares or gradient descent. Normalized RBF networks improve approximation by normalizing activations. RBF networks are universal approximators and have been applied to interpolation, time series prediction, and control problems.

# Recurrent Neural Network (RNN)

RNNs are neural networks with cycles allowing them to maintain and update hidden state representations over time sequences, enabling modeling of temporal dependencies. Variants include simple RNNs (Elman, Jordan networks), Long Short-Term Memory (LSTM) networks designed to overcome vanishing gradients, Gated Recurrent Units (GRU), bidirectional RNNs, and hierarchical or recursive RNNs.

RNN training uses backpropagation through time, with techniques like teacher forcing to improve convergence and stability. Applications span language modeling, machine translation, speech recognition, and time series prediction. Extensions include echo state networks, neural Turing machines, and continuous-time variants, with ongoing development targeting long-range dependencies, scalability, and interpretability.