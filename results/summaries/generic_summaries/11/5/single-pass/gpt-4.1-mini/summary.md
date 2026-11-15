# Radial Basis Function (RBF) Network
RBF networks consist of an input layer, a hidden layer with radial basis function activation (often Gaussian), and a linear output layer. The output is a weighted sum of radial basis functions centered at vectors in input space, typically using Euclidean or Mahalanobis distance. They are universal function approximators on compact sets. Normalized versions weight the basis functions by their sum, theoretically justified via stochastic kernel approximations. Extensions include local linear models to improve approximation.

Training involves two main steps: unsupervised selection of centers (via sampling or clustering) and supervised linear weight fitting (often using least squares or gradient descent). Additional backpropagation can fine-tune parameters. RBFs are applied to function interpolation, approximation (demonstrated with chaotic time series such as the logistic map), time series prediction, and control. Normalization generally improves accuracy, especially in higher dimensions.

# Autoencoder
An autoencoder is a neural network architecture composed of an encoder mapping input data to a lower-dimensional latent space and a decoder reconstructing the input from this latent code. It is trained to minimize a reconstruction loss, typically the mean squared error, over a dataset. Undercomplete autoencoders compress data and learn meaningful latent representations, while overcomplete autoencoders may learn identity mappings but can still capture useful features.

Variations include:
- Variational Autoencoders (VAEs), which frame the model probabilistically, optimizing a lower bound on data likelihood via amortized variational inference.
- Sparse Autoencoders enforcing sparsity in latent representations through hard thresholding (k-sparse) or sparsity regularization.
- Denoising Autoencoders trained to reconstruct clean inputs from corrupted data.
- Contractive Autoencoders adding penalties on encoder Jacobians to enforce robustness to small input changes.
- Minimum Description Length Autoencoders combining code length and reconstruction error.
- Concrete Autoencoders designed for discrete feature selection via relaxations of categorical sampling.

Deeper autoencoders provide better compression and generalization, with pretraining and joint training strategies enhancing performance. Applications include dimensionality reduction, information retrieval, anomaly detection, medical imaging, drug discovery, and machine translation.

# Recurrent Neural Network (RNN)
RNNs process sequential data by maintaining a hidden state that captures past inputs. Historically motivated by neuroscience and statistical mechanics, modern RNNs are dominated by architectures like Long Short-Term Memory (LSTM) networks and Gated Recurrent Units (GRU), designed to address vanishing gradient problems. Variants include bidirectional RNNs (processing sequences forward and backward), stacked RNNs, and specialized forms such as Hopfield networks and Echo State Networks.

RNNs are widely applied in machine translation, speech recognition and synthesis, time series prediction, handwriting recognition, biological sequence analysis, and control. Training typically employs backpropagation through time, with techniques like teacher forcing improving convergence. Advanced training methods include global optimization (e.g., genetic algorithms). Other specialized RNN forms include recursive networks, neural Turing machines, and independently recurrent networks addressing various specific challenges in sequence modeling.

# Graph Neural Network (GNN)
GNNs learn on graph-structured data by iteratively updating node representations via permutation-equivariant message passing layers, which aggregate information from neighbors. They incorporate local pooling to coarsen graphs and global pooling to generate fixed-size graph embeddings. Common architectures include Graph Convolutional Networks (GCN), Graph Attention Networks (GAT), and Gated Graph Sequence Neural Networks (GGS-NN), differing primarily in message functions and aggregation.

Challenges such as oversmoothing and oversquashing arise when stacking many layers; mitigations include skip connections and modified aggregation. GNNs sometimes face difficulties in heterophilic graphs (low homophily), which has spurred research in generalized graph learning.

Applications span protein folding (e.g., AlphaFold), social network analysis, combinatorial optimization, cybersecurity, water distribution forecasting, computer vision (image as graph patches), and natural language processing.

# Hopfield Network
Hopfield networks are fully connected recurrent neural networks functioning as associative memory models. Units are binary or continuous, with symmetric weights ensuring energy minimization dynamics leading to convergence to stable memory states. Training typically uses Hebbian or Storkey learning rules, storing memory patterns as attractors.

Limitations include spurious patterns and limited memory capacity (~0.14 times the number of neurons). Modern Hopfield networks (Dense Associative Memories) extend classical models by incorporating higher-order interactions, achieving super-linear or exponential memory capacity with continuous states.

Hopfield networks provide models of human memory and have been applied to optimization problems like the traveling salesman. Recent advances generalize them mathematically to layered architectures with guaranteed convergence.

# Variational Autoencoder (VAE)
VAEs are probabilistic generative models combining an encoder mapping inputs to distributions over latent variables and a decoder generating data from latent samples. They optimize the evidence lower bound (ELBO), balancing reconstruction accuracy and KL divergence between approximate posteriors and priors. The reparameterization trick enables efficient gradient-based training.

VAEs extend autoencoders with stochastic latent variables and probabilistic interpretation, allowing sampling and generation. Variants include β-VAE (with disentanglement), conditional VAE, and hybrids with GANs. Statistical distances different from KL divergence have been used to define alternative loss functions.

# Convolutional Neural Network (CNN)
CNNs are deep neural networks specialized for processing grid-like data such as images. They consist of convolutional layers with locally connected filters that share weights spatially, producing feature maps capturing hierarchical spatial features. Pooling layers reduce spatial dimensions and introduce translation invariance. Fully connected layers near the output perform classification or regression.

CNN kernels slide over inputs performing dot-product operations followed by nonlinear activations (commonly ReLU). Architectural advances include depthwise separable convolutions, dilated convolutions, max pooling, and dropout regularization. CNNs rely on receptive fields and weight sharing, enabling efficient parameter use compared to fully connected networks.

Historically inspired by neuroscience and early visual cortex models, CNNs revolutionized computer vision with models like LeNet, AlexNet, and ResNet. Applications include image recognition, video analysis, speech, drug discovery, and games. Training progresses with large datasets, data augmentation, and fine-tuning.

# Class Activation Mapping (CAM)
CAM and its variants (e.g., Grad-CAM, Grad-CAM++, Score-CAM, LayerCAM) provide visual explanations for CNN decisions by highlighting image regions influencing class predictions. Original CAM requires architectures with a global average pooling (GAP) layer before classification; Grad-CAM generalizes this using gradients of class scores with respect to convolutional features, enabling model-agnostic and layer-wise explanations.

Refinements like Guided Grad-CAM combine high-resolution guided backpropagation with Grad-CAM maps. Score-CAM is gradient-free, using model confidence scores for weighting. LayerCAM combines information from multiple layers for higher resolution. These methods address interpretability and localization challenges in CNNs.

# Transformer (Deep Learning)
Transformers are sequence models using self-attention mechanisms to capture dependencies regardless of distance, replacing recurrence in RNNs. They consist of encoder and decoder stacks with alternating multi-head attention and feedforward layers, employing positional encodings to retain sequence information.

Tokenization breaks inputs into tokens mapped to embeddings. Attention heads compute weighted sums of values, guided by queries and keys, allowing parallelizable long-distance interactions. Masked attention enables autoregressive generation.

Pretraining with self-supervised tasks followed by fine-tuning on specific tasks underlies the success of large models (BERT, GPT). Variants include encoder-only, decoder-only, and encoder-decoder architectures.

Improvements include alternative normalization, positional encodings (RoPE, ALiBi), efficient attention algorithms (FlashAttention), and scaling techniques. Transformers have transformed NLP and find applications in protein folding, vision, speech, and multimodal learning.

# Vision Transformer (ViT)
ViTs adapt Transformers to computer vision by splitting images into patches treated as tokens. An encoder-only transformer processes patch embeddings with positional information. ViTs achieve competitive or superior image classification performance compared to CNNs, especially with large datasets and pretraining.

Advances include masked autoencoders for self-supervised learning, hierarchical designs like Swin Transformer (shifted windows), and hybrid architectures combining convolution and attention. ViTs require larger datasets and careful optimization but capture long-range dependencies effectively.

Applications cover image classification, detection, segmentation, video understanding, anomaly detection, and image generation, with extensions like DINO focusing on self-distillation and robust feature learning.

# Generative Adversarial Network (GAN)
GANs are generative models consisting of a generator creating data samples and a discriminator distinguishing real from generated samples in a zero-sum game. Training alternates optimizing generator and discriminator so that generator learns to mimic the real data distribution.

Mathematically, GANs minimize Jensen–Shannon divergence between generated and real distributions. Training is challenging due to instability, mode collapse (missing modes in the data), vanishing gradients, and difficulty in convergence. Evaluation uses metrics like Inception score and Fréchet Inception Distance.

Variants include:
- Conditional GANs (generate conditioned on labels),
- Deep Convolutional GANs (DCGAN),
- Wasserstein GANs (WGAN) using Earth Mover's distance for improved gradients,
- Adversarial Autoencoders,
- InfoGAN (encouraging disentangled representations),
- Bidirectional GANs (BiGAN),
- CycleGAN (image-to-image translation without paired data),
- StyleGAN series with progressive growth and style control.

GANs are used in image generation, image-to-image translation, art, fashion, medical imaging, astronomy, and malicious deepfakes generation.

# Wasserstein GAN (WGAN)
WGAN replaces the GAN discriminator with a critic constrained to be Lipschitz continuous. It optimizes the Wasserstein-1 distance (Earth Mover’s distance) instead of Jensen–Shannon divergence. This provides smoother gradients and alleviates problems like mode collapse and vanishing gradients.

Training enforces Lipschitz constraints via methods such as weight clipping, spectral normalization, or gradient penalties. WGAN’s critic better informs the generator, leading to more stable training and improved generator performance.

Theoretical foundations rely on Kantorovich-Rubenstein duality, connecting the Wasserstein distance to optimization over Lipschitz functions. WGANs are widely adopted in high-resolution image generation and other generative modeling tasks.