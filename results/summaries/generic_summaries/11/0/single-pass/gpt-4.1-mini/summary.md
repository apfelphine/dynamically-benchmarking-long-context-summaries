# Summary

---

## Generative Adversarial Networks (GANs)

GANs consist of two competing neural networks—a generator and a discriminator—engaged in a zero-sum game where the generator aims to mimic a target data distribution, and the discriminator attempts to distinguish between real and generated samples. Mathematically, this is modeled over a probability space with the generator selecting a distribution and the discriminator choosing a function to maximize the likelihood of correctly classifying inputs.

Practically, the generator maps random latent vectors to candidate data samples, while the discriminator assesses their authenticity, both trained via backpropagation. GANs are implicit generative models providing advantages over autoregressive and VAEs but face challenges like unstable convergence and mode collapse. Various architectures and objective functions have been developed, including conditional GANs, Wasserstein GANs (which leverage Wasserstein distance for improved gradients), and multi-player variants such as InfoGAN and CycleGAN for tasks like domain translation without paired data.

Training issues like vanishing gradients and mode collapse have inspired techniques such as two-time scale updates and invertible data augmentation. Evaluation metrics include Inception Score and Fréchet Inception Distance. GANs find applications across science (e.g., astronomy, medical imaging), art, fashion, and malicious uses (deepfakes). The original GAN formulation underpins variants like Wasserstein GANs that provide better theoretical properties and training stability.

---

## Transformer (Deep Learning)

Transformers are neural architectures designed primarily for sequence modeling without recurrent structures, relying instead on self-attention mechanisms to capture dependencies in data. They evolved from RNN and LSTM predecessors, overcoming limitations like vanishing gradients and enabling parallel computation over sequences. The original "Attention Is All You Need" paper introduced multi-head attention, positional encoding (sine and cosine functions for token positions), and encoder-decoder stacks.

Transformer training often uses learning rate warmup, with typical workflows involving large-scale self-supervised pretraining and supervised fine-tuning on various NLP tasks (e.g., language modeling, translation). Architecturally, transformers include tokenizers, embedding layers, multiple attention and feedforward layers, and un-embedding layers. Variants such as encoder-only (e.g., BERT), decoder-only (e.g., GPT), and prefixLM models serve distinct purposes.

Enhancements include alternative activations (GELU, SwiGLU), normalizations (RMSNorm), and positional embeddings (RoPE, ALiBi). Efficient implementations optimize computations using KV caching and FlashAttention. Large-scale transformers are widely used in language, vision (Vision Transformer - ViT), speech, and multimodal tasks. Applications span machine translation, code generation, protein folding, and complex video understanding.

---

## Wasserstein GAN

Wasserstein GANs modify classical GANs by replacing the Jensen-Shannon divergence with the Wasserstein (Earth-Mover) distance as the measure of distribution difference. This results in a more meaningful training signal and mitigates issues like mode collapse and vanishing gradients. The discriminator (critic) is required to be a Lipschitz function, enforced by techniques such as weight clipping, spectral normalization, or gradient penalty.

The Wasserstein GAN objective involves maximizing the expected critic output difference between generated and real data. This approach provides better gradient information to the generator, improving training stability, especially when data distributions lie on low-dimensional manifolds.

---

## Convolutional Neural Networks (CNNs)

CNNs are neural networks designed to process data with grid-like topology, such as images, by exploiting spatial locality via convolutional layers with shared weights and limited receptive fields. Key components include convolutional layers producing feature maps, pooling layers for downsampling (max and average pooling), and fully connected layers for classification.

The architecture reduces parameters compared to fully connected networks by leveraging localized connectivity and parameter sharing, offering translation equivariance and robustness to local perturbations. Variants include depthwise separable convolutions and dilated convolutions.

Historically inspired by neuroscience (Hubel and Wiesel’s visual cortex studies), CNNs gained prominence through architectures like LeNet and AlexNet, aided by GPU accelerations. Applications cover image and video recognition, natural language processing, anomaly detection, and drug discovery.

---

## Vision Transformer (ViT)

Vision Transformers adapt the transformer architecture to image data by dividing images into patches and processing them as token sequences. ViTs leverage multi-head self-attention to learn global contextual relationships among patches, providing an alternative to traditional CNNs.

ViT variants include masked autoencoders for self-supervised learning, Swin Transformers introducing windowed attention for efficiency, and enhancements like multihead attention pooling. Despite requiring larger datasets due to fewer inductive biases, ViTs achieve state-of-the-art results in classification, object detection, and image generation tasks.

---

## Hopfield Network

Hopfield networks are recurrent neural networks functioning as associative memory systems. Units have binary or continuous activations and symmetric connection weights learned via Hebbian or Storkey rules. The network dynamics minimize an energy function, converging to stable attractor states representing stored memory patterns.

Extensions, termed modern Hopfield networks or Dense Associative Memories, use higher-order interactions to exponentially increase storage capacity. These networks relate closely to optimization problems, pattern retrieval, and models of human memory and cognition.

---

## Class Activation Mapping (CAM)

CAM and its variants (Grad-CAM, Grad-CAM++, Score-CAM, Layer-CAM) are techniques for visualizing regions in input images that contribute most to a CNN’s predictions. CAM uses global average pooling and weighted sums of feature maps, while Grad-CAM generalizes this using gradients, allowing applicability to any differentiable CNN without architectural changes.

Further refinements improve localization accuracy, resolution, and interpretability, enabling insights into model decision-making and aiding trust in model predictions.

---

## Recurrent Neural Networks (RNNs)

RNNs are neural architectures with loops enabling information persistence through sequences. Notable variants include Elman and Jordan networks, Long Short-Term Memory (LSTM), and Gated Recurrent Units (GRU), addressing gradient vanishing and capturing long-term dependencies.

Configurations comprise stacked, bidirectional, and encoder-decoder setups. Training uses Backpropagation Through Time and techniques like teacher forcing. Applications include language modeling, speech recognition, robotics, and sequence classification.

---

## Variational Autoencoder (VAE)

VAEs are probabilistic generative models combining encoder-decoder neural networks with variational Bayesian inference. They map input data to probabilistic latent spaces and reconstruct data while optimizing an Evidence Lower Bound (ELBO), balancing reconstruction loss and latent distribution regularization (measured by KL divergence).

Reparameterization tricks enable gradient-based training despite sampling. Variants include conditional VAEs, β-VAEs for disentangled representations, and others using alternative statistical distances. Applications span image generation, anomaly detection, and molecular design.

---

## Radial Basis Function (RBF) Network

RBF networks are feedforward models with an input layer, a hidden layer employing radial basis (usually Gaussian) activations, and a linear output layer. They perform function approximation and interpolation based on distances from centers in the input space.

Centers can be selected via clustering or sampling; weights are learned by pseudoinverse or gradient descent methods. RBFs are universal approximators, effective in time series prediction and control of chaotic systems.

---

## Graph Neural Networks (GNNs)

GNNs process data represented as graphs, learning node and graph-level representations via permutation-equivariant message-passing layers, local pooling, and global pooling operations. Variants include Graph Convolutional Networks (GCNs) and Graph Attention Networks (GATs).

They are powerful in modeling relational data, with applications in protein folding (AlphaFold), social network analysis, combinatorial optimization, cybersecurity, and computer vision. GNNs have limitations based on graph isomorphism tests but ongoing research aims to improve their expressiveness.

---

## Autoencoder

Autoencoders learn compact representations by encoding input data into a latent space and decoding it back to reconstruct the input. Training minimizes the reconstruction error, often using undercomplete architectures to force meaningful compression.

Variants include sparse autoencoders promoting sparsity, denoising autoencoders enhancing robustness, contractive autoencoders ensuring small Jacobian norms, and variational autoencoders incorporating probabilistic modeling. Applications cover dimensionality reduction, anomaly detection, image processing, drug discovery, and machine translation.

---

# End of Summary