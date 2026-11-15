# Vision Transformer (ViT) and Related Architectures
## History and Evolution
Transformers originated in Natural Language Processing (NLP) with the 2017 paper "Attention Is All You Need." Initially, applying Transformer ideas to computer vision involved replacing CNN kernels with self-attention, achieving superior performance, though not strictly being a ViT. The true ViT emerged in 2020 by adapting an encoder-only Transformer, surpassing CNN dominance in image classification. Subsequent developments included the Masked Autoencoder (2022) for unsupervised training, leading to cross-fertilization between ViT and CNN concepts. Key variants introduced in 2021 aimed for efficiency or accuracy, such as the Swin Transformer, which used convolution-like sliding windows.

## Architecture and Variants
The original ViT functions as a BERT-like, encoder-only Transformer. It processes an input image ($\mathbb{R}^{H\times W\times C}$) by splitting it into square patches, linearly embedding them, adding positional encodings, and passing the sequence through Transformer encoders. The attention mechanism iteratively refines semantic relationships between patches.

Various architectural improvements exist:
*   **Pooling:** Alternatives to the original dummy `<CLS>` token include Global Average Pooling (GAP) and Multihead Attention Pooling (MAP), with GAP and MAP generally outperforming BERT-like pooling.
*   **Masked Autoencoder (MAE):** This model uses two end-to-end ViTs (encoder and decoder) trained to reconstruct masked image patches, leading to derivatives like Medically Supervised MAE and SiamMAE for video.
*   **DINO:** A self-supervised method utilizing teacher-student distillation, preventing collapse via "sharpening" and "centering." DINOv3 introduced image-text alignment capabilities.
*   **Swin Transformer:** Improves efficiency by using "shifted window-based" attention and a merge layer analogous to pooling.
*   **TimeSformer:** Designed for video understanding, it employs factorized self-attention across space and time dimensions.

## Comparison with CNNs
ViT leverages self-attention to capture long-range dependencies, contrasting with CNNs, which rely on local connectivity. ViT is more sensitive to hyperparameters and requires more data, often benefiting from preprocessing with small convolutional filters to improve stability.

## Applications
ViTs are applied across numerous tasks, including image classification, object detection, and image synthesis (as backbones for GANs and Diffusion models). DINO is particularly noted for learning useful representations for biological data clustering.

# Generative Models (GANs, VAEs, etc.)
## Generative Adversarial Networks (GANs)
A GAN is a zero-sum game between a generator ($\mu_G$) and a discriminator ($D$), aiming for $\mu_G \approx \mu_{\text{ref}}$. The original GAN objective involves the Jensen-Shannon divergence ($D_{JS}$).

**Wasserstein GAN (WGAN):** WGAN addresses the gradient instability of the original GAN by using the Wasserstein metric, which is calculated via the Kantorovich-Rubenstein duality. This requires the discriminator (now a critic) to have a bounded Lipschitz norm, enforced via weight clipping, spectral normalization, or gradient penalties.

**Training Issues:** Standard GANs face challenges like unstable convergence, mode collapse (where the generator misses data modes), and vanishing gradients. Solutions include the Two Time-Scale Update Rule (TTUR) and WGAN.

**Variants:**
*   **Conditional GAN (cGAN):** Allows generation conditioned on auxiliary information, such as class labels.
*   **Architectural Variants:** Include DCGAN (pure convolutional), SAGAN (with self-attention), and TransGAN (pure Transformer).
*   **Objective Variants:** Include non-saturating loss, Hinge loss, and Least Squares GAN.

## Variational Autoencoders (VAEs)
VAEs are generative models that aim to maximize the data likelihood by optimizing a lower bound called the Evidence Lower Bound (ELBO). They use an encoder ($E_\phi$) to map data ($x$) to a latent space ($z$), and a decoder ($D_\theta$) to reconstruct $x'$ from $z$.

**Mathematical Basis:** The ELBO loss is equivalent to minimizing reconstruction error while minimizing the Kullback–Leibler (KL) divergence between the approximate posterior $q_\phi(z|x)$ and the prior $p_\theta(z)$.
**Optimization:** The reparameterization trick allows for stochastic backpropagation through the latent variable sampling process.
**Variations:**
*   **$\beta$-VAE:** Uses a weighted KL divergence ($\beta>1$) to enforce disentangled latent representations.
*   **Denoising Autoencoder (DAE):** Trained by intentionally corrupting inputs to learn robust representations.
*   **Contractive Autoencoder (CAE):** Adds a regularization loss on the Jacobian of the encoder to ensure small input variations lead to small code variations.
*   **Minimum Description Length AE (MDL-AE):** Minimizes a combined cost of code length and reconstruction error.

## Other Generative & Representation Models
*   **Hopfield Network:** An associative memory model using energy functions (Lyapunov functions) to converge to local energy minima, enabling content-addressable memory. Modern versions (Dense Associative Memories) increase storage capacity beyond the linear scaling of classical models.
*   **Radial Basis Function (RBF) Network:** A three-layer network using a radial basis function (often Gaussian) in the hidden layer. They are universal approximators on compact sets, and can be trained via two steps: clustering for centers, followed by linear fitting for weights. Normalization improves accuracy.
*   **Autoencoder Applications:** Used for dimensionality reduction (related to PCA), information retrieval (semantic hashing), and anomaly detection (reconstruction error serves as the anomaly score).

# Transformer and CNN Concepts
## Convolutional Neural Networks (CNNs)
CNNs mitigate the "curse of dimensionality" of MLPs by exploiting spatial locality and parameter sharing.

**Core Features:**
1.  **Local Connectivity:** Neurons connect only to a restricted region (receptive field) of the previous layer. Stacking layers increases the effective receptive field.
2.  **Shared Weights (Filters):** The same filter is applied across the entire input, granting translational equivariance.
3.  **Pooling:** Downsamples feature maps (e.g., max or average pooling), reducing computation and granting local translational invariance.

**Layer Types:**
*   **Convolutional Layer:** The core operation, performing a dot product between the kernel and the input patch to generate a feature map.
*   **Pooling Layers:** Used for dimension reduction.
*   **Fully Connected Layers:** Used in the final stages for classification.
*   **Depthwise Separable Convolution:** An efficient alternative combining depthwise and pointwise convolutions.

**History:** CNN concepts trace back to Fukushima's Neocognitron (1980) and were made trainable via backpropagation by LeCun et al. (1989), culminating in breakthroughs on ImageNet in 2012 (AlexNet).

## Comparison: ViT vs. CNNs
CNNs excel due to their inherent inductive biases—local connectivity and parameter sharing—making them data-efficient and robust to local perturbations. ViTs excel at capturing global relationships via self-attention but require significantly more data to train effectively.

# Sequence and Graph Models
## Recurrent Neural Networks (RNNs)
RNNs process data sequentially, with the hidden state ($h_t$) serving as a memory of past inputs.

**Modern Architectures:**
*   **LSTM (Long Short-Term Memory):** Solves the vanishing gradient problem using internal gating mechanisms (e.g., forget gates), allowing it to learn long-term dependencies.
*   **GRU (Gated Recurrent Unit):** A simpler, more parameter-efficient variant of LSTM.
*   **BiRNN:** Processes input both forward and backward, providing context from both preceding and succeeding tokens.
*   **Encoder-Decoder:** Used for sequence transduction (e.g., machine translation), where an encoder compresses input into a sequence of hidden states that the decoder uses to generate the output.

**Training:** Training typically involves **Backpropagation Through Time (BPTT)**. **Teacher Forcing** is a common technique where the decoder is fed the true target sequence during training to stabilize learning.

## Graph Neural Networks (GNNs)
GNNs are designed to operate on graph-structured data.

**Fundamental Components:**
1.  **Permutation Equivariant Layer (Message Passing):** Nodes update their representations by aggregating messages from neighbors ($\mathbf{h}_u = \phi (\mathbf{x}_u, \bigoplus_{v \in N_u} \psi(\mathbf{x}_u, \mathbf{x}_v, \mathbf{e}_{uv}))$). Stacking layers increases the receptive field.
2.  **Local/Global Pooling:** Used to coarsen the graph (downsampling).

**Types:**
*   **Graph Convolutional Network (GCN):** Generalizes CNNs to graphs, using a first-order approximation of a localized spectral filter.
*   **Graph Attention Network (GAT):** Combines GNNs with attention, where learnable coefficients ($\alpha_{uv}$) determine the importance of neighbors.
*   **Gated Graph Sequence Neural Network (GGS-NN):** Extends GNNs to output sequences by integrating message passing into a GRU cell.

**Challenges:** GNNs are limited by the Weisfeiler–Leman Graph Isomorphism Test. The **heterophily** problem, where connected nodes have dissimilar labels, challenges standard GNN performance.

# Miscellaneous Topics
*   **Class Activation Mapping (CAM/Grad-CAM):** Methods used for model interpretability in CNNs. CAM requires a Global Average Pooling layer, while Grad-CAM generalizes this by using gradients to weight feature maps, making it model-agnostic. Fine-tuned versions like Guided Grad-CAM and Score-CAM enhance localization.
*   **Hyperparameters:** Key CNN hyperparameters include padding, stride, and number of filters, which control the receptive field and spatial output size. Regularization methods like Dropout and L2 regularization are used to prevent overfitting.
*   **Autoencoders (General):** Function as compression tools ($X \to Z \to X'$). They are foundational for anomaly detection (high reconstruction error flags anomalies) and dimensionality reduction, with VAEs adding a probabilistic, smooth latent space.