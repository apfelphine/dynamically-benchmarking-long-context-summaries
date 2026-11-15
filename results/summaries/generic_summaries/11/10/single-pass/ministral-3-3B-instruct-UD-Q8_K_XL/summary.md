### **Vision Transformer (ViT) Overview**

#### **History**
Vision Transformers emerged from the Transformer architecture originally designed for natural language processing (NLP). In 2019, researchers adapted the Transformer to computer vision by replacing convolutional kernels with self-attention mechanisms, initially in ResNet architectures. The **Vision Transformer (ViT)** (2020) achieved state-of-the-art performance in image classification by treating images as sequences of patches, embedding them, and applying self-attention. This approach overcame the limitations of convolutional neural networks (CNNs) by capturing long-range dependencies in images.

Subsequent advancements included:
- **Masked Autoencoder (MAE, 2022)**, which extended ViT to unsupervised pretraining by masking patches and reconstructing them.
- **Variants like Swin Transformer (2021)**, which improved efficiency and robustness by using sliding windows and pyramid processes.
- **DINO (2021)**, a self-distillation method for pretraining ViTs without labels.
- **Cross-fertilization** between ViT and CNN approaches, leading to hybrid models.

#### **Architecture**
The ViT architecture is inspired by BERT, an encoder-only Transformer for NLP. An image of shape \( \mathbb{R}^{H \times W \times C} \) is divided into square patches, each converted into a vector via a linear embedding. Positional encodings are added to retain spatial information. These vectors are fed into multiple Transformer encoder layers, where self-attention mechanisms capture semantic relationships between patches.

The output is processed by a classification head (e.g., a shallow MLP) to generate class probabilities. The **<CLS> token** (a special token) aggregates information from all patches, enabling the model to predict the entire image class.

#### **Variants**
1. **Original ViT**
   - Supervised-trained to predict image labels using the <CLS> token.
   - Requires a special token for classification, limiting flexibility.

2. **Architectural Improvements**
   - **Pooling Methods**: Global Average Pooling (GAP) and Multihead Attention Pooling (MAP) replace the <CLS> token, improving generalization.
   - **Re-attention**: Enhances training deep ViTs by modifying multihead attention modules.

3. **Masked Autoencoder (MAE)**
   - Uses two ViTs: an encoder to process unmasked patches and a decoder to reconstruct masked patches.
   - Achieves high performance with high mask ratios, improving efficiency and memory usage.
   - Derivatives like **Medically Supervised MAE** and **Gray Level Co-occurrence Matrix MAE (GLCM-MAE)** address medical imaging challenges.

4. **DINO**
   - Self-distillation method where the student network mimics the teacher (exponential average of past states).
   - Uses sharpening and centering strategies to prevent collapse and improve feature transferability.
   - **DINOv2 (2024)** and **DINOv3 (2025)** further improved performance with advanced architectures and training techniques.

5. **Swin Transformer**
   - Uses shifted windows for self-attention, reducing computational cost.
   - Introduces merge layers to combine neighboring tokens, analogous to pooling in CNNs.

6. **TimeSformer**
   - Applies factorized self-attention to video data, dividing attention into space and time dimensions.

7. **ViT-VQGAN**
   - Combines ViT with vector quantization (VQ) and generative adversarial networks (GANs) for image generation.
   - Enables autoregressive image generation and text-to-image synthesis.

8. **Other Notable Models**
   - **Visual Transformer, CoAtNet, CvT, DeiT**: Variants with different design goals (efficiency, robustness, or domain-specific applications).

#### **Comparison with CNNs**
ViTs and CNNs differ in inductive biases:
- **CNNs** apply the same filters across the entire image, making them data-efficient and robust to local perturbations.
- **ViTs** use self-attention to capture long-range dependencies but require more data and are sensitive to hyperparameters and network depth.
- ViTs are more robust to input distortions like adversarial patches or permutations.

#### **Applications**
ViTs excel in diverse computer vision tasks, including:
- **Image classification, object detection, video deepfake detection, image segmentation, anomaly detection, and synthesis**.
- **Generative models** like GANs and diffusion models (e.g., DALL-E, Stable Diffusion).
- **Medical imaging** for lesion classification and disease diagnosis.
- **Weather and climate prediction** with large-scale models.
- **Autonomous driving** and **3D reconstruction**.

---

### **Wasserstein GAN (WGAN) Overview**

#### **Motivation**
The original Generative Adversarial Network (GAN) uses the Jensen–Shannon divergence to measure the distance between the generator’s distribution and the reference distribution. However, this approach suffers from:
- **Mode collapse**, where the generator fails to generalize properly.
- **Unstable training**, as the discriminator can become too good, leading to vanishing gradients.
- **High computational cost** due to the Jensen–Shannon divergence’s complexity.

The **Wasserstein GAN (WGAN)** addresses these issues by using the **Wasserstein distance**, which is more stable and computationally efficient. The Wasserstein distance measures the "work" needed to transform one distribution into another, providing a better gradient signal for the generator.

#### **Definition**
A WGAN game is defined over a probability space \( (\Omega, \mathcal{B}, \mu_{\text{ref}}) \), where:
- **Generator** aims to minimize the Wasserstein distance between its distribution \( \mu_G \) and the reference distribution \( \mu_{\text{ref}} \).
- **Discriminator (Critic)** aims to maximize the same distance.
- The discriminator’s strategy set includes functions with bounded Lipschitz norm \( \|D\|_L \leq K \).

The objective function is:
\[ L_{WGAN}(\mu_G, D) = \mathbb{E}_{x \sim \mu_G} [D(x)] - \mathbb{E}_{x \sim \mu_{\text{ref}}} [D(x)] \]

#### **Comparison with GAN**
- **Gradient Signal**: WGAN provides a better gradient for the generator compared to GAN, reducing mode collapse.
- **Training Stability**: WGAN avoids the vanishing gradient problem by using a Lipschitz-constrained critic.
- **Computational Efficiency**: The Wasserstein distance is more efficient to compute than the Jensen–Shannon divergence.

#### **Training Methods**
Training WGANs involves:
1. **Upper-Bounding the Lipschitz Norm**:
   - **Weight Clipping**: Limits the weights of the discriminator to a fixed range.
   - **Spectral Normalization**: Normalizes the weights to ensure the Lipschitz norm is bounded.
   - **Gradient Penalty**: Penalizes the discriminator for violating the Lipschitz constraint.

2. **Advantages**:
   - **No Mode Collapse**: WGANs consistently generate diverse samples without collapsing into a few modes.
   - **Better Gradient Flow**: The critic’s gradient is more stable, improving training dynamics.

#### **Applications**
WGANs are used in:
- **Generative modeling** for high-quality image synthesis.
- **Style transfer** and **image-to-image translation**.
- **Advanced generative models** like StyleGAN, which leverage WGAN principles for improved stability and quality.

---

### **Class Activation Mapping (CAM) Overview**

#### **Background**
Class activation mapping (CAM) and its variants (e.g., Grad-CAM) provide visual explanations for CNN predictions by highlighting regions in an image that contribute most to a specific class. These methods are rooted in:
- **Gradient-based visualization**: Highlighting pixels with the largest gradients for a target class.
- **Guided backpropagation**: Ensuring gradients flow through positive activations, improving interpretability.

#### **Base Versions**
1. **Class Activation Mapping (CAM)**
   - Uses **Global Average Pooling (GAP)** to aggregate feature maps into a single class score.
   - Requires modifying the CNN architecture to include GAP before the final linear classifier.
   - **Advantages**: Model-specific, interpretable by design.
   - **Drawbacks**: Limited to CNNs with GAP layers, less flexible.

2. **Gradient-Weighted Class Activation Mapping (Grad-CAM)**
   - Computes gradients of the class score with respect to feature maps, then pools them to generate a heatmap.
   - **Advantages**: Model-agnostic, works with any differentiable CNN.
   - **Drawbacks**: Focuses on discriminative regions, may miss context or coarser details.

#### **Fine-Tuned Versions**
- **Guided Grad-CAM**: Combines Grad-CAM with guided backpropagation for high-resolution details.
- **Grad-CAM++**: Uses higher-order gradients to improve localization for multiple instances.
- **Score-CAM**: Replaces gradients with model confidence scores, improving robustness.
- **LayerCAM**: Enhances Grad-CAM by using gradients from intermediate layers for finer details.

#### **Applications**
CAM and its variants are used in:
- **Explainable AI**: Visualizing CNN decisions for critical systems (e.g., self-driving cars).
- **Medical imaging**: Highlighting regions of interest in diagnostic tasks.
- **Computer vision**: Improving interpretability in object detection and classification.

---
### **Generative Adversarial Network (GAN) Overview**

#### **Definition**
A GAN consists of two competing models:
- **Generator (G)**: Creates synthetic data from a noise distribution.
- **Discriminator (D)**: Distinguishes real data from generated data.

The training process is a zero-sum game where:
- **Generator** aims to minimize the loss, making its output indistinguishable from real data.
- **Discriminator** aims to maximize the loss, improving its ability to distinguish real from fake data.

#### **Training Challenges**
1. **Unstable Convergence**: GANs often fail to converge due to mode collapse or vanishing gradients.
2. **Mode Collapse**: The generator may produce limited diversity in its outputs.
3. **Training Strategies**:
   - **Two-Time Scale Update Rule (TTUR)**: Adjusts learning rates to stabilize training.
   - **Wasserstein GAN (WGAN)**: Uses the Wasserstein distance for better gradient flow.

#### **Variants**
1. **Conditional GAN (cGAN)**: Generates data conditioned on additional input (e.g., class labels).
2. **Wasserstein GAN (WGAN)**: Uses Lipschitz-constrained critics to improve stability.
3. **InfoGAN**: Maximizes mutual information between latent variables and generated data.
4. **CycleGAN**: Transforms images between domains without paired data.
5. **BigGAN**: Scales up GANs for high-resolution image generation.

#### **Applications**
GANs are applied in:
- **Image synthesis** (e.g., StyleGAN, DeepDream).
- **Medical imaging** (synthetic data generation).
- **Art and fashion** (image inpainting, photorealistic renders).
- **Adversarial attacks** (fake media, deepfake detection).
- **Robotics and reinforcement learning** (policy optimization).

---

### **Variational Autoencoder (VAE) Overview**

#### **Architecture**
A VAE consists of:
- **Encoder**: Maps input data to a latent space using a neural network.
- **Decoder**: Reconstructs the input from the latent representation.
- **Prior and Posterior Distributions**: The encoder approximates the posterior distribution, while the decoder models the prior.

#### **Training**
VAEs are trained using the **Evidence Lower Bound (ELBO)**, which balances reconstruction error and the Kullback-Leibler (KL) divergence between the approximate and true posterior distributions. The ELBO is optimized via backpropagation.

#### **Variants**
1. **β-VAE**: Adjusts the KL divergence weight to enforce disentangled latent representations.
2. **Sparse Autoencoder (SAE)**: Encourages sparsity in the latent space to improve feature learning.
3. **Denoising Autoencoder (DAE)**: Trains the model to reconstruct noisy inputs, improving robustness.
4. **Contractive Autoencoder (CAE)**: Adds a contractive regularization term to resist input perturbations.
5. **Minimum Description Length Autoencoder (MDL-AE)**: Optimizes for compact latent representations.

#### **Applications**
VAEs are used in:
- **Dimensionality reduction** (e.g., feature learning).
- **Anomaly detection** (by reconstructing normal data).
- **Image processing** (denoising, super-resolution).
- **Drug discovery** (generating novel molecules).
- **Popularity prediction** (social media content).

---

### **Hopfield Network Overview**

#### **Structure**
A Hopfield network is a recurrent neural network with:
- **Binary threshold units**: Each neuron has two states (+1 or -1).
- **Symmetric weights**: Connections between neurons are bidirectional and symmetric.
- **Energy function**: Measures the network’s state, with local minima representing stable states.

#### **Training and Memory**
- **Hebbian Learning Rule**: Updates weights based on simultaneous activations of neurons.
- **Associative Memory**: Stores patterns as local minima in the energy landscape.
- **Spurious Patterns**: False attractors that can interfere with memory retrieval.

#### **Capacity**
The storage capacity of a Hopfield network is limited by the number of neurons and the number of stored patterns. Modern variants (e.g., **Dense Associative Memories**) improve this capacity using non-linearities and higher-order interactions.

#### **Applications**
Hopfield networks are used in:
- **Associative memory** (retrieving patterns from partial inputs).
- **Optimization** (solving combinatorial problems).
- **Neuromorphic computing** (simulating brain-like functions).

---

### **Radial Basis Function Network (RBF) Overview**

#### **Architecture**
An RBF network consists of:
- **Input Layer**: Processes input data.
- **Hidden Layer**: Uses radial basis functions (e.g., Gaussian) to map input to output.
- **Output Layer**: Linear combination of hidden layer outputs.

#### **Training**
RBF networks are trained in two steps:
1. **Determine centers and widths** of the RBF functions (e.g., via clustering).
2. **Fit a linear model** to the hidden layer outputs using least squares or gradient descent.

#### **Applications**
RBF networks are used in:
- **Function approximation** (e.g., chaotic time series).
- **Time series prediction** (e.g., logistic map).
- **Control theory** (e.g., chaotic time series control).

---

### **Recurrent Neural Network (RNN) Overview**

#### **History**
RNNs emerged from neuroscience and statistical mechanics, evolving into modern architectures like LSTM and GRU. They are designed to process sequential data by maintaining a hidden state that captures temporal dependencies.

#### **Architectures**
1. **Standard RNN**: Processes sequences by updating hidden states iteratively.
2. **Stacked RNN**: Combines multiple RNNs for deeper processing.
3. **Bidirectional RNN (BiRNN)**: Processes sequences in both directions.
4. **Encoder-Decoder RNN**: Uses two RNNs for sequence transduction (e.g., machine translation).
5. **LSTM/GRU**: Specialized RNNs with gates to mitigate vanishing gradients.

#### **Training**
- **Teacher Forcing**: Uses ground truth labels for training.
- **Backpropagation Through Time (BPTT)**: Computes gradients over time.
- **Gradient Descent**: Optimizes weights to minimize error.

#### **Applications**
RNNs are used in:
- **Natural language processing** (e.g., machine translation, text generation).
- **Speech recognition** (e.g., Google Translate).
- **Time series forecasting** (e.g., stock market prediction).
- **Robotics** (e.g., policy learning).