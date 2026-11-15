# **Summary of Deep Learning Architectures**

## **1. Autoencoders and Variants**
Autoencoders are neural networks designed to learn a compressed representation of input data by encoding it into a latent space and then decoding it back to reconstruct the original input. Mathematically, they consist of an encoder mapping a message space $\mathcal{X}$ to a code space $\mathcal{Z}$ (where $\dim(\mathcal{Z}) < \dim(\mathcal{X})$ for compression) and a decoder mapping $\mathcal{Z}$ back to $\mathcal{X}$. The quality of the autoencoder is typically measured by a reconstruction loss, such as Mean Squared Error, and training involves minimizing this loss using gradient descent.

Several variations address specific limitations or goals:
*   **Variational Autoencoders (VAE):** Unlike standard autoencoders that output fixed vectors, VAEs model the latent space as a probability distribution (e.g., Gaussian). This allows for generative modeling and sampling, optimizing an Evidence Lower Bound (ELBO) that balances reconstruction error with the divergence of the learned posterior from a prior.
*   **Denoising Autoencoders (DAE):** Trained on corrupted inputs, DAEs learn robust representations by attempting to reconstruct the original clean data, making them effective for noise reduction and feature extraction.
*   **Sparse and Contractive Autoencoders:** Sparse Autoencoders (SAE) enforce sparsity in the latent codes to improve interpretability and classification performance, often using regularization terms. Contractive Autoencoders (CAE) add regularization to make the encoder less sensitive to small input perturbations, stabilizing the latent representation.

## **2. Graph Neural Networks (GNNs)**
Graph Neural Networks (GNNs) are designed to process data structured as graphs, where nodes represent entities (e.g., atoms, people) and edges represent relationships. The core mechanism is **message passing**, where node representations are updated iteratively by aggregating information from neighboring nodes. This allows the network to capture local structure and propagate information across the graph, with the receptive field expanding with each layer.

Key architectural components include:
*   **Message Passing Layers (MPNNs):** The fundamental building block where nodes update their features based on neighbors' features and edge features.
*   **Pooling Layers:** Used for downsampling or creating fixed-size representations of the entire graph (Global Pooling) or selecting specific subgraphs (Local Pooling like Top-k or Self-attention pooling).
*   **Specialized Architectures:** Graph Convolutional Networks (GCN) approximate spectral convolutions, while Graph Attention Networks (GAT) learn attention coefficients to weigh the importance of different neighbors. GNNs are applied to protein folding (AlphaFold), social network analysis, and combinatorial optimization.

## **3. Radial Basis Function (RBF) Networks**
RBF networks are function approximators typically consisting of an input layer, a hidden layer of radial basis neurons, and a linear output layer. Each hidden neuron responds strongly only to inputs near a specific center vector $\mathbf{c}_i$, with the response decaying as the distance increases. The output is a weighted sum of these localized responses.

*   **Training:** Training is often a two-step process. First, the centers are determined unsupervised (e.g., via clustering or k-means). Second, the output weights are fitted using linear regression (e.g., Least Squares) or gradient descent.
*   **Normalization:** Normalized RBF networks divide each basis function by the sum of all basis functions, ensuring the output is a convex combination. This provides theoretical justification in probabilistic contexts and often improves generalization.
*   **Applications:** RBF networks are effective for time series prediction, chaotic system modeling (e.g., the logistic map), and control theory due to their ability to approximate complex nonlinear dynamics.

## **4. Recurrent Neural Networks (RNNs)**
RNNs are designed to process sequential data by maintaining a hidden state that acts as memory, allowing the network to capture temporal dependencies. The core function $f_\theta(x_t, h_t) \to (y_t, h_{t+1})$ takes an input $x_t$ and previous hidden state $h_t$ to produce an output and a new state.

*   **Architectures:** Standard RNNs suffer from vanishing gradients over long sequences. **Long Short-Term Memory (LSTM)** and **Gated Recurrent Units (GRU)** introduce gating mechanisms to regulate information flow, enabling the modeling of long-term dependencies. **Bidirectional RNNs** process sequences in both forward and backward directions to capture full context.
*   **Training:** RNNs are trained using **Backpropagation Through Time (BPTT)**, which unrolls the network over time steps. Advanced methods like **Teacher Forcing** assist in training autoregressive models by providing ground truth inputs during training.
*   **Applications:** RNNs and their variants (LSTM, GRU) are foundational for natural language processing (machine translation), speech recognition, and time series forecasting.

## **5. Convolutional Neural Networks (CNNs)**
CNNs are specialized neural networks designed for processing grid-like data, such as images, by exploiting spatial locality and weight sharing. They consist of convolutional layers (feature extraction), pooling layers (downsampling), and fully connected layers (classification).

*   **Mechanisms:** **Convolution** slides learnable filters across the input to detect local patterns (edges, textures). **Pooling** reduces spatial dimensions, providing translation invariance. **Weight sharing** drastically reduces the number of parameters compared to fully connected networks.
*   **Inductive Biases:** CNNs assume translation equivariance (features move with the input) and locality.
*   **Applications:** Dominant in image recognition (e.g., ImageNet), object detection, medical imaging (segmentation, diagnosis), and video analysis. Modern variants include **Deep Belief Networks** (pre-training) and **Residual Networks** (ResNet) to handle very deep architectures.

## **6. Transformers and Vision Transformers**
Transformers revolutionized deep learning by replacing recurrence and convolution with **self-attention mechanisms**, allowing models to process sequences in parallel. The architecture consists of embedding layers, encoder/decoder stacks with multi-head attention, and feed-forward networks.

*   **Attention Mechanism:** Self-attention computes relationships between all pairs of tokens simultaneously, calculating weights based on query-key-value interactions. **Multi-head attention** allows the model to focus on different representational subspaces.
*   **Vision Transformer (ViT):** Adapted from NLP, ViT splits images into patches, linearly embeds them, and processes them as a sequence. While effective, they often require more data than CNNs and are sensitive to initialization.
*   **Efficient Variants:** **Swin Transformer** uses shifted windows to reduce quadratic complexity, and **Masked Autoencoders (MAE)** use self-supervised pretraining by masking patches.

## **7. Generative Adversarial Networks (GANs)**
GANs are a class of generative models based on a minimax game between two networks: the **Generator** (trying to create realistic data) and the **Discriminator** (trying to distinguish real data from generated data).

*   **Mathematical Formulation:** The goal is to find a generator distribution $\mu_G$ that matches a reference distribution $\mu_{ref}$, such that the discriminator's error is minimized.
*   **Training Challenges:** GANs suffer from mode collapse (generating limited variety) and vanishing gradients.
*   **Variants:**
    *   **Wasserstein GAN (WGAN):** Uses the Wasserstein distance (Earth Mover's Distance) for a more stable gradient signal.
    *   **StyleGAN:** Introduces a style-based approach where latent codes control different levels of image style and texture, enabling high-fidelity generation.
    *   **CycleGAN:** Enables unpaired image-to-image translation (e.g., day to night photos) using cycle consistency loss.

## **8. Hopfield Networks and Associative Memory**
The Hopfield network is a classic recurrent neural network that acts as a content-addressable memory system. It consists of binary neurons with symmetric weights, evolving according to local update rules until reaching a stable state (attractor).

*   **Storage:** Patterns are stored by adjusting weights via the Hebbian learning rule ($w_{ij} = \sum V_i V_j$).
*   **Dynamics:** The network minimizes an energy function (Ising model), ensuring convergence to local minima.
*   **Limitations:** Classical Hopfield networks have low storage capacity (scaling linearly with neurons) and are prone to **spurious patterns** (unwanted attractors).
*   **Modern Extensions:** **Dense Associative Memories** (Modern Hopfield Networks) use higher-order interactions and continuous variables to achieve exponential memory capacity.

## **9. Class Activation Mapping (CAM)**
CAM is a technique used to interpret the predictions of Convolutional Neural Networks (CNNs) by highlighting the regions in an input image responsible for a specific class prediction.

*   **Global Average Pooling (GAP):** The original CAM method requires a GAP layer, which reduces feature maps to scalars, allowing the model's learned weights to be applied directly to the feature maps to generate a heatmap.
*   **Gradient-weighted CAM (Grad-CAM):** A more general approach that computes gradients of the target class score with respect to the feature maps. These gradients are global-averaged to produce weights, which are then applied to the feature maps.
*   **Advantages:** Grad-CAM is model-agnostic and works with any CNN architecture, providing visual explanations without retraining.