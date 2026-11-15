### **Transformer (Deep Learning)**

#### **History**
Transformers emerged as a breakthrough in sequence modeling, overcoming limitations of recurrent neural networks (RNNs) like the vanishing gradient problem. Early models, such as LSTM (1995), addressed these issues but still operated sequentially. The **Transformer** (2017) introduced self-attention mechanisms, enabling parallel processing of sequences, fundamentally altering NLP. It replaced RNNs in seq2seq tasks, notably in machine translation, and became foundational for modern AI, including large language models like GPT-3 and BERT.

#### **Attention with Seq2Seq**
Before transformers, seq2seq models (2014) used LSTMs or GRUs, which processed sequences sequentially, limiting efficiency. Attention mechanisms (RNN Search, 2016) addressed this by allowing models to focus on relevant parts of input sequences during decoding. Transformers adopted self-attention, enabling parallel processing and superior performance in tasks like machine translation, surpassing statistical methods.

#### **Parallelizing Attention**
Transformers’ self-attention mechanism, though powerful, required quadratic computation time relative to sequence length. Early attempts to parallelize attention (e.g., decomposable attention) improved efficiency but were limited. The **Transformer** (2017) introduced multi-head attention, allowing parallel processing across multiple attention heads, enabling widespread adoption in large-scale models.

#### **AI Boom Era**
Transformers became central to the AI boom, powering generative models like ELMo (2018) and BERT (2018). BERT, an encoder-only transformer, improved contextual embeddings, while GPT series (decoder-only) excelled in text generation. Beyond text, transformers extended to vision (Vision Transformer), speech (Whisper), and multimodal tasks, driving advancements in AI applications.

---

### **Training Methods**
#### **Stabilizing Training**
The original transformer architecture struggled with convergence. Warmup learning rates and layer normalization (pre-LN) were introduced to stabilize training. Pre-LN (2018) simplified training by eliminating the need for warmup, enhancing model performance.

#### **Pretrain-Finetune**
Transformers are typically pretrained using self-supervised learning on large datasets (e.g., The Pile), followed by fine-tuning on task-specific data. Pretraining tasks include language modeling, next-sentence prediction, and question answering. This approach leverages vast data to learn generalizable features, improving downstream task performance.

#### **Tasks**
Transformers handle three main task types:
- **Masked Language Modeling**: Predict masked tokens based on context.
- **Autoregressive Modeling**: Generate sequences step-by-step, conditioned on previous tokens.
- **Prefix Language Modeling**: Predict tokens based on a prefix of the sequence.

---

### **Architecture**
#### **Tokenization**
Text is converted into numerical tokens via tokenizers (e.g., Byte Pair Encoding, BPE). Tokens are mapped to integers, enabling matrix operations in transformers. Special tokens like `[CLS]` (classification) and `[SEP]` (separator) are used for structured inputs.

#### **Embedding and Positional Encoding**
Tokens are converted into dense vectors via embedding layers. Positional encoding ensures transformers understand sequence order, as they lack inherent positional information. Sinusoidal positional encodings (original Transformer) or learned embeddings (e.g., RoPE) are used.

#### **Encoder-Decoder Architecture**
Transformers use encoder-decoder structures:
- **Encoder**: Processes input sequences via self-attention and feedforward layers, producing contextualized representations.
- **Decoder**: Uses cross-attention to incorporate encoder outputs and self-attention to generate sequences autoregressively.

#### **Attention Mechanisms**
- **Scaled Dot-Product Attention**: Computes attention weights by scaling dot products of queries and keys, normalized by the square root of key dimension.
- **Multihead Attention**: Combines multiple attention heads to capture diverse semantic relationships, improving model flexibility.

#### **Masked Attention**
Causal masking in decoders prevents tokens from attending to future positions, ensuring autoregressive generation. Non-masked attention allows cross-attention between encoder and decoder.

---

### **Variants and Adaptations**
#### **Encoder-Only/Decoder-Only**
- **Encoder-Only**: Used for tasks like text embedding (BERT).
- **Decoder-Only**: Used for text generation (GPT series).
- **PrefixLM**: A decoder-only variant with prefix masking, enabling efficient generation.

#### **Efficient Implementations**
- **KV Caching**: Stores key-value pairs to avoid recomputation during inference.
- **FlashAttention**: Optimizes attention computation for GPUs, reducing memory usage.
- **Multi-Query Attention**: Reduces computational overhead by sharing keys and values across attention heads.

#### **Multimodal Applications**
Transformers extend beyond text to vision (Vision Transformer), speech (Whisper), and multimodal tasks (e.g., DALL-E for image generation). These models leverage tokenization techniques for diverse modalities.

---

### **Radial Basis Function Network (RBFN)**

#### **Network Architecture**
RBF networks consist of three layers:
- **Input Layer**: Receives input vectors.
- **Hidden Layer**: Uses RBF activation functions (e.g., Gaussian) centered at learned parameters.
- **Output Layer**: Linear combination of hidden layer outputs.

#### **Training**
RBFNs are trained in two steps:
1. **Center Selection**: Randomly sampled or via clustering (e.g., k-means).
2. **Weight Optimization**: Linear weights are adjusted to minimize reconstruction error (e.g., least squares).

#### **Applications**
RBFNs are used for function approximation, time series prediction, and control systems. Their local nature makes them robust to noise and outliers.

---

### **Class Activation Mapping (CAM)**

#### **Background**
CAM methods visualize neural network decisions by highlighting discriminative regions. **Class Activation Mapping (CAM)** and **Gradient-Weighted CAM (Grad-CAM)** are foundational techniques:
- **CAM**: Uses global average pooling (GAP) to generate class-specific heatmaps.
- **Grad-CAM**: Computes gradients of class scores to identify influential regions.

#### **Fine-Tuned Versions**
- **Guided Grad-CAM**: Combines Grad-CAM with guided backpropagation for sharper visualizations.
- **Grad-CAM++**: Uses higher-order gradients to improve localization accuracy.
- **Score-CAM**: Replaces gradients with model confidence scores for robustness.

---

### **Autoencoder**

#### **Mathematical Principles**
An autoencoder consists of an encoder (compresses input) and a decoder (reconstructs input). Training involves minimizing reconstruction error, often using L2 loss or cross-entropy.

#### **Variations**
- **Variational Autoencoder (VAE)**: Uses probabilistic models and KL divergence to enforce latent space structure.
- **Sparse Autoencoder**: Encourages sparsity in latent representations for better feature learning.
- **Denoising Autoencoder**: Learns robust representations by reconstructing noisy inputs.

#### **Applications**
Autoencoders are used for dimensionality reduction, anomaly detection, and image processing. They are also integral to modern AI systems like VAEs in Stable Diffusion and transformers in image generation.

---

### **Graph Neural Network (GNN)**

#### **Architecture**
GNNs process graph-structured data using message passing layers:
- **Message Passing**: Nodes update representations by aggregating messages from neighbors.
- **Local Pooling**: Reduces graph size via downsampling (e.g., top-k pooling).
- **Global Pooling**: Provides fixed-size graph representations (e.g., sum/mean).

#### **Message Passing Layers**
- **Graph Convolutional Network (GCN)**: Uses spectral filtering for graph convolution.
- **Graph Attention Network (GAT)**: Applies attention mechanisms to focus on relevant nodes.
- **Gated Graph Sequence Neural Network (GGS-NN)**: Extends GNNs to sequence tasks.

#### **Applications**
GNNs are pivotal in protein folding (AlphaFold), social network analysis, and combinatorial optimization. They enable modeling of complex relationships in graph-structured data.