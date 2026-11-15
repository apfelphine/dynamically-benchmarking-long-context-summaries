# Neural Network Architectures and Methods

## Class Activation Mapping (CAM)
Class activation mapping (CAM) and its variants provide interpretability for convolutional neural networks (CNNs) by highlighting image regions important for class prediction. Original CAM modifies the CNN architecture by replacing fully connected layers with a global average pooling (GAP) layer, making visual explanations model-specific. Grad-CAM generalizes the approach using backpropagated gradients, enabling applicability to any CNN, including intermediate layers without architectural changes. Extensions like Guided Grad-CAM, Grad-CAM++, Score-CAM, and LayerCAM further improve localization, resolution, and flexibility of activation map generation.

## Vision Transformers (ViT)
Vision Transformers apply transformer architectures to image classification. The model processes images by splitting them into patches, encoding their position, and feeding them through transformer layers. ViT outperforms traditional CNNs on large datasets by capturing long-range dependencies. Variants such as the Masked Autoencoder (MAE), DINO (self-distillation), Swin Transformer (windowed attention), and TimeSformer (video factorized attention) introduce improvements for efficiency, robustness, self-supervised learning, and domain adaptation. Transformers surpass CNNs in many vision tasks, require substantial data, and exhibit resilience to image distortion.

## Variational Autoencoders (VAE)
VAEs are generative models that encode data into a lower-dimensional latent space using a neural network (encoder) and reconstruct the data via a decoder. They optimize a lower bound on data likelihood by minimizing reconstruction error and the Kullback–Leibler divergence (or other statistical distances) between learned and target distributions. Variants include β-VAE for disentangled representations, conditional VAE for labeled data, and hybrid models (e.g., VAE-GAN). Recent research explores alternative statistical distances for regularization.

## Wasserstein GAN
Wasserstein GAN addresses weaknesses in standard GAN training, such as mode collapse and vanishing gradients, by using the Wasserstein distance instead of Jensen–Shannon divergence in its objective. This improves the feedback provided by the discriminator and stabilizes training. Regularization methods such as weight clipping, spectral normalization, and gradient penalty are used to constrain the Lipschitz norm of the discriminator during training.

## Graph Neural Networks (GNN)
GNNs operate on graph-structured data, updating node representations through message passing and aggregation. Key types include message passing neural networks (MPNN), graph convolutional networks (GCN), graph attention networks (GAT), and gated graph sequence neural networks (GGS-NN). Pooling layers (top-k, self-attention) coarsen graphs, increasing receptive field. GNNs face limitations in expressiveness and distinguishing certain graph structures, and specialized adaptations address heterophily. Applications span protein folding, social networks, combinatorial optimization, cybersecurity, and computer vision.

## Generative Adversarial Networks (GAN)
GANs consist of a generator and discriminator in a zero-sum game to model data distributions and generate realistic samples. Variants include conditional GANs, architecture-specific GANs (DCGAN, SAGAN, TransGAN), objective changes (Wasserstein, hinge loss, least squares), and multi-player setups (BiGAN, CycleGAN, InfoGAN). StyleGAN architectures leverage progressive and style-based training for high-quality image synthesis. Challenges persist in training stability, mode collapse, and evaluation methods (Inception score, FID, LPIPS). GANs have broad scientific, medical, artistic, and malicious applications.

## Autoencoder
Autoencoders learn compressed representations of data via an encoder–decoder architecture, minimizing reconstruction error. Undercomplete autoencoders achieve dimensionality reduction; sparse, denoising, contractive, MDL, and concrete autoencoders introduce specific regularizations and selection mechanisms. Deep autoencoders improve efficiency and feature representation. Applications include dimensionality reduction, information retrieval, anomaly detection, image processing, drug discovery, and communication systems.

## Transformer Networks
The transformer replaces recurrence with multi-head self-attention, enabling parallel processing of sequence data. Core components include tokenization, embedding layers, positional encoding, encoder–decoder layers, feedforward networks, and scaled dot-product attention. Training is stabilized by layer normalization, warmup, and pretrain–finetune paradigms. Architectural advances include alternative activations, normalization, positional encodings (RoPE, ALiBi), and efficient attention (FlashAttention, MQA, GQA). Transformers excel in NLP, multimodal tasks (vision, audio), and large-scale generative modeling.

## Radial Basis Function Networks (RBF)
RBF networks use radial basis functions in hidden layers for function approximation and interpolation. They are universal approximators and trained using center determination (e.g., k-means), linear weight fitting (least squares or pseudoinverse), and regularization. Applications include time series prediction, control, and function interpolation, with normalized architectures demonstrating improved accuracy.

## Convolutional Neural Networks (CNN)
CNNs extract hierarchical features from data via convolutional layers, pooling, and nonlinear activations (e.g., ReLU). Shared weights, receptive fields, and pooling provide translational invariance and efficient learning. Regularization methods (dropout, DropConnect, stochastic pooling, weight decay) mitigate overfitting. CNN architectures are widely used for image and video analysis, NLP, anomaly detection, and specialized domains like drug discovery and board games. Transfer learning and interpretability (saliency maps) are increasingly important for practical deployment.

## Recurrent Neural Networks (RNN)
RNNs process sequential data by maintaining hidden states across time. Core variants include Elman and Jordan networks, LSTM (with forget gates for long-term memory), and GRU (simplified LSTM). Further configurations include stacked, bidirectional, encoder–decoder, pixel RNNs, NTMs, and continuous-time RNNs. RNN training challenges involve vanishing gradients and are addressed via methods like teacher forcing, BPTT, IndRNN, and CRBP. Applications cover machine translation, time series prediction, speech recognition, and more.

## Hopfield Networks
Hopfield networks are associative memory models using binary or graded neurons converging to attractor states with an energy function. Classical models have limited storage capacity, improved by advanced learning rules (Storkey), and modern extensions (Dense Associative Memory) increase memory capacity via nonlinear energy functions. Used for pattern storage, recall, and optimization problems, Hopfield networks link to human memory analogies and are foundational in statistical mechanics approaches to neural computation.