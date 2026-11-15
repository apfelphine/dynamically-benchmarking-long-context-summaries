# Class activation mapping

## Background
The following methods laid the groundwork for the class activation maps approaches, forming the conceptual basis of using gradients to highlight class-discriminative regions.


### Class model visualization and saliency maps for convolutional neural networks
The class model visualization and image-specific saliency maps approaches have been presented in the foundational work "Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps" by Karen Simonyan, Andrea Vedaldi, and Andrew Zisserman
 and it generalizes the deconvnet method by Zeiler and Fergus.

- Class model visualization synthesizes an artificial input image that strongly activates the output neurons associated with a target class. Given a trained, fixed model, this method starts with a zero-initialized image, backpropagates the gradients from the class score to the image pixels, updates the image pixels increasing the specific class scores and it repeats the pixel updating process, showing an encoded (idealized version) prototype of the class of interest.[6]
- Image-specific class saliency visualization method provides a visual explanation by highlighting the most relevant pixels in an image for predicting a certain class C of interest. This is done by computing the gradient of the class score with respect to the input image,  ${\displaystyle I_{0},}$ 
  ${\displaystyle w=\left.{\frac {\partial S_{C}}{\partial I}}\right|_{I_{0}}}$ 
 approximating the model locally (around  ${\displaystyle I_{0}}$ ) as linear, using a first-order Taylor expansion:
  ${\displaystyle S_{C}(I)\approx w_{C}^{T}I+b}$ .
 The magnitude of  ${\displaystyle w_{C}}$ , the gradient, indicates the importancy of the pixels: larger gradients suggest greater influence on the prediction. Once the gradient is known, the saliency map is defined as the maximum absolute gradient across the color channels:
  ${\displaystyle M_{ij}=max_{C}\left|{\frac {\partial S_{C}}{\partial I_{ij}^{C}}}\right|}$  
 resulting in an saliency map (i.e. heatmap).[6][8]



### Guided backpropagation
The concept of guided backpropagation can be traced for the first time in the paper by Springenberg et al. "Striving For Simplicity: The All Convolutional Net"  and also this method builds upon the work by Zeiler and Fergus "Visualizing and Understanding Convolutional Networks".

Guided backpropagation core is to understand what a CNN is learning, by visualizing the patterns that activate more strongly individual neurons (or filters), in architectures which do not rely on max-pooling layer.

When propagating gradients back through a rectified linear unit (ReLU), guided backpropagation passes the gradient if and only if the input to the ReLU was positive (forward pass) and the output gradient is positive (backward signal), tackling both inactive neurons, negative gradients and suppressing the noise. The result displays sharper, high-resolution visualizations of what each neuron is responding to.

Guided backpropagation represents a simple and practical method for model interpretability, helping understand how and where neural networks detect semantic concepts across layers. Moreover, it can be applied to any network architecture, due to its working principle.


## Base versions
Class activation mapping and gradient-weighted class activation mapping are the original and most widely used methods for visual explanations in convolutional neural networks. These methods serve as the foundation for many later developments in explainable AI.

Notation: In this article, the symbols i and j represent integer indices that disappear inside sums or averages, while x and y are the continuous (or up-sampled integer) coordinates of the final heat-map that is plotted.


### Class activation mapping (CAM)
Class activation mapping (CAM) was the first, and the original, version of CAM methods, and it gave the name to the whole category. The approach was firstly introduced by Zhou et al. in their seminal work "Learning Deep Features for Discriminative Localization".

This approach achieves class-specific heatmaps by modifying image classification CNN architectures, replacing fully-connected layers with convolutional layers and a final global average pooling layer.

Its main scope is to localize and highlight discriminative regions of an input image that a CNN uses to identify a particular class, without needing explicit bounding box annotations.


#### Global average pooling (GAP)
Global average pooling (GAP) represents the key element in the original CAM approach.

It is a dimensionality reduction technique and, similarly to other pooling layers, it allows the downsampling of the feature maps, calculating representative values for a specific region of the feature map. The particularity of GAP is that it calculates a single value for an entire feature map, significantly reducing the model dimensions.


#### Mathematical description
The mathematical description considers as its key the combination of convolutional and GAP layers.

In CAM, it is mandatory to have the GAP layer after the last convolutional layer and before the final linear classifier layer. This last element of the architecture connects the output logits (the network predictions)  ${\displaystyle y^{C}}$ , to the GAP values, with its respective fine-tuned weights,  ${\displaystyle w_{k}^{C}}$ .

Considering  ${\displaystyle A^{k}}$  as the last feature maps of the last convolutional layer, GAP produces one value for each feature map, by averaging all the matrix elements (i, j) of the feature map:

 ${\displaystyle F^{k}={\frac {1}{mn}}\sum _{i=1}^{m}\sum _{j=1}^{n}A_{ij}^{k}}$ 

with
 ${\displaystyle A^{k}={\begin{bmatrix}A_{11}^{k}&A_{12}^{k}&\cdots &A_{1n}^{k}\\A_{21}^{k}&A_{22}^{k}&\cdots &A_{2n}^{k}\\\vdots &\vdots &\ddots &\vdots \\A_{m1}^{k}&A_{m2}^{k}&\cdots &A_{mn}^{k}\end{bmatrix}}=\left\{A_{ij}^{k}\mid 1\leq i\leq m,\ 1\leq j\leq n\right\}}$ 

Namely, in the GAP layer, each feature map is reduced to a single scalar via GAP, producing k values, hence reducing the dimensionality of the network. A k ${\displaystyle \times }$ m ${\displaystyle \times }$ n tensor is reduced to k scalars, shrinking the parameter count for the linear classifier head.

The final output logits are calculated as the linear sum of the GAP values, weights and bias:

 ${\displaystyle y^{C}=\sum _{k}w_{k}^{C}F^{k}}$ 

The localization map is computed as follows:

 ${\displaystyle L_{CAM}^{C}(x,y)=ReLU(\sum _{k}w_{k}^{C}A_{k}(x,y))}$ 

namely,  ${\displaystyle A_{k}(x,y)}$  is the activation of node k in the target layer of the model, and  ${\displaystyle w_{k}^{C}}$  is the class-specific weight, for the channel k, in the linear classifier layer.


#### Advantages and drawbacks
The use of the GAP layer represents an example of an interpretability by design (IBD) approach. IBD refers to a technique which uses the model's own architecture to help explain its predictions.

The main drawback of CAM is that it is highly model-specific, being applicable to CNN architectures whose layer before the softmax one is a GAP.

Since the approach relies on the post-GAP weights for the overall evaluation, the method can't be applied to intermediate layers.

The choice of dealing with an IBD approach restricts the possibility to generalize the model architecture. Moreover, IBD methods often require re-training of the model.


### Gradient-weighted class activation mapping (Grad-CAM)
Gradient-weighted class activation mapping (Grad-CAM) is a generalized version of CAM and it tackles its architectural limitations. Grad-CAM computes the gradient of a target class score, the pre-softmax logit, with respect to the feature maps of a convolutional neural network. The gradients are global-average-pooled to obtain importance weights, which are used to compute a class-specific localization map by linearly weighting the feature maps. The result is a heatmap that highlights the regions in the input image that are the most influential for predicting the target class.

The main advantage of Grad-CAM, with respect to the standard CAM, is that it is model agnostic (provided that the network still needs to be differentiable), meaning that it generates visual explanation for any CNN-based network without architectural changes or re-training, making it broadly applicable to pre-trained models.


#### Mathematical description
Considering:

- y ${\displaystyle ^{C}}$  the logits (i.e. the pre-softmax activated neurons responsible for a certain class prediction) of interest;
- A ${\displaystyle ^{k}}$  the feature activated map for a specific convolutional layer;
- L ${\displaystyle _{\text{Grad-CAM}}^{C}}$  ∈  ${\displaystyle \mathbb {R} ^{u\times v}}$  the class-discriminative localization map, of width u and height v for any class c;


Grad-CAM, employing backpropagation, computes the logit gradient with respect to the feature map A ${\displaystyle ^{k}}$  as

 ${\displaystyle {\frac {\partial {y^{C}}}{\partial {A^{k}}(i,j)}}}$ 

highlighting the importance of a certain class discrimination decision process of the logit.

These gradients are global-average-pooled over each element of the feature map (hence, highlighting the "importance" of the elements of a feature map k for a target class C):

 ${\displaystyle \alpha _{k}^{C}={\frac {1}{uv}}\sum _{i}\sum _{j}{\frac {\partial {y^{C}}}{\partial {A^{k}}(i,j)}}}$ 

So, to account for the total number of feature maps, each of them is multiplied by its weight (via dot-product) and element-wise summation is done:

 ${\displaystyle \sum _{k}\alpha _{k}^{C}A^{k}}$ 

It can be observed that, due to the intrinsic nature of the gradient operation, some elements of the weighted feature map will have negative value, so, since only elements that have increased the logit of the predicted class are of interest, a ReLU activation function is applied:

 ${\displaystyle L_{Grad-CAM}^{C}(x,y)=ReLU(\sum _{k}\alpha _{k}^{C}A^{k}(x,y))}$ 

Lastly, the output heatmap image dimensions are upsampled to the original image size to match the input dimensions.


#### Advantages and drawbacks
Grad-CAM addresses the most important CAM limitations. It makes CAM free from the GAP layer need, generalizing its behavior and enabling visual explanation at intermediate layers.

However, Grad-CAM focuses on the most discriminative region when contributing to classification. If multiple similar objects are present, Grad-CAM often highlights only one of them, or part of one, providing also coarser maps and lower localization accuracy.

Moreover, Grad-CAM retrieves backwards information (the gradients), without taking into consideration how the activation flowed forward during prediction (unless combined with the guided backpropagation technique), resulting in a certain probability of missing patterns highlighted in the forward signal. On top of that, Grad-CAM heat maps are low-resolution when choosing a very deep layer.

Lastly, false emphasis in the heatmap may be present when large gradients are computed for low activation values. Grad-CAM assumes that gradient implies importance, ignoring the activation features value.


### Grad-CAM and CAM comparison
| Feature                                           | CAM                                                                                                        | Grad-CAM                                                                                            |
| ------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| Architecture                                      | Requires CNN whose last convolutional layer is followed by GAP                                             | Works with any CNN architecture that can be backpropagated                                          |
| Flexibility                                       | Only works with GAP-based networks                                                                         | Generalized, gradient-based version, it works with any pre-trained CNN at any layer                 |
| Working principle                                 | Applies weights (class-specific linear classifier learned after GAP) from the output layer to feature maps | Compute importance by calculating the gradients of the class score with respect to the feature maps |
| Mathematical description of the working principle | ${\displaystyle ReLU(\sum _{k}w_{k}^{C}A_{k})}$                                                            | ${\displaystyle ReLU(\sum _{k}\alpha _{k}^{C}A^{k})}$                                               |



## Fine-tuned versions
Several methods have refined Grad-CAM to improve clarity and flexibility. Guided Grad-CAM, Grad-CAM++, Score-CAM, and Layer-CAM enhance aspects such as localization accuracy, gradient independence, and multi-layer visualization. These techniques build directly on the principles of CAM and Grad-CAM.


### Guided Grad-CAM
Guided Grad-CAM fuses the coarse, class‐discriminative localization of Grad-CAM with the high‐resolution details of guided backpropagation. Grad-CAM heatmap is first computed for the target class, and it is upsampled to the input size. Then, a Guided Backpropagation saliency map for the same class is computed. A final element‐wise product of the two results in the Guided Grad-CAM visualization map.

The result is a high-resolution, class-specific saliency map that highlights exactly which pixels contribute most to the network's decision.


### Grad-CAM++
Grad-CAM++ introduces a more refined way of computing the weights for each feature map, bypassing the global average of the gradients approach provided by Grad-CAM. This approach aims to improve the visual effect when multiple target instances are present in a single image.

Specifically, Grad-CAM++ employs pixel-wise gradients (via higher-order gradients), to compute the importance of a specific pixel for a prediction, lighting up multiple object instances in the same image.

These improvements allow for a more sensible and detailed output heatmap.

The associated mathematical framework is defined by the following localization map:

 ${\displaystyle L_{Grad-CAM++}^{C}(x,y)=\sum _{k}w_{k}^{C}A_{k}(x,y)}$ 

in which the coefficient  ${\displaystyle w_{k}^{C}}$  is defined as:

 ${\displaystyle w_{k}^{C}=\sum _{i,j}\alpha _{k}^{C}(i,j)\times ReLU({\frac {\partial y^{C}}{\partial A_{k}(i,j)}})}$ 

with  ${\displaystyle A^{k}(x,y)}$ , the activation of node k in the target layer of the model at position (x,y);  ${\displaystyle y^{C}}$  the logit score for class C, and  ${\displaystyle \alpha _{k}^{C}(i,j)}$  being:

 ${\displaystyle \alpha _{k}^{C}(i,j)={\frac {\frac {\partial ^{2}y^{C}}{\partial (A_{k}(i,j))^{2}}}{2\times {\frac {\partial ^{2}y^{C}}{\partial (A_{k}(i,j))^{2}}}+\sum _{a,b}A_{k}(a,b)\times {\frac {\partial ^{3}y^{C}}{\partial (A_{k}(i,j))^{3}}}}}}$ 

While addressing some Grad-CAM problems, Grad-CAM++ method still relies on gradients, and it only improves the underlying math. It is, however, still based on the idea of assigning a direct and valid relationship between gradient and importance.

Notation: (a,b) indexes all pixel positions in the feature‐map, exactly like (i,j) does, but for the summation in the denominator.


### Score-CAM
Score-CAM is a gradient-free CAM technique, thus redefining the original Grad-CAM and Grad-CAM++ working principles. It uses the model confidence scores instead of gradients.

Score-CAM performs the following operations:

1. Extracts the feature maps  ${\displaystyle A_{k}}$  of the final convolutional layers, as in the original Grad-CAM;
2. Upsamples each activation map  ${\displaystyle A_{k}}$  to the same input image dimensions, defining a mask  ${\displaystyle M_{k}}$  and each mask is normalized;
3. Multiplies the original input image by the mask, defining a masked image  ${\displaystyle X'_{k}=M_{k}\odot X}$  ( ${\displaystyle \odot }$  is the element-wise multiplication);
4. Gets a confidence score (a softmax probability is the output value after the softmax operation; the logit is the value before the softmax) for the masked images  ${\displaystyle X'_{k}}$ , by feeding it into the CNN (either the soft-max probability or the raw logit can be used; both yield similar results in practice);
5. Considers that confidence score as the weight  ${\displaystyle w_{k}^{c}}$  for the feature map  ${\displaystyle A_{k}}$ .


These operations allow to replace the gradient calculations with the actual model outputs, building more accurate heatmaps.

Mathematically, the localization map is defined as:

 ${\displaystyle L_{Score-CAM}^{C}(x,y)=ReLU(\sum _{k}w_{k}^{c}A_{k}(x,y))}$ 

and the coefficient  ${\displaystyle w_{k}^{C}}$  s:

 ${\displaystyle w_{k}^{C}=softmax_{k}(y^{C}(X'_{k}))={\frac {exp(y^{C}(X'_{k}))}{\sum _{m}exp(y^{C}(X'_{m}))}}}$ 

where  ${\displaystyle A_{k}(x,y)}$  is the activation of channel k  at location (x,y),  ${\displaystyle y^{C}(X)}$  is the logit for class C for an input X and  ${\displaystyle M_{k}}$  is the mask, defined as:

 ${\displaystyle M_{k}(x,y)={\frac {U(A_{k})(x,y)-min_{x,y}U(A_{m})}{max_{x,y}U(A_{k})-min_{x,y}U(A_{k})}}}$ 

with  ${\displaystyle U}$  as the upsampling operation.

Since the process of score calculation is repeated for every channel, Score-CAM is slow with respect to gradient-based methods. Moreover, it focuses on regions highlighted by individual feature maps, ignoring the context of the full image, reducing interpretability in complex scenes.


### LayerCAM
LayerCAM enhances backwards class-specific gradients using both intermediate and final convolutional layers. Combining information across layers allows to achieve higher resolution and more fine-grained detail, improving localization.

Specifically, for each position in the feature map, LayerCAM evaluates the gradient. The positive gradients are employed as the weights,  ${\displaystyle w_{k}^{C}}$ . Namely:

 ${\displaystyle w_{k}^{C}(x,y)=ReLU({\frac {\partial y^{C}}{\partial A_{k}(i,j)}}(x,y))}$ 

The activations are then weighted, and the final class activation map is retrieved by summing over the channels.

 ${\displaystyle L_{Layer-CAM}^{C}(x,y)=ReLU(\sum _{k}w_{k}^{C}(x,y)A_{k}(x,y))}$ 

This technique offers high-resolution heatmaps, flexible localization and per-location precision, employing positive gradients.