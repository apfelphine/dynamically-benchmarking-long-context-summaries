# Final Summary: Radial Basis Function (RBF) Networks

**Architecture and Principles**
Distinct from autoencoders, **RBF networks** utilize a three-layer architecture (input, Gaussian RBF hidden layer, linear output) to achieve universal approximation. The output $\varphi(\mathbf{x})$ is a weighted sum of radial basis functions normalized by their sum, which provides theoretical justification for stochastic data flow and kernel approximation. While Gaussian basis functions are inherently local, they can be extended with linear terms. Normalization significantly improves accuracy, particularly as input dimensionality increases.

**Training Process**
Training typically follows a two-step algorithm, often followed by optional fine-tuning:
1.  **Unsupervised Step:** Center vectors $\mathbf{c}_i$ are determined via random sampling, k-means clustering, or the Orthogonal Least Square Learning Algorithm, with widths fixed proportional to the maximum distance between centers.
2.  **Supervised Step:** Linear weights $w_i$ are optimized using three primary methods:
    *   **Pseudoinverse Solution:** An explicit linear approach ($\mathbf{w} = \mathbf{G}^+\mathbf{b}$) guaranteeing a minimum.
    *   **Gradient Descent:** Iteratively adjusts weights against the objective function's gradient.
    *   **Projection Operator Training:** An iterative method scaling updates by the error's projection onto the basis direction.

**Applications and Performance**
*   **Function Approximation:** Used for general tasks where centers are not fixed. With fixed centers at known data points, weights are solved via linear algebra. A case study on the logistic map ($x(t+1) = 4x(t)[1-x(t)])$ utilized 5 basis functions ($N=5$) and projector operator training ($\nu=0.3$), achieving an RMS error of 0.15 in a single pass. Further refinement using normalized radial basis functions with $\beta=6$ reduced the RMS error on a test set to **0.084**, outperforming unnormalized functions.
*   **Time Series Prediction:** RBF networks predict future states by iterating an estimated function based on known initial conditions. However, predictions are accurate only for a few time steps due to the **sensitive dependence on initial conditions** inherent in chaotic systems, as measured by the Lyapunov exponent.
*   **Control of Chaotic Systems:** This application aims to manipulate chaotic systems (e.g., the logistic map) to drive outputs toward a desired trajectory $d(t)$. It calculates a control parameter $c[x(t),t]$ to compensate for natural dynamics errors, utilizing a learning algorithm similar to the projection operator method that adjusts weights based on the error ($\varepsilon$) between actual and desired next steps.