# Extension Proposal: Fourier Neural Operator for Characteristic Function Evolution

## 1. Problem Reformulation: The Forward Operator
Instead of the current **Inverse Problem** formulation (History $\to$ Price Surface), we propose a **Forward Problem** formulation where the Neural Operator learns the time-evolution of the asset's probability distribution in the frequency domain.

### Mathematical Basis
In affine jump-diffusion models (like Heston), the characteristic function $\phi(u, \tau)$ of the log-price $X_T$ obeys a semi-analytical evolution equation. For Heston, this is governed by a **Riccati Ordinary Differential Equation**:

$$ \frac{\partial \phi(u, \tau)}{\partial \tau} = D(u) \phi(u, \tau) + \text{NonLinear}(\phi) $$

where $u$ is the Fourier frequency mode and $\tau$ is time-to-maturity.

### The Learning Task
We treat the characteristic function $\phi(\cdot, t)$ as the infinite-dimensional state of the system. The goal is to learn the operator $\mathcal{K}$ that advances this state:

$$ \phi(\cdot, t + \Delta t) = \mathcal{K}(\phi(\cdot, t)) $$

## 2. Proposed Architecture: Fourier Neural Operator (FNO)
The **Fourier Neural Operator (FNO)** is the ideal architecture for this formulation because the state $\phi$ is already defined in the frequency domain.

### Specification
*   **Domain**: The frequency space $u \in [u_{min}, u_{max}]$.
*   **Input**: The complex-valued characteristic function $\phi(u, t)$ at time $t$.
    *   Represented as a 2-channel tensor (Real, Imaginary) on a 1D grid.
*   **Output**: The characteristic function $\phi(u, t+\Delta t)$ at the next time step.
*   **Architecture**: 1D FNO (Li et al., 2020).
    *   Fourier Layers perform global convolution via FFT.
    *   Since the input is already in frequency domain, the "Fourier Transform" step of FNO becomes an identity (or a domain shift), making the architecture extremely efficient.

## 3. Advantages over DeepONet
1.  **Homogeneous Domain**: Input and Output are both functions of $u$. There is no domain mismatch (Time $\to$ Surface), eliminating the need for the Branch/Trunk split.
2.  **Rough Volatility**: Rough Heston models are characterized by **Fractional Riccati Equations**. An FNO could empirically learn the fractional derivative operator from data without needing to explicitly solve the fractional ODE, potentially generalizing better to $H < 0.5$.
3.  **Universal Pricing**: Once $\phi(u, T)$ is predicted, **any** European option price can be computed efficiently using the Fast Fourier Transform (FFT) method (Carr-Madan formula).

## 4. Implementation Roadmap
1.  **Data Generation**: Simulate Heston/Rough Heston paths. Compute empirical characteristic functions (ECF) via FFT of the density histograms at each time step.
2.  **Training**: Train 1D FNO to map $\text{ECF}_t \to \text{ECF}_{t+1}$.
3.  **Loss Function**: $L^2$ norm in frequency domain. By Parseval's theorem, minimizing error in $\phi$ minimizes error in the probability density function (PDF).
