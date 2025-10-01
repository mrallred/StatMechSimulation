# From Monte Carlo Simulations to Analytic Solutions of Ising Models

Statistical mechanics applies statistical and probablistic methods to systems of many interacting degrees of freedom that are coupled to each other through some quenched interactions. These methods were originally developed by physicists including Boltzmann, Maxwell, and Gibbs, where they were used to explain macroscopic physical properties like temperature, pressure and heat capactiy, in terms of microscopic paramters characterized by probability distributions. 

The ideas of statistical mechanics can be used to analyze neural systems, biological and artifical, providing a valuable theoretical framework for conceptualizing their computational and information storage capacities.

## Spin Glasses
Spin glasses are the paradigmatic disordered systems to which statistical mechanical approaches are useful.

Experimentially, these types of systems can be created by rapidly cooling a liquid magnetic alloy made up of strongly interacting ions (a.k.a spins) immeresed in a weakly interacting substrate. As a result, the strength of the pairwise interactions between ions is some function $ J(\bf{r}) $ of the distance vector $\bf{r}$ between them. Since ion positions depend on the specific experimental relization of the alloy, distances between them are randomly distributed, and values of $ J(\bf{r}) $ are randomly positive and negative, leading to rough energy landscapes and "glassy" behavior of the system.

To make these systems tractable, in models we typically assume the distance between ions are fixed, as on a lattice, and that the couplings between ions are quenched random variables.

### Generic features of spin glass models:
There are two main features all spin glass models share:
#### 1. Quenched disorder
**Defintion:** The static parameters in the Hamiltonian that do not change as the system evolves. 

In the case of spin glasses, this is the the coupling matrix $ J(\bf{r}) $, as once the system has been created these values do not change.


## Frustrated Triangle

<img src="graphics/frustrated_triangle.png">

N = 3

Number of Possible Ising Spin Configurations: $2^{3} = 8$
$$ 
J = 
\begin{bmatrix}
0 & -1 & 1 \\
-1 & 0 & 1 \\
1 & 1 & 0
\end{bmatrix}
$$ 

|      | Configuration $S_{i}$ | $H(S_{i}, J)$ | $\mathbb{P}(S_{i})$ |
| -----| --------------------- | ------------- | ------------------- |
|     A| $\left[1,1,1 \right]$ | $-1.0$        |  $\frac{1}{3} e^{-3\beta}$| 
|     B| $\left[1,1,-1 \right]$ | $-1.0$        |  $\frac{1}{3} e^{-3\beta}$| 
|     C| $\left[1,-1,1 \right]$ | $-1.0$        |  $\frac{1}{3} e^{-3\beta}$| 
|     D| $\left[1,-1,-1 \right]$ | $-1.0$        |  $\frac{1}{3} e^{-3\beta}$| 
|     E| $\left[-1,1,1 \right]$ | $-1.0$        |  $\frac{1}{3} e^{-3\beta}$| 
|     F| $\left[-1,1,-1 \right]$ | $-1.0$        |  $\frac{1}{3} e^{-3\beta}$| 
|     G| $\left[-1,-1,1 \right]$ | $-1.0$        |  $\frac{1}{3} e^{-3\beta}$| 
|     H| $\left[-1,-1,-1 \right]$ | $-1.0$        |  $\frac{1}{3} e^{-3\beta}$| 


## Hamiltonian
$$ \mathcal{H}(S,J) = - \sum_{i\lt j}^{N} J_{ij}\sigma_{i}\sigma_{j} - h\sum_{i}{\sigma_{i}}$$

$$ \text{where } \sigma_{i} = \pm1 $$

## Boltzmann Distribution
This is the probability of the system being in certain configuation at a particular temperature.

$$\mathbb{P}^{Boltzmann}(S_{i}, J) = \frac{1}{Z(J, \beta)} e^{-\beta H(S_{i},J)}$$ 

where $\beta = 1/T$ is  inverse temperature and the denominator is a normalizing factor called the **partition function (sum):**
$$ 
Z(J, \beta) = \sum_{S}e^{-\beta H(S,J)}
$$



# Presentation notes
### Introduction
-   Myles Allred
-   DRP summer: statistical mechanics of spin glasses
-   mentored by Jennifer
-   Sherrington-Kirkpatrikck (SK) model

### SK Model
<img src="graphics/SKModel.png">

Fully connected graph with N = 30 degree of freedom

J is a adjacency matrix of fully connected graph -- a mean field approximation as every node interacts with every other node.

The weights $J_{ij}$ are drawn from a zero mean guassian distribution with variance $\frac{1}{N}$

## Measures 
**Hamiltonian**: 
A cost function or energy measure for a certain configuration of spins
$$ \mathcal{H}(S,J) = - \sum_{i\lt j}^{N} J_{ij}\sigma_{i}\sigma_{j}$$

$$ \text{where } \sigma_{i} = \pm1 $$

**Boltzmann Probability**: Probability measure for a certain state based on its energy and the temperature of the system
$$\mathbb{P}^{Boltzmann}(S_{i}, J) = \frac{1}{Z(J, \beta)} e^{-\beta H(S_{i},J)}$$ 

where $\beta = 1/T$ is  inverse temperature and the denominator is a normalizing factor called the 

**Partition Function (sum)**: Normalizing factor for probabilities
$$ 
Z(J, \beta) = \sum_{S}e^{-\beta H(S,J)}
$$


**Free Energy**: A energy measure of the entire system, independent of current state. 
$$F(J) = -T \text{ ln } Z(J)$$




### Ways to analyze this system:

| **Approach**  | **Pros** | **Cons** |
| ------------  | -------- | -------- |
| 1. Brute Force| Exact    | Computationally intractable |
| 2. Monte Carlo Sampling| Efficient and simple, represents behavior at thermodynaic equilibrium | Only tells us about a single walk, may get stuck in local energy minima.
| 3. Replica Method | Exact, analytic solution that tells us about all equillibrium configurations | Diffucult to solve, requires an anasatz (assumption) about the overlap of equillibrium configurations to solve.
### Brute force: 


$$
J = 
\begin{bmatrix} 
0 & 0.01586 & -0.6870 & 0.01393 \\
0.01586 & 0 & 0.0919 & -0.0169 \\
-0.6870 & 0.0919 & 0 & 0.0583 \\
0.0139 & 0.0169 & -0.1583 & 0
\end{bmatrix}
$$


