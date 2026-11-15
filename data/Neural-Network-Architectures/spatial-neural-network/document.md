# Spatial neural network

## History
Openshaw (1993) and Hewitson et al. (1994) started investigating the applications of the a-spatial/classic NNs to geographic phenomena. They observed that a-spatial/classic NNs outperform the other extensively applied a-spatial/classic statistical models (e.g. regression models, clustering algorithms, maximum likelihood classifications) in geography, especially when there exist non-linear relations between the geo-spatial datasets' variables. Thereafter, Openshaw (1998) also compared these a-spatial/classic NNs with other modern and original a-spatial statistical models at that time (i.e. fuzzy logic models, genetic algorithm models); he concluded that the a-spatial/classic NNs are statistically competitive. Thereafter scientists developed several categories of SNNs – see below.


## Spatial models
Spatial statistical models (aka geographically weighted models, or merely spatial models) like the geographically weighted regressions (GWRs), SNNs, etc., are spatially tailored (a-spatial/classic) statistical models, so to learn and model the deterministic components of the spatial variability (i.e. spatial dependence/autocorrelation, spatial heterogeneity, spatial association/cross-correlation) from the geo-locations of the geo-spatial datasets' (statistical) individuals/units.


## Categories
There exist several categories of methods/approaches for designing and applying SNNs.

- One-Size-Fits-all (OSFA) spatial neural networks, use the OSFA method/approach for globally computing the spatial weights and designing a spatial structure from the originally a-spatial/classic neural networks.[2]
- Spatial Variability Aware Neural Networks (SVANNs) use an enhanced OSFA method/approach that locally recomputes the spatial weights and redesigns the spatial structure of the originally a-spatial/classic NNs, at each geo-location of the (statistical) individuals/units' attributes' values.[3] They generally outperform the OSFA spatial neural networks, but they do not consistently handle the spatial heterogeneity at multiple scales.[10]
- Geographically Weighted Neural Networks (GWNNs) are similar to the SVANNs but they use the so-called Geographically Weighted Model (GWM) method/approach by Lu et al. (2023), so to locally recompute the spatial weights and redesign the spatial structure of the originally a-spatial/classic neural networks.[1][9] Like the SVANNs, they do not consistently handle spatial heterogeneity at multiple scales.[1]



## Applications
There exist case-study applications of SNNs in: 

- energy for predicting the electricity consumption;[11]
- agriculture for classifying the vegetation;[12]
- real estate for appraising the premises.[13][1]