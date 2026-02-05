# Index

The methods pages document the public API of the modules that live in the `BayesDensity` monorepo.
To get an overview of the functionality common to all `BayesDensity` models, it is strongly recommended to read the [general API](../api/general_api.md) page before studying the individual method docs.

## Overview

The following table provides a complete list of all the models available in `BayesDensity`, the corresponding module, along with the algorithms that are currently available for posterior inference.

| Model    | MCMC   | VI    | Module         |
| :---------- | :----: | :----: | :------------------------- |
| [`BSplineMixture`](@ref) | ✅ | ✅ | `BayesDensityBSplineMixture` |
| [`FiniteGaussianMixture`](@ref) | ✅ | ✅ | `BayesDensityFiniteGaussianMixture` |
| [`HistSmoother`](@ref) | ✅ | ✅ | `BayesDensityHistSmoother` |
| [`PitmanYorMixture`](@ref) | ✅ | ✅ | `BayesDensityPitmanYorMixture` |
| [`RandomFiniteGaussianMixture`](@ref) | ✅ | ✅ | `BayesDensityFiniteGaussianMixture` |