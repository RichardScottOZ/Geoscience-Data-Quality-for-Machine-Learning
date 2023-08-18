# Geoscience-Data-Quality-for-Machine-Learning

A problem exists when building broad scale models, for example, Australia.

## Disparate datasets, breaking them down into broad domains:

- Geophysics (Gravity, Magnetics, Radiometrics, Seismic, Electromagnetic, Induced Polarisation, Magnetotelluric...)
- Geology (Lithology, Stratigraphy, Structure, Hydro..)
- Remote Sensing (Landsat, ASTER, Sentinel...)
- Geochemistry (Rock, Soil, Water, Assay techniques...)

## Variety of data layers:

- Direct observations
- Gridded Data
- Interpretations (Solid geology, SEEBase...)
- Derivations (e.g. ASTER band ratios, Rolling up of rock units...)
- Machine Learning Models (Regolith Depth...)
- Inversions

## Quality

- Age of science
- Technology used
- Resolution (Pixel size, map scale, survey spacing, detection limits..)
- Survey Type
- Human ratings? e.g. 1-10
- Downsampling/Upsampling
- Missing data (Geophysic survey blanks, Remote sensing gaps on old satellites..)

## Dimensionality

- 1
- 2
- 3
- 4
- more? (Depth Slices...)

## Scale

- World
- Country
- State
- Region
- Local

## Outputs

- Variance of different model runs

## Categorisation

How, thinking in a raster fashion, to get a combined per-pixel Data Quality rating for a map output.

- Some sort of normalised ranking for each quality area?
- Weightings?
- Simple qualitative (3/2/1, Good/Average/Bad, High/Medium/Low or other ordinals).
- Exists / Missing

  # Reference
  - [https://www.researchgate.net/profile/Alan_Aitken/publication/326193704/figure/fig1/AS:646297606443016@1531100765653/](https://www.researchgate.net/publication/326193704_A_role_for_data_richness_mapping_in_exploration_decision_making)

![sample map output](https://github.com/RichardScottOZ/Geoscience-Data-Quality-for-Machine-Learning/blob/main/reliability_index.png "Sample Quality Map - derived from Leonardo Uieda's Australia Gravity Data repository work")

![Framework from Aitken et al](
https://www.researchgate.net/profile/Alan_Aitken/publication/326193704/figure/fig1/AS:646297606443016@1531100765653/Four-levels-of-data-richness-Level-1-considers-the-presence-of-nearby-data-level-2_W640.jpg "Framework from Aitken et al")

