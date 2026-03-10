# Geoscience-Data-Quality-for-Machine-Learning

A Python package for assessing geoscience data quality for machine learning.

A problem exists when building broad scale models, for example, Australia.
Disparate datasets from many domains need to be assessed for quality before
being combined into machine learning pipelines. This package provides tools
to quantify and map data quality across geoscience datasets.

## Installation

```bash
pip install -e .
```

With optional dependencies:

```bash
# For Excel file support
pip install -e ".[excel]"

# For gravity point-density analysis (verde, xarray, pooch)
pip install -e ".[gravity]"

# For visualization (matplotlib)
pip install -e ".[viz]"

# Everything
pip install -e ".[all]"

# Development (includes tests)
pip install -e ".[dev]"
```

## Package Modules

### `geoscience_data_quality.quality_model`
Quality scoring model for geoscience datasets. Load quality models from
CSV/Excel, compute resolution scores, final quality scores, and filter
by domain or sub-domain.

```python
from geoscience_data_quality import load_quality_model, compute_final_score, compute_resolution_score

model = load_quality_model("DataQuality_Models.csv")
res_score = compute_resolution_score(90.0)   # finer resolution → higher score
final = compute_final_score(score=3, presence=1.0, resolution_score=res_score)
```

### `geoscience_data_quality.vector`
Analyze quality fields (confidence, observation method, positional accuracy,
metadata) in geological vector datasets.

```python
from geoscience_data_quality import analyze_quality_fields, get_quality_summary

results = analyze_quality_fields(geology_gdf, fields=["confidence", "obsmethod"])
summary = get_quality_summary(geology_gdf)
```

### `geoscience_data_quality.survey`
Fetch, filter, and fix geophysical survey metadata from WFS services such
as Geoscience Australia's GADDS.

```python
from geoscience_data_quality import fetch_ga_survey_metadata, filter_surveys, fix_survey_geometry

surveys = fetch_ga_survey_metadata()
mag_line = filter_surveys(surveys, measure_type="magnetic", dataset_type="line")
gdf = fix_survey_geometry(mag_line, swap_coordinates=True)
```

### `geoscience_data_quality.rasterize`
Rasterize vector quality attributes onto reference grids or new grids
defined by bounds and resolution.

```python
from geoscience_data_quality import rasterize_vector_attribute

array = rasterize_vector_attribute(
    gdf, column="max_line_spacing_m",
    reference_raster="model_raster.tif",
    output_path="survey_quality.tif",
    sort_ascending=False,  # smallest (best) value wins in overlaps
)
```

### `geoscience_data_quality.point_density`
Compute observation point density for datasets like gravity stations
(requires the `gravity` optional dependencies).

```python
from geoscience_data_quality.point_density import compute_point_density

coords, counts = compute_point_density((longitude, latitude), spacing=0.1)
```

## Disparate datasets, breaking them down into broad domains

- Geophysics (Gravity, Magnetics, Radiometrics, Seismic, Electromagnetic, Induced Polarisation, Magnetotelluric...)
- Geology (Lithology, Stratigraphy, Structure, Hydro..)
- Remote Sensing (Landsat, ASTER, Sentinel...)
- Geochemistry (Rock, Soil, Water, Assay techniques...)

## Variety of data layers

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

## Reference

- [A role for data richness mapping in exploration decision making (Aitken et al)](https://www.researchgate.net/publication/326193704_A_role_for_data_richness_mapping_in_exploration_decision_making)

![sample map output](https://github.com/RichardScottOZ/Geoscience-Data-Quality-for-Machine-Learning/blob/main/reliability_index.png "Sample Quality Map - derived from Leonardo Uieda's Australia Gravity Data repository work")

![Framework from Aitken et al](
https://www.researchgate.net/profile/Alan_Aitken/publication/326193704/figure/fig1/AS:646297606443016@1531100765653/Four-levels-of-data-richness-Level-1-considers-the-presence-of-nearby-data-level-2_W640.jpg "Framework from Aitken et al")

