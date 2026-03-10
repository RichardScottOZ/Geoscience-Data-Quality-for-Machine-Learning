"""Tools for assessing geoscience data quality for machine learning."""

__version__ = "0.1.0"

from geoscience_data_quality.quality_model import (
    compute_final_score,
    compute_resolution_score,
    load_quality_model,
)
from geoscience_data_quality.rasterize import rasterize_vector_attribute
from geoscience_data_quality.survey import (
    fetch_ga_survey_metadata,
    filter_surveys,
    fix_survey_geometry,
)
from geoscience_data_quality.vector import (
    analyze_quality_fields,
    get_quality_summary,
)

__all__ = [
    "__version__",
    "analyze_quality_fields",
    "compute_final_score",
    "compute_resolution_score",
    "fetch_ga_survey_metadata",
    "filter_surveys",
    "fix_survey_geometry",
    "get_quality_summary",
    "load_quality_model",
    "rasterize_vector_attribute",
]
