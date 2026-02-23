# Forest Edge Mapping and Analysis

## Overview

This repository contains code and workflows for mapping and analyzing forest edge dynamics using remote sensing data. The project focuses on quantifying spatial and temporal patterns of forest edges, supporting investigations into forest fragmentation, landscape structure, and associated ecological processes.

The tools provided here are designed to support reproducible research and can be adapted for regional or global forest monitoring applications.

---

## Objectives

* Generate forest edge maps from time-series land cover data
* Quantify changes in forest edge extent and configuration
* Support analyses of fragmentation and landscape dynamics
* Provide reproducible workflows for scientific studies

---

## Repository Structure

* `code/` — Scripts for forest edge detection, processing, and analysis
* `data_processing/` — Utilities for preparing input datasets
* `analysis/` — Scripts for statistical analysis and visualization
* `figures/` — Example outputs and figure generation workflows
* `docs/` — Supporting documentation

*(Adjust folders if your repo structure differs.)*

---

## Methods Summary

Forest edges are derived from land cover datasets using spatial neighborhood analysis to identify transitions between forest and non-forest classes. Time-series processing enables tracking of edge dynamics across years, allowing assessment of fragmentation trends and landscape change.

The workflows emphasize:

* Consistent classification rules
* Scalable processing for large datasets
* Transparent assumptions
* Reproducibility

---

## Data Availability

All data are available in the main text or the supplementary materials.

The forest edge maps by chunks from 2000 to 2020 with 5-year gaps, along with the validation samples, are available via the following link:

👉 https://gofile.me/7g1zu/enA1L3vRY

## Forest Edge Products

### Product Description

This repository is accompanied by gridded forest edge products that quantify both forest area and forest edge length at 0.01° spatial resolution. These datasets are designed to support analyses of forest fragmentation, landscape structure, and temporal dynamics of forest boundaries.

The products provide spatially explicit metrics that can be used to investigate how forest extent and edge configuration change over time, enabling applications in ecology, conservation planning, carbon accounting, and land-use change research.

---

### Data Access

All 0.01° forest area and forest edge length datasets are available via Google Drive:

https://drive.google.com/drive/folders/1DReDApISiudMYr574RWRkjwhWQJdS11e?usp=sharing

These files include gridded summaries that can be directly used for regional or large-scale analyses.

---

### Metrics Included

* **Forest Area**
  Total forest area within each 0.01° grid cell.

* **Forest Edge Length**
  Total length of forest–nonforest boundaries within each grid cell, representing the degree of landscape fragmentation and boundary complexity.

Together, these metrics provide complementary information on both the extent and configuration of forest landscapes.

---

### Forest Definition

Forest area and edge metrics are derived using the land-cover definition from:

**Peter Potapov’s Global Land Cover and Land Use Change Dataset (2000–2020)**

In this dataset, forest is defined as areas where tree height is greater than or equal to 5 meters at the Landsat pixel scale. This definition ensures consistency with widely used global forest monitoring frameworks and allows for comparison with other studies.

---

### Intended Uses

These products are suitable for:

* Quantifying forest fragmentation patterns
* Assessing habitat connectivity
* Studying biodiversity responses to edge effects
* Evaluating carbon and ecosystem dynamics
* Supporting conservation and land management decisions
* Large-scale ecological modeling

---

### Notes on Interpretation

Forest edge length reflects spatial configuration rather than forest condition. Changes in edge metrics may arise from deforestation, forest regrowth, disturbance, or land-use conversion, and should be interpreted alongside contextual information when possible.

Users should ensure consistency in spatial resolution and projection when integrating these datasets with other geospatial products.

---


For further information, please contact:

**Dr. Min Chen**
📧 [min.chen@wisc.edu](mailto:min.chen@wisc.edu)

---

## Requirements

Typical dependencies include:

* Python (3.8+)
* numpy
* pandas
* rasterio / GDAL
* geopandas
* matplotlib
* scipy

See individual scripts for additional requirements.

---

## Usage

1. Prepare input land cover datasets according to instructions in `data_processing/`.
2. Run edge detection scripts in `code/`.
3. Use analysis scripts to compute metrics and generate figures.

Example:

```
python run_edge_detection.py
python analyze_edge_dynamics.py
```

---

## Reproducibility Notes

* Ensure consistent coordinate reference systems across datasets.
* Verify classification schemes match expected forest definitions.
* Large datasets may require high-performance computing resources.

---

## Citation

If you use this repository, please cite the associated study.

---

## Contributing

Contributions, suggestions, and issue reports are welcome. Please open an issue or submit a pull request.

---

## License

Specify license information here (e.g., MIT, GPL, or custom academic use license).

---

## Contact

For questions about the methodology or data, please contact the repository maintainers (Hangkai You) or Dr. Min Chen.
