# Emission-Aware Power-to-X Scheduling and Product Carbon Intensity Tracking

This repository contains the code developed for emission-aware scheduling and dynamic product carbon intensity tracking in a grid-connected, multi-product Power-to-X plant producing hydrogen (H₂), ammonia (NH₃), and methanol (MeOH).

## Workflow

The repository includes code for:

- day-ahead electricity-generation forecasting;
- construction of average emission factor (AEF) and marginal emission factor (MEF) proxy signals;
- rolling-horizon mixed-integer quadratically constrained programming (MIQCP) scheduling;
- ex-post propagation of carbon mass through production, storage, conversion, ammonia cracking, and product withdrawal;
- analysis of annual and hourly sold-product carbon intensity.

## Data

The study uses publicly available data obtained from:

- ENTSO-E Transparency Platform;
- MIBGAS;
- Ember;
- Global Forecast System (GFS).

Process parameters, model assumptions, and data-processing procedures are reported in the manuscript and Supplementary Information.

The processed CSV files used directly by the notebooks are not currently included in the repository. They can be provided upon reasonable request. The corresponding raw data are available from the public sources listed above.

## Computational Environment

The workflow was developed and executed in Google Colab. The notebooks are organized according to the main stages of the forecasting, emission-signal construction, optimization, and carbon-accounting workflow.

## Related Manuscript

**Ali Norouzi and Erdal Aydin**

*Tracking Dynamic Carbon Intensity from Electricity Supply to Sold Hydrogen, Ammonia, and Methanol under Emission-Aware Power-to-X Scheduling*

The complete journal citation and DOI will be added following publication.

## Contact

For questions regarding the code or processed input data:

- Ali Norouzi: anorouzi24@ku.edu.tr
- Alternative email: realalinorouzi@gmail.com
