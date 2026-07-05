# Emission-Aware PtX Scheduling

This repository contains the code used for emission-aware scheduling and product carbon intensity tracking in a grid-connected multi-product Power-to-X plant producing H₂, NH₃, and MeOH.

The workflow includes:

* day-ahead electricity generation forecasting,
* construction of AEF and MEF-based emission signals,
* rolling horizon MIQCP scheduling,
* ex-post product carbon intensity propagation,

The study uses publicly available input data from ENTSO-E, MIBGAS, Ember, and GFS, together with process parameters and assumptions reported in the manuscript and Supplementary Information. The CSV files used in the code will be available upon request, even though all data is publicly available. The workflow was implemented in Google Colab.

## Paper
Norouzi et al, 2026, DOI:
"Tracking Product Carbon Intensity in Power-to-X Plants: An Optimization Framework for Emission-Aware Hydrogen, Ammonia, and Methanol Production"
