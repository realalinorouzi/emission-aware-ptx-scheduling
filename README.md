Emission-Aware Power-to-X Scheduling and Product Carbon Intensity Tracking
This repository contains the computational workflow developed for the study:
Ali Norouzi and Erdal Aydin
Hourly Carbon Intensity of Sold Hydrogen, Ammonia, and Methanol under Emission-Aware Scheduling
The study evaluates how alternative day-ahead electricity-emission signals affect the operation of a grid-connected, multi-product Power-to-X (PtX) plant and how the resulting carbon burden propagates through hydrogen (H₂), ammonia (NH₃), and methanol (MeOH) production, storage, conversion, and final product withdrawal.
Workflow
The repository contains code for:
1. day-ahead electricity-generation forecasting for the Spanish power system;
2. construction of the average emission factor (AEF), MSDR Soft MEF, and UpRamp MEF signals;
3. rolling-horizon mixed-integer quadratically constrained programming (MIQCP) scheduling of the PtX plant;
4. ex-post propagation of assigned carbon through production, storage, synthesis, ammonia cracking, and withdrawal; and
5. analysis of annual and hourly sold-product carbon intensity.
The main workflow is organized so that the processed outputs from the forecasting and emission-signal stages can be used directly as inputs to the optimization stage.
Reproduction Materials
To support reproduction of the reported calculations, the repository includes:
- source notebooks/code for forecasting, emission-signal construction, optimization, carbon accounting, and post-processing;
- trained Keras ANN model files (.keras) for the technology-specific electricity-generation forecasting models;
- processed 2024 model-ready input files used directly by the optimization, including:
  - hourly H₂, NH₃, and MeOH demand profiles;
  - hourly local PV and wind availability;
  - hourly day-ahead grid electricity prices;
  - hourly AEF, Soft MEF, and UpRamp MEF signals; and
  - supporting utility/input files required by the rolling-horizon model;
- model parameters and assumptions documented in the manuscript, Supplementary Information, and Appendices.
Providing the trained ANN files and the final 2024 forecast/emission-signal outputs allows the downstream workflow to be reproduced without retraining the forecasting models.
Data Sources
The underlying public data used to construct the processed inputs were obtained from:
- ENTSO-E Transparency Platform — electricity generation, load and generation forecasts, wind and solar forecasts, scheduled cross-border exchanges, and day-ahead electricity prices;
- MIBGAS — day-ahead natural-gas prices;
- Ember — technology-specific electricity emission factors;
- Global Forecast System (GFS) — meteorological inputs used to construct local PV and wind availability.
The corresponding preprocessing procedures, feature construction, technology aggregation, demand construction, and model assumptions are described in the manuscript and Supplementary Information.
Forecasting Models
Separate feed-forward artificial neural networks (ANNs) were trained for the electricity-generation technology groups used in the study. The forecasting structure uses a 24-hour historical context and day-ahead exogenous information to predict the following 24 hours.
The repository provides the trained .keras model files used to generate the reported 2024 forecasts. ANN architecture, hyperparameters, train/validation/test partitioning, preprocessing, and out-of-sample performance are reported in the Supplementary Information.
Saved-model environment: Keras 3.10.0.
Optimization Environment
The reported rolling-horizon MIQCP calculations were implemented in Python and solved using:
- Gurobi Optimizer: 13.0.1
- Execution environment: Google Colab CPU runtime
- Operating system: Ubuntu 22.04.5 LTS
- Processor: Intel Xeon @ 2.20 GHz
- CPU allocation: 1 physical core, 2 logical processors
- Solver threads: up to 2
- Global time limit: 350 s per daily optimization
- Target second-priority MIP gap: 0.01
The annual study horizon contains 366 sequential daily optimization problems for each rolling-horizon model trajectory.
Reproducing the Optimization Stage
The processed 2024 input files supplied in this repository are the model-ready inputs used in the reported optimization runs. A user interested primarily in reproducing the scheduling and carbon-accounting results can therefore start from these files rather than rebuilding the upstream public datasets.
At a high level:
1. load the processed 2024 demand, renewable-availability, electricity-price, and emission-signal files;
2. load the process and storage parameters used by the optimization model;
3. run the rolling-horizon M0/M1/M2 optimization workflow;
4. retain the sequential carry-over states between daily solves; and
5. run the ex-post carbon-propagation and sold-product CI calculations on the solved hourly trajectories.
For the complete mathematical formulation and parameter definitions, refer to the manuscript Supplementary Information and Appendices.
Data and Code Availability
The code and reproduction materials are provided in this repository:
https://github.com/realalinorouzi/emission-aware-ptx-scheduling
The raw source datasets remain available from their respective public providers listed above.
Related Manuscript
Ali Norouzi and Erdal Aydin
Hourly Carbon Intensity of Sold Hydrogen, Ammonia, and Methanol under Emission-Aware Scheduling
The complete journal citation and DOI will be added following publication.
Contact
For questions regarding the code or reproduction materials:
- Ali Norouzi: anorouzi24@ku.edu.tr
- Alternative email: realalinorouzi@gmail.com
