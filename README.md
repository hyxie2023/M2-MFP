# M<sup>2</sup>-MFP
This repository contains the core implementations for "M<sup>2</sup>-MFP: A Multi-Scale and Multi-Level Memory Failure Prediction Framework for Reliable Cloud Infrastructure" as presented in our paper.

**Note:** The complete dataset and code will be provided in future updates.

## Overview

The framework consists of the following key modules:

- **Binary Spatial Feature Extractor (BSFE)** (implemented in [sec4.2_BSFE.py](core_code/sec4.2_BSFE.py)):
  - Designed to capture high-order features that contain potential fault representations from the spatial information of Correctable Errors (CEs).

- **Time-patch Scale Prediction Module** (implemented in [sec4.3_time_patch_module.py](core_code/sec4.3_time_patch_module.py)):
  - During training, historical CEs are aggregated using a sliding window. Multiple levels of BSFE are then applied to extract multi-level spatial features from time-patch scale data.

- **Time-point Scale Prediction Module** (implemented in [sec4.4_time_point_module.py](core_code/sec4.4_time_point_module.py)):
  - For time-point scale data, BSFE extracts bit-level high-order features from all CEs of each DIMM. A customized decision tree is then trained to generate a rule set for fault prediction.

During inference, CEs are batch-processed through the Time-patch Scale Prediction Module and streamed into the Time-point Scale Prediction Module. The final fault prediction results are obtained by merging the outputs from both modules.

## Data Availability

Currently, a subset of the dataset is available, which comprises 7 months of Memory Log data and 5 months of Memory Failure record data. A complete dataset and additional code implementations will be released in the future.

### Dataset Description

Our dataset comprises log data from Intel Purley and Intel Whitley DIMMs, collected from over 70,000 DIMMs between January and September 2024. For each month, we computed:
- The number of DIMMs containing Correctable Errors (CEs).
- The number of DIMMs that experienced failures.
- The total number of CEs recorded.

Below is the summary table of DIMM and CE counts from our dataset:

| Month   | Intel Purley DIMM count | Intel Purley Fault DIMM count | Intel Purley CE count | Intel cascade DIMM count | Intel cascade Fault DIMM count | Intel cascade CE count | All DIMM count | All Fault DIMM count | All CE count  |
|---------|-------------------------:|------------------------------:|----------------------:|-------------------------:|------------------------------:|----------------------:|---------------:|---------------------:|--------------:|
| 2024-01 | 26450                  | 163                           | 102423615            | 1466                   | 15                           | 1843056              | 27916         | 178                  | 104266671     |
| 2024-02 | 24682                  | 133                           | 97336381             | 1682                   | 8                            | 1763660              | 26364         | 141                  | 99100041      |
| 2024-03 | 26919                  | 174                           | 106745981            | 1948                   | 16                           | 1682764              | 28867         | 190                  | 108428745     |
| 2024-04 | 29776                  | 144                           | 107168332            | 2067                   | 17                           | 1947564              | 31843         | 161                  | 109115896     |
| 2024-05 | 29046                  | 155                           | 119407409            | 2191                   | 12                           | 2494964              | 31237         | 167                  | 121902373     |
| 2024-06 | 32610                  | [REDACTED]                           | 118064609            | 2270                   | [REDACTED]                           | 2580721              | 34880         | [REDACTED]                  | 120645330     |
| 2024-07 | 37853                  | [REDACTED]                           | 148794345            | 2564                   | [REDACTED]                           | 3604404              | 40417         | [REDACTED]                  | 152398749     |
| 2024-08 | [REDACTED]             | [REDACTED]                    | [REDACTED]           | [REDACTED]             | [REDACTED]                   | [REDACTED]           | [REDACTED]    | [REDACTED]           | [REDACTED]    |
| 2024-09 | [REDACTED]             | [REDACTED]                    | [REDACTED]           | [REDACTED]             | [REDACTED]                   | [REDACTED]           | [REDACTED]    | [REDACTED]           | [REDACTED]    |
| Total   | [REDACTED]                  | [REDACTED]                          | [REDACTED]           | [REDACTED]                   | [REDACTED]                          | [REDACTED]             | [REDACTED]         | [REDACTED]                 | [REDACTED]    |


You can access the current dataset via [SmartMem](https://www.codabench.org/competitions/3586/).

## License

This project is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) License. See [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) for details.
