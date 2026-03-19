# Bayesian Finite-Size Scaling Analysis

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXXXX)

**Author:** Shi Yan (施延)
- Affiliation: Shijiazhuang Medical College
- Address: No.1 Tongxin Road, Lingshou County, Shijiazhuang, Hebei 050062, China
- Phone: +86 13187666610
- Email: 15891843770@163.com
  
**Paper:** "Robust Bayesian Inference Protocol for Finite-Size Scaling: Quantifying Evidence for Conformal Field Theory Corrections in Quantum Many-Body Systems"

## Overview

This repository provides the data and analysis code for the paper on Bayesian model selection for finite-size scaling in quantum critical systems. The framework uses DMRG simulation data from the critical XXZ chain with open boundary conditions to validate the boundary CFT prediction of 1/N² corrections to entanglement entropy.

## Key Results

- **Central charge:** c = 1.008 (95% CI: [0.976, 1.040])
- **Boundary entropy:** g = 0.352 (95% CI: [0.330, 0.374])
- **Evidence ratio:** K ≈ 1800:1 favoring 1/N² over 1/N corrections
- **Sample size:** n = 30 data points (N = 16 to 300)

## Repository Structure

```
.
├── data/
│   └── XXZ_OBC_data.csv          # Raw DMRG data (Table S1)
├── src/
│   └── scaling_analysis.py       # Core analysis class
├── examples/
│   └── XXZ_chain_analysis.ipynb  # Jupyter notebook for reproduction
├── README.md                      # This file
└── LICENSE                        # MIT License
```

## Data Description

### XXZ_OBC_data.csv

The CSV file contains entanglement entropy data from DMRG simulations of the spin-1/2 XXZ chain at criticality (Δ=1) with open boundary conditions.

| Column | Description |
|--------|-------------|
| `N` | System size (chain length) |
| `S_A` | von Neumann entanglement entropy of half-chain |
| `sigma_num` | Numerical uncertainty (DMRG convergence) |
| `sigma_trunc` | Truncation error (finite bond dimension χ=1600) |
| `sigma_total` | Combined uncertainty (quadrature sum) |

**Simulation parameters:**
- Maximum bond dimension: χ_max = 1600
- Energy variance convergence: < 10⁻¹⁰
- Truncation error: < 10⁻⁸
- Software: ITensor (v3.2.0)

## Quick Start

### Requirements

```bash
pip install numpy scipy pandas matplotlib jupyter
```

### Basic Usage

```python
from scaling_analysis import BayesianScalingAnalyzer, load_data_from_csv

# Load data
N, S_A, sigma_num, sigma_trunc = load_data_from_csv('data/XXZ_OBC_data.csv')

# Initialize analyzer
analyzer = BayesianScalingAnalyzer(N, S_A, sigma_num, sigma_trunc)

# Compare models
results = analyzer.compare_models()
analyzer.print_comparison_table(results)

# Bootstrap confidence intervals
boot_params = analyzer.bootstrap_fit(analyzer.model_1_over_N2, n_bootstrap=1000)
ci = analyzer.compute_confidence_intervals(boot_params)
```

### Reproduce Paper Results

```bash
python src/scaling_analysis.py
```

Or open `examples/XXZ_chain_analysis.ipynb` in Jupyter Notebook for interactive analysis.

## Methodology

### 1. Finite-Size Scaling Ansatz (OBC)

For open boundary conditions, boundary CFT predicts:

```
S_A(N) = (c/6) ln(N) + g + B/N² + O(N⁻⁴)
```

where:
- `c` = central charge (universal)
- `g` = boundary entropy (Affleck-Ludwig)
- `B` = correction amplitude

### 2. Model Comparison

We compare four competing correction forms:
- 1/N (PBC-like)
- 1/N² (BCFT prediction for OBC)
- 1/N³ (higher-order)
- ln(N)/N (logarithmic)

### 3. Bayesian Evidence

Bayes factor computed as:
```
K = exp(-ΔBIC/2)
```

where BIC = χ² + k ln(n) is the Bayesian Information Criterion, and ΔBIC = BIC_i - BIC_min.

**Note on BIC limitations:**
- BIC is an asymptotic approximation valid for large n; for n=30, small-sample bias may affect evidence ratios
- The model space is limited to four heuristic forms; other corrections (e.g., continuous α in 1/N^α) are not considered
- Evidence ratios should be interpreted with appropriate caution

### 4. Uncertainty Quantification

- **Fisher matrix:** Asymptotic standard errors
- **Bootstrap:** Non-parametric 95% confidence intervals (n_bootstrap=1000)

## Citation

If you use this code or data, please cite:

```bibtex
@article{yan2024bayesian,
  title={Robust Bayesian Inference Protocol for Finite-Size Scaling: 
         Quantifying Evidence for Conformal Field Theory Corrections 
         in Quantum Many-Body Systems},
  author={Yan, Shi},
  journal={Phys. Rev. Res.},
  year={2024},
  publisher={American Physical Society}
}
```

## Data Availability

- **GitHub:** https://github.com/shiyan/bayesian-fss
- **Zenodo:** https://doi.org/10.5281/zenodo.XXXXXXXXX

## License

- **Code:** MIT License
- **Data:** CC-BY-4.0

## Contact

For questions or issues, please open a GitHub issue or contact:
- **Name:** Shi Yan (施延)
- **Affiliation:** Shijiazhuang Medical College
- **Address:** No.1 Tongxin Road, Lingshou County, Shijiazhuang, Hebei 050062, China
- **Phone:** +86 13187666610
- **Email:** yanshi@stumail.smc.edu.cn

## Acknowledgments

DMRG simulations were performed on the HPC cluster at Shijiazhuang Medical College using ITensor (http://itensor.org/).
