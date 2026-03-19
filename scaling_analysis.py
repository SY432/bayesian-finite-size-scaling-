"""
Bayesian Finite-Size Scaling Analysis for OBC
Author: Shi Yan (施延)
Affiliation: Shijiazhuang Medical College
Address: No.1 Tongxin Road, Lingshou County, Shijiazhuang, Hebei 050062, China
Phone: +86 13187666610
Paper: "Robust Bayesian Inference Protocol for Finite-Size Scaling"
DOI: 10.5281/zenodo.XXXXXXXXX (to be assigned)
"""

import numpy as np
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')


class BayesianScalingAnalyzer:
    """
    Bayesian analyzer for finite-size scaling with open boundary conditions.
    
    Implements weighted least-squares fitting, BIC model comparison,
    and bootstrap confidence intervals.
    
    Note on BIC limitations:
    - BIC is an asymptotic approximation (n -> infinity) that may have
      small-sample bias for n=30. The evidence ratios should be interpreted
      with caution.
    - The model space is limited to four heuristic forms; other corrections
      (e.g., continuous alpha in 1/N^alpha) are not considered.
    """
    
    def __init__(self, N_values, S_A_values, sigma_num, sigma_trunc):
        self.N = np.asarray(N_values, dtype=float)
        self.S_A = np.asarray(S_A_values, dtype=float)
        self.sigma_num = np.asarray(sigma_num, dtype=float)
        self.sigma_trunc = np.asarray(sigma_trunc, dtype=float)
        self.sigma_total = np.sqrt(self.sigma_num**2 + self.sigma_trunc**2)
        self.n_points = len(N_values)
    
    def model_1_over_N(self, N, c, g, A):
        """1/N correction model (PBC-like)."""
        return (c / 6.0) * np.log(N) + g + A / N
    
    def model_1_over_N2(self, N, c, g, B):
        """1/N^2 correction model (BCFT prediction for OBC)."""
        return (c / 6.0) * np.log(N) + g + B / (N ** 2)
    
    def model_1_over_N3(self, N, c, g, C):
        """1/N^3 correction model."""
        return (c / 6.0) * np.log(N) + g + C / (N ** 3)
    
    def model_logN_over_N(self, N, c, g, D):
        """ln(N)/N correction model."""
        return (c / 6.0) * np.log(N) + g + D * np.log(N) / N
    
    def fit_model(self, model_func, p0=None):
        """Fit model with weighted least-squares."""
        if p0 is None:
            p0 = [1.0, 0.5, 0.5]
        
        try:
            popt, pcov = curve_fit(
                model_func, self.N, self.S_A,
                sigma=self.sigma_total, p0=p0,
                absolute_sigma=True, maxfev=10000
            )
            
            residuals = self.S_A - model_func(self.N, *popt)
            chi2_val = np.sum((residuals / self.sigma_total) ** 2)
            n_params = len(popt)
            dof = self.n_points - n_params
            
            # BIC for weighted least squares: BIC = chi2 + k*ln(n)
            # This is the form used in the paper for model comparison
            bic = chi2_val + n_params * np.log(self.n_points)
            aic = 2 * n_params + chi2_val
            aic_c = aic + 2 * n_params * (n_params + 1) / (self.n_points - n_params - 1)
            chi2_reduced = chi2_val / dof if dof > 0 else np.inf
            
            return {
                'params': popt,
                'errors': np.sqrt(np.diag(pcov)),
                'chi2': chi2_val,
                'chi2_reduced': chi2_reduced,
                'dof': dof,
                'bic': bic,
                'aic': aic,
                'aic_c': aic_c,
                'success': True
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def compare_models(self):
        """Compare all candidate models using BIC."""
        models = {
            '1/N': self.model_1_over_N,
            '1/N^2 (BCFT)': self.model_1_over_N2,
            '1/N^3': self.model_1_over_N3,
            'LogN/N': self.model_logN_over_N
        }
        
        results = {}
        for name, model in models.items():
            res = self.fit_model(model)
            if res['success']:
                results[name] = res
        
        # Compute Delta BIC and Bayes factors
        # K_ij = exp(-(BIC_i - BIC_j)/2) is the evidence ratio for model i vs model j
        if results:
            min_bic = min(v['bic'] for v in results.values())
            for name in results:
                delta_bic = results[name]['bic'] - min_bic
                results[name]['delta_bic'] = delta_bic
                # K = exp(-delta_bic/2) is the evidence ratio for this model vs best model
                # For best model, K=1; for worse models, K<1
                results[name]['bayes_factor'] = np.exp(-delta_bic / 2.0)
        
        return results
    
    def print_comparison_table(self, results):
        """Print formatted model comparison table."""
        print("=" * 80)
        print("Bayesian Model Comparison (n={}, k=3)".format(self.n_points))
        print("=" * 80)
        print(f"{'Model':<18} {'chi2/dof':<10} {'BIC':<10} {'DeltaBIC':<10} {'K(vs Best)':<12} {'Evidence'}")
        print("-" * 80)
        
        # Kass & Raftery (1995) evidence grades for K = evidence ratio vs best model
        def evidence_grade(K):
            if K > 1/3: return "Comparable"
            elif K > 1/20: return "Weak"
            elif K > 1/150: return "Moderate"
            else: return "Strongly rejected"
        
        for name, res in sorted(results.items(), key=lambda x: x[1]['bic']):
            K = res['bayes_factor']
            grade = "Best model" if K == 1 else evidence_grade(K)
            print(f"{name:<18} {res['chi2_reduced']:.2f}       "
                  f"{res['bic']:.1f}      {res['delta_bic']:.1f}       "
                  f"{K:.4f}      {grade}")
        print("=" * 80)
    
    def bootstrap_fit(self, model_func, n_bootstrap=1000, seed=42):
        """
        Non-parametric bootstrap resampling for confidence intervals.
        
        Uses direct resampling of (N, S_A, sigma) observation triplets,
        which is the correct approach for heteroscedastic data where
        sigma varies with N.
        
        This differs from residual bootstrap which assumes homoscedasticity.
        """
        np.random.seed(seed)
        params_boot = []
        failed_fits = 0
        
        # Get best-fit parameters for initial guess
        best_fit = self.fit_model(model_func)
        if not best_fit['success']:
            return np.array([])
        
        p0 = best_fit['params']
        
        for _ in range(n_bootstrap):
            # True non-parametric bootstrap: resample observation indices
            indices = np.random.choice(self.n_points, size=self.n_points, replace=True)
            N_resampled = self.N[indices]
            S_resampled = self.S_A[indices]  # Direct resampling of observations
            sigma_resampled = self.sigma_total[indices]
            
            try:
                popt, _ = curve_fit(
                    model_func, N_resampled, S_resampled,
                    sigma=sigma_resampled, p0=p0, absolute_sigma=True,
                    maxfev=10000
                )
                params_boot.append(popt)
            except Exception:
                failed_fits += 1
                continue
        
        if failed_fits > 0:
            print(f"Warning: {failed_fits}/{n_bootstrap} bootstrap fits failed")
        
        return np.array(params_boot)
    
    def compute_confidence_intervals(self, boot_params, confidence=0.95):
        """Compute percentile confidence intervals from bootstrap samples."""
        if len(boot_params) == 0:
            return {}
        
        alpha = 1 - confidence
        lower = alpha / 2 * 100
        upper = (1 - alpha / 2) * 100
        
        ci = {}
        for i in range(boot_params.shape[1]):
            ci[f'param_{i}'] = {
                'mean': np.mean(boot_params[:, i]),
                'std': np.std(boot_params[:, i]),
                'ci_lower': np.percentile(boot_params[:, i], lower),
                'ci_upper': np.percentile(boot_params[:, i], upper)
            }
        return ci
    
    def get_model_function(self, model_name):
        """Get model function by name."""
        models = {
            '1/N': self.model_1_over_N,
            '1/N^2 (BCFT)': self.model_1_over_N2,
            '1/N^3': self.model_1_over_N3,
            'LogN/N': self.model_logN_over_N
        }
        return models.get(model_name)


def load_data_from_csv(filepath):
    """Load DMRG data from CSV file."""
    import pandas as pd
    df = pd.read_csv(filepath)
    return (df['N'].values, df['S_A'].values, 
            df['sigma_num'].values, df['sigma_trunc'].values)


def main():
    """Main analysis workflow."""
    import sys
    
    # Allow custom data file path
    if len(sys.argv) > 1:
        data_file = sys.argv[1]
    else:
        data_file = 'XXZ_OBC_data.csv'
    
    try:
        N, S_A, sigma_num, sigma_trunc = load_data_from_csv(data_file)
    except FileNotFoundError:
        print(f"Error: {data_file} not found.")
        return
    
    analyzer = BayesianScalingAnalyzer(N, S_A, sigma_num, sigma_trunc)
    results = analyzer.compare_models()
    analyzer.print_comparison_table(results)
    
    if results:
        best_name = min(results, key=lambda x: results[x]['bic'])
        best = results[best_name]
        
        print(f"\nBest model: {best_name}")
        print(f"  c = {best['params'][0]:.4f} ± {best['errors'][0]:.4f}")
        print(f"  g = {best['params'][1]:.4f} ± {best['errors'][1]:.4f}")
        
        # Get the correction parameter name based on model
        param_names_map = {
            '1/N': ['c', 'g', 'A'],
            '1/N^2 (BCFT)': ['c', 'g', 'B'],
            '1/N^3': ['c', 'g', 'C'],
            'LogN/N': ['c', 'g', 'D']
        }
        param_names = param_names_map.get(best_name, ['c', 'g', 'param'])
        print(f"  {param_names[2]} = {best['params'][2]:.4f} ± {best['errors'][2]:.4f}")
        
        # Bootstrap CI for the best model (not hardcoded to 1/N^2)
        best_model_func = analyzer.get_model_function(best_name)
        boot_params = analyzer.bootstrap_fit(best_model_func, n_bootstrap=1000)
        
        if len(boot_params) > 0:
            ci = analyzer.compute_confidence_intervals(boot_params)
            
            print(f"\nBootstrap 95% Confidence Intervals (n={len(boot_params)} successful fits):")
            for i, name in enumerate(param_names):
                if f'param_{i}' in ci:
                    ci_l = ci[f'param_{i}']['ci_lower']
                    ci_u = ci[f'param_{i}']['ci_upper']
                    print(f"  {name}: [{ci_l:.4f}, {ci_u:.4f}]")
        else:
            print("\nWarning: Bootstrap failed to produce valid samples")
    
    return results


if __name__ == "__main__":
    main()
