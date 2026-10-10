# \(f(T,B)\) cosmology MCMC

Used in R. Briffa, C. Escamilla-Rivera, J. Levi Said, and J. Mifsud, *\(f(T,B)\) gravity in the late Universe in the context of local measurements*, Phys. Dark Univ. **39**, 101153 (2023), [arXiv:2205.13560](https://arxiv.org/abs/2205.13560).

Late-time MCMC for three \(f(T,B)\) models and a ΛCDM comparison, with cosmic chronometers, Pantheon, and BAO.

Each model keeps its own solver and likelihood. \(\theta\) is not the same for every model: ΛCDM is \(H_0\), \(\Omega_m\), \(M\), and \(f_1\), \(f_2\), \(f_3\) also sample \(q\). These scripts use CC + Pantheon + BAO, with no \(H_0\) prior. The R21 and F21 prior functions are in each model’s `likelihoods.py`.

## Models

| Package | Paper | Lagrangian | Run |
| --- | --- | --- | --- |
| `models/f1/` | \(f_1\) power-law | \(\mathcal{F}_1(B)=\alpha_1(-B)^{p_1}\) | `python scripts/run_f1_cc_sn_bao.py` |
| `models/f2/` | \(f_2\) exponential | \(\mathcal{F}_2(B)=\alpha_2\exp[-p_2\sqrt{B/B_0}]\) | `python scripts/run_f2_cc_sn_bao.py` |
| `models/f3/` | \(f_3\) logarithmic | \(\mathcal{F}_3=\alpha_3\ln[p_3 B_0/B]\) | `python scripts/run_f3_cc_sn_bao.py` |
| `models/lcdm/` | Appendix A | ΛCDM | `python scripts/run_lcdm_cc_sn_bao.py` |

The \(f(T,B)\) solvers integrate the Friedmann equation with `mpmath.odefun`. ΛCDM is analytic. These runs are slower than the later \(f(T)\) Newton solvers. Lower `niter` in a script for a short test.

## Setup

```bash
python -m pip install -r requirements.txt
```

Place the survey files listed in `data/README.md` into `data/`. This paper uses the original Pantheon compilation, not Pantheon+.

## Citation

If you use this code, please cite

R. Briffa, C. Escamilla-Rivera, J. Levi Said, and J. Mifsud, *\(f(T,B)\) gravity in the late Universe in the context of local measurements*, Phys. Dark Univ. **39**, 101153 (2023), [arXiv:2205.13560](https://arxiv.org/abs/2205.13560), [doi:10.1016/j.dark.2022.101153](https://doi.org/10.1016/j.dark.2022.101153).

The survey files remain the property of those collaborations.
