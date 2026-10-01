# NASA-C-MAPSS-RUL-Prediction

A hybrid deep learning framework for **Remaining Useful Life (RUL) prediction of turbofan engines**, evaluated across all four NASA C-MAPSS benchmark subsets: **FD001, FD002, FD003, and FD004**.

This project investigates the combination of **multi-scale convolutional feature extraction, self-attention, and bidirectional recurrent modeling** for multivariate time-series degradation analysis.

The proposed architecture, **SA-CNN-GRU-DE**, combines a dual-branch convolutional encoder, multi-head self-attention, and a bidirectional GRU into a unified RUL prediction framework.

Beyond benchmark performance, the study also examines **component-level contributions, training stability, cross-dataset generalization, statistical significance, feature attribution, and predictive uncertainty**.

The objective is not simply to obtain a lower prediction error, but to investigate **why the architecture performs as it does, which components contribute to that performance, and how well the learned representation transfers to unseen operating conditions**.

---

## Architecture

**SA-CNN-GRU-DE** = **S**elf-**A**ttention **C**onvolutional **N**eural **N**etwork – **G**ated **R**ecurrent **U**nit with **D**ual **E**ncoder

The proposed architecture consists of three main stages:

### 1. Dual-Branch Convolutional Encoder

Two parallel Conv1D branches with different kernel sizes are used to capture temporal patterns at multiple scales:

* **Kernel size 3** — captures relatively local degradation patterns
* **Kernel size 7** — captures broader temporal degradation patterns

The representations from both branches are subsequently fused before being passed to the attention module.

### 2. Multi-Head Self-Attention

The fused convolutional representation is processed using multi-head self-attention.

This allows the model to learn dependencies between different positions within the temporal window and assign different levels of importance to the encoded sequence elements.

### 3. Bidirectional GRU

The attention-enhanced representation is passed through a **2-layer bidirectional GRU**, followed by a dense regression head that produces the final RUL estimate.

The overall architecture can be summarized as:

```text
                    Input Sensor Window
                            │
                 ┌──────────┴──────────┐
                 │                     │
             Conv1D (k=3)         Conv1D (k=7)
                 │                     │
                 └──────────┬──────────┘
                            │
                     Feature Fusion
                            │
                  Multi-Head Attention
                            │
                       2-Layer BiGRU
                            │
                    Dense Regression Head
                            │
                       Predicted RUL
```

The motivation is to combine complementary modeling capabilities:

* **CNNs** for local and multi-scale temporal patterns
* **Self-attention** for sequence-wide dependencies
* **Bidirectional GRUs** for sequential degradation dynamics

---

# Research Questions

The experiments are designed around several questions:

1. **Does the proposed hybrid architecture improve RUL prediction compared with standard deep learning baselines?**
2. **Which architectural components contribute to the observed performance?**
3. **Are the observed improvements consistent across independent training runs?**
4. **Does improved in-distribution performance translate to unseen operating conditions?**
5. **Can the model provide useful predictive uncertainty in addition to point estimates?**
6. **Which sensor features and temporal regions contribute most strongly to the predictions?**

This framing is important because a lower RMSE alone does not necessarily explain *why* a model performs better or whether the improvement is robust.

---

# Dataset

Experiments are conducted on all four NASA C-MAPSS turbofan degradation subsets:

* **FD001**
* **FD002**
* **FD003**
* **FD004**

The preprocessing pipeline consists of:

* RUL clipping at **125 cycles**
* RobustScaler normalization
* K-means-based per-condition normalization
* Exponential smoothing
* 30-cycle sliding-window construction

The four subsets provide different operating conditions and degradation characteristics, making them useful for evaluating both standard in-distribution performance and cross-dataset generalization.

---

# Preventing Data Leakage

A major consideration in time-series RUL prediction is preventing information from the same engine from appearing in both training and validation sets.

For this reason, the train/validation split is performed **at the engine level before window generation**.

Only after the engine-level split are the 30-cycle sliding windows constructed.

This prevents overlapping or highly correlated windows originating from the same engine from being distributed across different splits.

---

# Experimental Design

The proposed model is evaluated against several commonly used deep learning architectures:

* LSTM
* BiLSTM
* CNN-LSTM
* Transformer without positional encoding

Each configuration is trained using **5 independent random seeds**.

Rather than relying on a single training run, performance is summarized as:

**mean ± standard deviation across the 5 seeds**

This provides an indication of both average performance and sensitivity to initialization and training stochasticity.

The same five seeds are used across the compared configurations where applicable, enabling paired statistical comparisons.

---

# Evaluation Metrics

Model performance is evaluated using four complementary metrics:

### RMSE

Root Mean Squared Error measures the magnitude of prediction errors while placing greater emphasis on larger errors.

### MAE

Mean Absolute Error provides a more direct measure of the average absolute prediction error.

### R²

The coefficient of determination measures the proportion of variance in the target RUL values explained by the predictions.

### NASA Scoring Function

The NASA scoring function, or **S-Score**, is included to reflect the asymmetric penalty structure commonly used for RUL prediction on the C-MAPSS benchmark.

Using multiple metrics provides a broader evaluation than relying on RMSE alone.

---

# Main Results

Across the four C-MAPSS subsets, the full **SA-CNN-GRU-DE** configuration achieved the lowest average in-distribution RMSE among the evaluated baselines and ablation variants.

The experiments were conducted using **5 independent training seeds per configuration**, allowing the reported results to account for training variability.

An important observation, however, is that the in-distribution advantage of the full architecture **does not uniformly transfer to unseen operating conditions**.

This suggests that strong performance under the standard benchmark setting should not automatically be interpreted as evidence of stronger domain generalization.

The cross-dataset experiments therefore provide an important complementary perspective on the model's behavior.

---

# Ablation Study

To investigate the contribution of individual architectural components, a controlled ablation study is conducted using variants **M1–M4**.

The study examines the contribution of:

* the dual convolutional encoder
* self-attention
* bidirectional recurrence
* recurrent cell type

Each ablation modifies a specific architectural component while keeping the remaining experimental configuration fixed.

Where applicable, the ablation variants are matched to the full model's **parameter budget**.

This reduces the possibility that observed performance differences are simply caused by differences in model capacity.

The objective is therefore to isolate the contribution of individual architectural choices rather than simply comparing models of different sizes.

---

# Statistical Significance

Because neural network performance can vary across training runs, statistical comparisons are performed using the **5 shared random seeds**.

Two complementary statistical tests are used:

* **Paired t-test**
* **Wilcoxon signed-rank test**

Using both parametric and non-parametric tests provides complementary evidence regarding whether observed differences across configurations are consistent across training seeds.

The statistical tests are treated as supporting evidence alongside the actual performance distributions, rather than as a replacement for reporting model variability.

---

# Cross-Dataset Generalization

In addition to standard within-dataset evaluation, the project investigates **cross-dataset generalization**.

Models trained on one C-MAPSS subset are evaluated on another to examine how learned degradation representations behave when operating conditions and degradation characteristics change.

One of the key observations is that the performance advantage observed under in-distribution evaluation is **not consistently preserved under distribution shift**.

This highlights an important distinction between:

> **performance on the benchmark distribution**

and

> **generalization to previously unseen operating conditions.**

For predictive-maintenance applications, this distinction is particularly relevant because deployment conditions may differ from those represented during model development.

---

# Uncertainty Quantification

RUL prediction involves estimating a future failure horizon, so uncertainty is an important consideration in addition to point prediction.

Two approaches are investigated.

### MC-Dropout

Dropout remains active during inference and multiple stochastic forward passes are used to approximate predictive uncertainty.

This provides an empirical estimate of the variability of the model's predictions.

### Heteroscedastic Regression

A heteroscedastic prediction head is used to estimate both the expected RUL and an input-dependent variance.

This allows the model to represent situations in which some observations are intrinsically more difficult to predict than others.

The resulting framework therefore moves beyond a deterministic prediction such as:

```text
Predicted RUL = 42 cycles
```

towards:

```text
Predicted RUL = 42 cycles
+ an estimate of predictive uncertainty
```

---

# Explainability

The project also includes feature-attribution analysis using **Captum**.

The objective is to investigate which sensor variables and temporal regions contribute most strongly to the model's RUL predictions.

This provides an additional perspective on the learned representation and can help assess whether the model is relying on meaningful degradation-related signals.

Explainability is treated as a complementary analysis rather than as direct evidence of causal relationships between sensor variables and engine degradation.

---

# Reproducibility

To reduce dependence on a single stochastic training run, each configuration is evaluated using **5 independent random seeds**.

The experiments retain, where applicable:

* per-seed metrics
* mean performance
* standard deviation
* statistical comparison results
* generated plots
* ablation results
* cross-dataset results
* uncertainty estimates

This makes it possible to inspect both aggregate performance and the variability underlying the reported results.

---

# Repository Structure

```text
cmapss-prediction.ipynb
│
├── Data preprocessing
├── SA-CNN-GRU-DE
├── Baseline models
├── Multi-seed experiments
├── Performance comparison
├── Statistical analysis
└── Full experimental pipeline
├── Ablation study
├── Cross-dataset generalization
├── Feature attribution
└── Uncertainty quantification
```

The two notebooks are deliberately structured so that they can be executed **in parallel**, for example using two independent Kaggle sessions.

The first notebook covers the main model and baseline experiments, while the second continues with the later experimental sections.

This organization allows the computational workload to be distributed across separate sessions without changing the experimental methodology.

---

# Running the Project

## 1. Download C-MAPSS

Download the NASA C-MAPSS turbofan degradation dataset and update the dataset path in the notebook configuration cell.

## 2. Install Dependencies

```bash
pip install torch numpy pandas scikit-learn scipy matplotlib captum
```

`captum` is required only for the feature-attribution section.

## 3. Run the Notebooks

For the complete pipeline:

```text
sacnn-pred_multiseed.ipynb
```

Alternatively, the notebooks can be executed simultaneously:

```text
sacnn-pred_multiseed.ipynb
sacnn-pred_multiseed_ABLATION-ONWARD.ipynb
```

The second notebook is intended to begin from the later experimental sections and can therefore be used alongside the main notebook to parallelize computation.

## 4. Inspect the Results

Summary tables, plots, and per-seed metrics are saved to:

```text
./ablation_results/
```

and are also displayed within the notebooks.

---

# Project Outputs

The experimental pipeline produces several categories of results, including:

* Baseline comparison tables
* Multi-seed performance statistics
* Ablation results
* Statistical significance tests
* Cross-dataset generalization results
* Feature-attribution visualizations
* MC-Dropout uncertainty estimates
* Heteroscedastic uncertainty estimates
* Per-seed metrics and plots

The intention is to make the evaluation process sufficiently transparent to inspect not only the final aggregate results but also the variation between individual training runs.

---

# Key Findings

The experiments provide several observations about the proposed architecture and the broader RUL prediction problem:

* **SA-CNN-GRU-DE achieves strong in-distribution performance** across the four C-MAPSS subsets evaluated.
* The combination of **multi-scale convolution, self-attention, and bidirectional recurrent modeling** provides a useful framework for temporal degradation modeling.
* The controlled ablation study helps isolate the contribution of individual architectural components.
* Evaluating **5 independent training seeds** provides a more reliable estimate of performance than relying on a single run.
* The observed in-distribution advantage **does not uniformly transfer to cross-dataset settings**, highlighting the effect of distribution shift.
* **Uncertainty estimation** provides additional information beyond a single point RUL prediction.
* **Feature attribution** provides a way to inspect which sensor variables and temporal regions influence the model's predictions.

Overall, the project is intended not simply as another C-MAPSS benchmark implementation, but as an investigation into **architecture, robustness, generalization, explainability, and uncertainty in deep learning-based RUL prediction**.

---

# Limitations and Considerations

While the results are encouraging, several considerations are important when interpreting them.

First, C-MAPSS is a benchmark dataset and does not fully reproduce the complexity of real-world predictive-maintenance environments.

Second, cross-dataset experiments provide only a limited approximation of real deployment distribution shift.

Third, uncertainty estimates produced by MC-Dropout and heteroscedastic regression should be interpreted as model-based uncertainty estimates rather than guaranteed calibrated probabilities.

Finally, statistical comparisons across five seeds provide useful evidence about training variability, but the relatively small number of seeds means that statistical conclusions should still be interpreted cautiously.

These considerations are part of the reason the project evaluates the model from multiple perspectives rather than relying on a single performance metric.

---

# License

This project is licensed under the **MIT License**.

See the [`LICENSE`](LICENSE) file for the full license text.
