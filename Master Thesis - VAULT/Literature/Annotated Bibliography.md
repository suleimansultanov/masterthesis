# Annotated Bibliography

**Thesis:** *Adaptive Hybrid Kernel SVMs: Dynamic Feature Weighting for Noisy, High-Dimensional IoT Healthcare Data*

> Source filter applied: only designated peer-reviewed sources (IEEE, ACM, ScienceDirect/Elsevier, Springer, SAGE, MDPI, Nature, Frontiers, PLOS, JMLR, ICML, MIT Press). arXiv preprints excluded. Entries marked **(unverified)** need an author/DOI check before citing.

This bibliography is organized around the four themes of the literature review. Each entry records authors, year, title, venue, a short annotation, and a relevance note tying the work to the thesis and its research gap.

The thesis targets a specific gap: most kernel-combination methods (classical Multiple Kernel Learning in particular) learn a single, *global* set of kernel weights shared by every sample, and they do not couple kernel selection to per-feature, noise-aware weighting. The thesis proposes *adaptive* (per-sample / per-feature) dynamic weighting that remains robust under the sensor noise typical of IoT healthcare streams.

---

## Theme 1 — SVMs and Kernel Methods for High-Dimensional Biomedical Data

### Ghosh, S., Dasgupta, A., et al. (2024). *Exploring Kernel Machines and Support Vector Machines: Principles, Techniques, and Future Directions.* Mathematics, 12(24), 3935. MDPI. **(author list unverified)**
A recent review that consolidates the theory of kernel machines and SVMs, surveys established and emerging kernel functions, and identifies open directions including kernel design and parameter selection. It frames novel kernel construction as a central lever for improving SVM behavior in difficult feature spaces.
**Relevance (Theme 1):** Provides the methodological grounding for the thesis and explicitly positions custom/composite kernels as a frontier, supporting the move from fixed single kernels toward hybrid, adaptive constructions.

### Lubura, J., et al. (2024). *A distance-based kernel for classification via Support Vector Machines.* Frontiers in Artificial Intelligence, 7, 1287875. **(author list unverified)**
Introduces a distance/similarity-matrix kernel tailored to binary and categorical features and shows it handles both binary and multi-class problems competitively. The work demonstrates that bespoke kernels matched to data structure can outperform off-the-shelf RBF/polynomial choices.
**Relevance (Theme 1):** Evidence that kernel choice should adapt to the structure of the data — a premise the thesis extends from a single tailored kernel to a dynamically weighted hybrid of kernels.

### Akkur, E., et al. (2024). *Enhancing Cancerous Gene Selection and Classification for High-Dimensional Microarray Data Using a Novel Hybrid Filter and Differential Evolutionary Feature Selection.* Cancers, 16(23), 3913. MDPI. **(author list unverified)**
Proposes a hybrid filter plus differential-evolution feature-selection pipeline to reduce thousands of redundant, irrelevant, and noisy genes before classification, improving cancer-diagnosis accuracy on microarray data.
**Relevance (Theme 1):** Illustrates the canonical high-dimensional, small-sample, noisy-feature regime the thesis addresses, and motivates feature weighting/selection as a precondition for effective kernel classification.

### Devi, R. M., et al. (2025). *Cancer classification in high dimensional microarray gene expressions by feature selection using eagle prey optimization.* Frontiers in Genetics, 16, 1528810. **(author list unverified)**
Applies a bio-inspired (eagle prey) optimizer to select informative genes from high-dimensional expression data, feeding compact feature subsets to downstream classifiers including SVM.
**Relevance (Theme 1):** Reinforces that dimensionality reduction and feature relevance scoring are integral to SVM performance on biomedical data; the thesis internalizes this as learned per-feature weights rather than a separate selection stage.

### Mohammed, M., et al. (2025). *Multimodal feature-optimized approaches for cancer classification using microarray gene expression analysis.* Scientific Reports, 15. Nature. **(author list unverified)**
Combines multiple feature-optimization strategies for cancer classification on microarray data, reporting gains from fusing complementary feature representations.
**Relevance (Theme 1):** Supports the multi-representation premise behind MKL and the thesis's hybrid-kernel design — different feature views contribute unequally and should be weighted.

### Guyon, I., Weston, J., Barnhill, S., & Vapnik, V. (2002). *Gene Selection for Cancer Classification using Support Vector Machines.* Machine Learning, 46(1–3), 389–422. *(seminal)*
The foundational SVM-RFE paper: recursive feature elimination using SVM weight magnitudes to rank genes, establishing SVMs as a default tool for high-dimensional biomedical classification.
**Relevance (Theme 1):** Seminal anchor showing SVM weights themselves encode feature importance — the conceptual seed for the thesis's dynamic feature weighting.

---

## Theme 2 — Multiple Kernel Learning (MKL)

### Gönen, M., & Alpaydın, E. (2011). *Multiple Kernel Learning Algorithms.* Journal of Machine Learning Research, 12, 2211–2268. *(seminal)*
The standard survey of MKL: it formalizes learning a (typically convex/linear) combination of base kernels and categorizes methods by combination rule, optimization, and weighting scheme. It also notes that most formulations assign one global weight per kernel across all samples.
**Relevance (Theme 2):** Defines the baseline the thesis critiques — global, static kernel weights — and is the primary reference for stating the research gap.

### Gönen, M., & Alpaydın, E. (2008). *Localized Multiple Kernel Learning.* Proceedings of the 25th International Conference on Machine Learning (ICML), 352–359. *(seminal, directly on the gap)*
Introduces a gating model that makes kernel weights *input-dependent*, so different regions of the input space use different kernel combinations rather than one global weighting.
**Relevance (Theme 2 / core gap):** The closest classical precedent to the thesis's per-sample adaptivity; the thesis extends this idea by coupling localized kernel weighting with noise-robust per-feature weighting for IoT healthcare data.

### Han, Y., et al. (2013). *Localized Multiple Kernel Learning Via Sample-Wise Alternating Optimization.* IEEE Transactions on Cybernetics, 43(1), 137–148. **(author list unverified)**
Reformulates localized MKL with a sample-wise alternating optimization that learns kernel weights specific to each sample more efficiently and stably than the original gating approach.
**Relevance (Theme 2 / core gap):** Demonstrates technical feasibility of per-sample kernel weights at scale — directly supporting the thesis's adaptive weighting and offering an optimization template.

### Wang, T., et al. (2022). *Localized multiple kernel learning using graph modularity.* Pattern Recognition Letters, 156. Elsevier. **(author list unverified)**
Assigns sample-specific kernel weights by maximizing graph modularity, letting community/locality structure drive each base kernel's local influence.
**Relevance (Theme 2):** A modern instance of locality-aware kernel weighting; informs how the thesis can ground per-sample weights in data geometry rather than fixed coefficients.

### Cao, H., Jia, C., Li, Z., et al. (2024). *wMKL: multi-omics data integration enables novel cancer subtype identification via weight-boosted multi-kernel learning.* British Journal of Cancer, 130, 1001–1012.
Models each omics data type with several candidate kernels and learns kernel weights with a flexible weighting function, explicitly acknowledging the heterogeneous contribution of different data types and kernels.
**Relevance (Theme 2):** A current biomedical MKL system that learns weights across kernels — but at the data-type level, still globally. It exemplifies the static-weight limitation the thesis targets while validating MKL in healthcare-adjacent settings.

### Briscik, M., et al. (2024). *Supervised Multiple Kernel Learning approaches for multi-omics data integration.* BioData Mining, 17. Springer. **(author list/DOI unverified)**
Benchmarks several supervised MKL strategies — including a deep MKL variant that fuses kernel-PCA embeddings via sub-networks — for integrating heterogeneous omics layers.
**Relevance (Theme 2):** Surveys how kernel weights can be learned, including deep/end-to-end variants, giving the thesis a comparison space and showing the field moving toward more flexible (but still largely global) weighting.

### Ghanizadeh, A. N., Ghiasi-Shirazi, K., Monsefi, R., et al. (2024). *Neural Generalization of Multiple Kernel Learning.* Neural Processing Letters, 56, 12. Springer.
Recasts MKL as a neural architecture, generalizing the linear kernel combination so that weighting can be learned end-to-end with gradient methods.
**Relevance (Theme 2):** Bridges MKL and neural learning of weights; relevant to how the thesis might parameterize adaptive, data-dependent kernel weights rather than solving a fixed convex combination.

---

## Theme 3 — Noise Handling and Feature Weighting in IoT Healthcare

### Khan, M. M., & Alkhathami, M. (2024). *Anomaly detection in IoT-based healthcare: machine learning for enhanced security.* Scientific Reports, 14, 5872. Nature.
Models machine-learning techniques on IoT healthcare network traffic (CIC IoT dataset) for detecting anomalous/abnormal samples, with preprocessing and feature handling to cope with noisy, high-volume streams.
**Relevance (Theme 3):** Characterizes the noisy, security-sensitive IoT healthcare data environment the thesis operates in and the need for noise-tolerant feature handling.

> **Removed — not a designated source.** *Akhtar, M., et al. (2024). Flexi-Fuzz least squares SVM for Alzheimer's diagnosis* was excluded because it is an arXiv preprint only. Its role (per-sample fuzzy weighting for noise/imbalance robustness in healthcare) is covered by Lin & Wang (2002) and Liu et al. (2024) below. Re-add if a published journal version is found.

### Liu, Z., et al. (2024). *The fuzzy support vector data description based on tightness for noisy label detection.* Complex & Intelligent Systems, 10. Springer. **(author list unverified)**
Adds a tightness-based membership degree to SVDD to identify and discount noisy-labelled points, improving robustness when labels are unreliable.
**Relevance (Theme 3):** Demonstrates membership-based down-weighting for label noise — directly relevant to noisy IoT-collected labels and to the thesis's noise-aware weighting.

### Anand, P., et al. (2025/2026). *A robust and lightweight support vector machine for imbalanced and noisy data via Benders decomposition.* Neurocomputing. Elsevier. **(author list and year unverified)**
Develops a robust, computationally light SVM formulation for imbalanced and noisy data using Benders decomposition, targeting scalability alongside robustness.
**Relevance (Theme 3):** Addresses robustness and efficiency jointly — both required for IoT/edge healthcare deployment, a practical constraint the thesis must respect.

### Saif, S., et al. (2024). *Implementation of machine learning techniques with big data and IoT to create effective prediction models for health informatics.* Biomedical Signal Processing and Control, 91. Elsevier. **(author list unverified)**
Builds IoT/big-data health-informatics prediction pipelines with preprocessing to remove noise and feature analysis/optimization to select relevant features before modeling.
**Relevance (Theme 3):** Exemplifies the standard "denoise then weight/select features" IoT healthcare pipeline that the thesis aims to unify into a single adaptive model.

### Nayak, S. R., et al. (2025). *Enhancing IoT-based healthcare security with grey filter Bayesian CNN and optimization algorithms.* Scientific Reports, 15. Nature. **(author list unverified)**
Combines grey-filter denoising with a Bayesian CNN and metaheuristic optimization for secure IoT healthcare classification under noisy inputs.
**Relevance (Theme 3):** Shows explicit noise filtering plus learned optimization in IoT healthcare; contrasts with the thesis's goal of handling noise inside the kernel/weighting model rather than as a separate filter stage.

### Lin, C.-F., & Wang, S.-D. (2002). *Fuzzy Support Vector Machines.* IEEE Transactions on Neural Networks, 13(2), 464–471. *(seminal)*
The original fuzzy SVM: assigns each training point a fuzzy membership so that less reliable points (outliers/noise) contribute less to the decision boundary.
**Relevance (Theme 3):** Seminal origin of sample-level weighting for noise robustness; the conceptual basis for nearly all noise-aware weighting in this theme and for the thesis's per-sample weights.

---

## Theme 4 — Adaptive / Hybrid Kernel SVMs and Dynamic Feature Weighting

### Zhang, S., et al. (2023). *Bayesian-Optimized Hybrid Kernel SVM for Rolling Bearing Fault Diagnosis.* Sensors, 23(11), 5137. MDPI. **(author list unverified)**
Constructs a hybrid kernel combining polynomial and RBF kernels for an SVM and uses Bayesian optimization to tune the penalty factor, kernel parameters, and the global mixing coefficient between the two kernels.
**Relevance (Theme 4 / core gap):** Closest architectural template to the thesis (hybrid local+global kernel SVM), but the mixing weight is a single tuned constant — exactly the static-weight limitation the thesis replaces with per-sample/per-feature adaptivity.

### Author(s) unverified (2025). *Enhanced SVM-based model for predicting cyberspace vulnerabilities: analyzing the role of user group dynamics and capital influx.* PLOS ONE. **(author list unverified)**
Proposes an SVM with three adaptive mechanisms: variance-based dynamic kernel-parameter adjustment, risk-aware kernel-weight optimization, and skewness-driven kernel-function switching.
**Relevance (Theme 4 / core gap):** A direct example of *dynamic* kernel parameters and weights driven by data statistics — a strong precedent for the thesis's adaptive weighting, here outside healthcare.

### Author(s) unverified (2025). *Hybrid Edge-Based Intrusion Detection for IoT Using Adaptive Lightweight SVM with Dynamic Feature Pruning.* IEEE (conference, IEEE Xplore document 11135296). **(author list unverified)**
Couples a lightweight adaptive SVM with dynamic feature pruning, using mutual-information and correlation analysis to select the most critical features on the fly at the IoT edge.
**Relevance (Theme 4 / core gap):** Combines adaptivity, dynamic feature weighting/pruning, and IoT-edge constraints — the same three pressures the thesis faces, supporting the feasibility of dynamic feature weighting in IoT settings.

### Singh, A. K., et al. (2023). *A Hybrid Particle Swarm Optimization Algorithm with Dynamic Adjustment of Inertia Weight Based on a New Feature Selection Method to Optimize SVM Parameters (DWPSO-SVM).* Entropy. **(author list and venue unverified — PMC10047894)**
Uses a PSO variant with dynamically adjusted inertia weight together with a feature-selection method to jointly optimize the selected features and SVM hyperparameters.
**Relevance (Theme 4):** Demonstrates joint, dynamic optimization of feature weighting and SVM parameters — methodologically aligned with the thesis, though weighting remains at the dataset level rather than per-sample.

### Author(s) unverified (2025). *A dynamically adaptive hybrid fuzzy–ANN–SVM architecture for agro-climatic yield forecasting under uncertainty.* Theoretical and Applied Climatology. Springer. **(author list unverified)**
Integrates fuzzy logic, ANN, and SVM into a dynamically adaptive hybrid that handles nonlinear patterns and uncertainty, with components that adjust to changing input conditions.
**Relevance (Theme 4):** Shows hybrid adaptive architectures that fuse fuzzy weighting (noise/uncertainty handling) with SVM generalization — a structural analogue to the thesis's fuzzy-weighted adaptive hybrid kernel SVM, transferable to healthcare.

### Tian, Y., et al. (2024). *Multimodal classification of Alzheimer's disease and mild cognitive impairment using custom MKSCDDL kernel over CNN.* **(author list and venue unverified — PMC10799876)**
Combines a custom kernel-based discriminative dictionary-learning module (MKSCDDL) with CNN features for explainable multimodal Alzheimer's/MCI classification.
**Relevance (Theme 4):** Demonstrates custom, learned kernels layered on deep features for biomedical diagnosis, supporting the thesis's hybrid-kernel direction and its emphasis on per-modality/per-feature contribution.

### Schölkopf, B., & Smola, A. J. (2002). *Learning with Kernels: Support Vector Machines, Regularization, Optimization, and Beyond.* MIT Press. *(seminal)*
The definitive reference on kernel methods and SVMs, covering kernel construction, the property that sums/products of valid kernels remain valid kernels, regularization, and optimization.
**Relevance (Theme 4):** Provides the theoretical guarantee (closure of kernels under convex combination) that legitimizes hybrid kernels and adaptive weighted kernel combinations — the mathematical foundation of the thesis.

---

## Synthesis & Gaps

Across Themes 1–2 the literature establishes two settled facts. First, SVMs and kernel methods remain a strong default for high-dimensional, small-sample biomedical data, and performance hinges on matching the kernel to the structure of the data (Guyon et al.; Lubura et al.; Ghosh, Dasgupta et al., 2024). Second, when no single kernel or feature view suffices, Multiple Kernel Learning combines several kernels and learns their weights (Gönen & Alpaydın, 2011). The recurring limitation, however, is that classical and most current MKL formulations — including recent biomedical systems such as wMKL (Cao et al., 2024) and the supervised-MKL benchmarks (Briscik et al., 2024) — learn a *single global* weighting shared by every sample. The localized-MKL line (Gönen & Alpaydın, 2008; Han et al., 2013; Wang et al., 2022) is the principal exception: it makes kernel weights input-dependent. But that work was developed largely outside noisy IoT healthcare, and it does not couple localized kernel weighting to explicit, noise-aware *feature* weighting.

Themes 3–4 supply the missing ingredients but in separate strands. The noise-handling literature shows that per-sample membership weighting (Lin & Wang, 2002; Liu et al., 2024) and IoT-specific denoising/feature-selection pipelines (Khan & Alkhathami, 2024; Saif et al., 2024; Nayak et al., 2025) reliably improve robustness when sensor data are noisy, imbalanced, or mislabelled. The adaptive/hybrid-kernel literature shows that hybrid kernels (Zhang et al., 2023), data-driven dynamic kernel parameters and weights (the 2025 adaptive-SVM and edge-IDS works), and joint feature/parameter optimization (DWPSO-SVM) are all individually feasible — and that kernel theory permits these compositions (Schölkopf & Smola, 2002). Yet hybrid-kernel SVMs still typically fix the kernel mixing as a single tuned constant, and adaptive-feature-weighting methods rarely operate jointly with adaptive kernel combination.

The gap the thesis addresses sits precisely at the intersection these strands leave open: an SVM that (a) combines multiple kernels with weights that adapt *per sample* rather than globally, (b) simultaneously learns *per-feature* weights that down-weight noisy or irrelevant sensor channels, and (c) does so under the noise, imbalance, and resource constraints of IoT healthcare streams. Localized MKL contributes per-sample kernel adaptivity, fuzzy/weighted SVMs contribute noise-robust sample and feature weighting, and hybrid-kernel SVMs contribute the composite-kernel architecture — but no surveyed work unifies all three. An *adaptive hybrid kernel SVM with dynamic feature weighting* is therefore a well-motivated and, on the evidence above, technically achievable contribution.
