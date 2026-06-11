const { Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType, LevelFormat } = require("docx");
const fs = require("fs");

const P = (t) => new Paragraph({
  alignment: AlignmentType.JUSTIFIED,
  spacing: { after: 160, line: 276 },
  children: [new TextRun(t)],
});
const H1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(t)] });
const H2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(t)] });
const NUM = (t, ref) => new Paragraph({
  numbering: { reference: ref, level: 0 },
  alignment: AlignmentType.JUSTIFIED,
  spacing: { after: 120, line: 276 },
  children: [new TextRun(t)],
});

const children = [
  H1("1. Introduction"),

  H2("1.1 Background and Motivation"),
  P("The Internet of Medical Things and wearable sensing now produce continuous, high-dimensional data from several modalities at once. Modern healthcare systems lean on sensor networks that track physiological signals such as heart rate, blood oxygen, ECG waveforms, accelerometer-based motion, and ambient conditions. These signals are rarely clean. Sensors differ in quality, the environment interferes with them, and patients differ physiologically from one another, so the resulting data is awkward for classical machine learning methods that assume clean, stationary inputs (Al-amri et al., 2021; AL-Dhief et al., 2020)."),
  P("For this data to support clinical decisions, a classifier has to do three things at once: handle high dimensionality without overfitting, stay robust to the noise and outliers that sensor collection inevitably produces, and adapt to shifting data characteristics without being re-tuned from scratch each time. Elderly care is an especially demanding case. Older adults often wear several monitoring devices simultaneously, adherence is uneven, and the clinical relevance of a given feature can change as a person's health changes (Talaat & El-Balka, 2023; Abdellatif et al., 2022)."),
  P("Support Vector Machines are a sensible starting point for high-dimensional classification. They rest on statistical learning theory, offer generalization guarantees through margin maximization, and model nonlinear structure through the kernel trick (Vallabhuni & Debasis, 2025; Wang et al., 2023; Muriira et al., 2018). The kernel trick maps data implicitly into a higher-dimensional space where linear separation becomes possible, which helps with the nonlinear decision boundaries common in biomedical problems."),
  P("The difficulty is that a standard SVM with a fixed kernel tends to struggle on noisy, multimodal healthcare data from IoT devices. One kernel function seldom captures the variety of structure in heterogeneous sensor data. A linear kernel picks up broad trends but misses local nonlinearities; an RBF kernel captures local structure but can overfit to noise in high-dimensional spaces. The reliability of individual features also drifts over time and across patients, which leaves a static kernel choice poorly matched to dynamic IoT healthcare settings (Al-hajjar & Al-Qurabat, 2023; Mahdavinejad et al., 2018)."),

  H2("1.2 Research Gap and Novelty Statement"),
  P("Multiple kernel learning improved on single-kernel SVMs by learning a convex combination of base kernels, so that different kernels capture different aspects of the data (Gonen & Alpaydin, 2011; Rakotomamonjy et al., 2008). Weighted SVM variants, in a separate line of work, handle noisy or imbalanced features by assigning different importance to different feature dimensions. The two lines have mostly developed apart, and their combination in the IoT healthcare setting has drawn little attention."),
  P("As far as we are aware, no existing approach handles the following three things together: data-quality-driven, per-feature weight adaptation built into the kernel evaluation, where feature weights respond to measurable indicators of sensor reliability such as missingness rates and variance stability; adaptive fusion of global and local kernel components that balances the complementary strengths of linear and RBF kernels; and a focus on noisy, high-dimensional IoT healthcare streams, where these problems bite hardest."),
  P("Standard MKL methods (Gonen & Alpaydin, 2011; Rakotomamonjy et al., 2008) fix the kernel combination weights during training and leave them unchanged at inference. Feature-weighted SVMs, such as the feature-and-sample weighted SVM of Zhang et al. (2011) and the weighted SVM of Yang et al. (2005), usually take static feature weights from preprocessing steps like mutual information or variance thresholds, without tying them to how the kernels are combined. Fuzzy SVMs address noise sensitivity by giving samples membership degrees (Lin & Wang, 2004), but they do not feed per-feature data-quality indicators into the kernel evaluation itself. This thesis connects these strands: feature weights modulate the kernel evaluation directly, so unreliable features are de-emphasized inside the similarity computation rather than filtered out beforehand."),

  H2("1.3 Scope of This Thesis"),
  P("To stay feasible within a Master's timeline, the work concentrates on a single core contribution: the design and evaluation of an adaptive hybrid kernel SVM that combines dynamic feature weighting with an adaptive linear-RBF kernel mixture. The scope is kept narrow on purpose so the central idea can be developed and tested carefully."),
  P("Concretely, the thesis formulates a composite kernel that combines a global linear component and a local RBF component through a tunable mixture coefficient, with both components operating on feature-weighted inputs. It develops a feature weighting mechanism that computes per-feature reliability scores from missingness rates and variance stability and feeds those weights straight into the kernel computation. It gives a complete mathematical formulation covering the optimization objective, the feature weight computation, and the algorithmic pseudocode. Finally, it evaluates the framework on two public benchmark datasets, OPPORTUNITY and CASAS, against a focused set of baselines, using accuracy, F1-score, and noise robustness as the primary criteria."),
  P("Several extensions are deliberately left out so the project stays manageable, and Section 8 returns to them as future directions: online or streaming adaptation of kernel weights at inference, deployment and optimization on edge devices, fairness auditing across demographic subgroups, privacy-preserving computation such as federated learning or differential privacy, and deep learning hybrid architectures."),

  H2("1.4 Research Objectives and Hypotheses"),
  P("The primary objective is to design and evaluate an adaptive hybrid kernel SVM, combining a global linear component and a local RBF component with both an adaptive mixture coefficient and data-quality-driven per-feature weights, for classifying noisy, high-dimensional IoT healthcare data. Success is defined as a measurable improvement over a single-kernel SVM and over static MKL on accuracy, macro-averaged F1, and noise robustness. This breaks down into four sub-objectives:"),
  NUM("Derive a complete mathematical formulation of the model, covering the composite kernel, the per-feature weight computation, the SVM optimization objective, and the hyperparameter space.", "obj"),
  NUM("Implement the model by extending a standard SVM solver in Python and scikit-learn through a custom precomputed kernel matrix.", "obj"),
  NUM("Validate the model empirically on the OPPORTUNITY and CASAS datasets against three baselines (SVM-RBF, SVM-Linear, and static MKL).", "obj"),
  NUM("Analyse noise robustness through synthetic noise injection at a range of intensities, producing performance degradation curves.", "obj"),
  P("The empirical work is organised around three hypotheses. The first concerns the value of the hybrid kernel itself, the second concerns robustness to noise, and the third isolates the contribution of the feature weighting:"),
  NUM("H1 (hybrid kernel advantage). On multimodal, noisy datasets, the adaptive hybrid kernel SVM achieves higher accuracy and macro-F1 than any single fixed kernel, whether RBF or linear.", "hyp"),
  NUM("H2 (noise resilience). The data-quality-driven weights inside the kernel give better resistance to synthetic noise injection than a standard kernel SVM or static MKL without weights, shown as smaller performance degradation as noise intensity rises.", "hyp"),
  NUM("H3 (feature weighting contribution). The weighting mechanism yields a measurable improvement over the same hybrid kernel without weights. This is tested by an ablation that compares the full model against a weight-free variant using the identical linear-RBF mixture.", "hyp"),

  H2("1.5 Overview of the Proposed Approach"),
  P("The proposed model keeps the familiar SVM training procedure and changes the similarity function it relies on. Instead of a single fixed kernel, the decision function uses a composite kernel that adds a global linear term and a local RBF term, balanced by a mixture coefficient. The linear term captures broad, global trends across the feature space, while the RBF term captures local nonlinear structure; the coefficient controls how much each contributes, so the model is not committed in advance to one view of the data."),
  P("The second ingredient is a per-feature weighting that sits inside the kernel rather than in front of it. Before the kernel compares two samples, each feature is scaled by a reliability weight derived from data-quality indicators: how often that feature is missing, and how stable its variance is across the data. Features that are frequently missing or erratic receive smaller weights, so they contribute less to the computed similarity, while stable and well-populated features count for more. Because this scaling happens inside the kernel evaluation, the effect propagates through the whole decision function, and unreliable sensor channels are de-emphasized without being discarded outright."),
  P("This design separates the present work from two nearby approaches. Static MKL would learn one global mixture weight and stop there, with no notion of per-feature reliability. A conventional preprocessing pipeline would score or select features first and then hand a fixed, reduced feature set to an ordinary kernel. Here the two mechanisms are joined: the kernel composition and the feature weighting operate together within a single model, and the implementation realises this through a precomputed kernel matrix that a standard SVM solver can consume directly."),

  H2("1.6 Contributions"),
  P("The thesis makes four contributions:"),
  NUM("A unified adaptive hybrid kernel framework. The main methodological contribution integrates data-quality-driven feature weights directly into a composite linear-RBF kernel. Unlike static MKL or a decoupled preprocessing stage, the model modulates the kernel similarity itself, so the decision function naturally de-emphasizes unreliable features.", "con"),
  NUM("Empirical evidence on noise robustness. The thesis provides systematic results on two IoT healthcare datasets, with noise-injection experiments that quantify the benefit of the weights under sensor degradation and offer practical guidance for deployment.", "con"),
  NUM("An ablation analysis of component contributions. By comparing the full model against static MKL, the thesis isolates the contribution of the weights, showing when and by how much they improve the hybrid kernel on its own.", "con"),
  NUM("A reusable implementation. The thesis delivers a documented, scikit-learn-compatible Python implementation, comprising the weighting module, the adaptive hybrid kernel, and a full experimental pipeline with configurations for reproducibility.", "con"),

  H2("1.7 Thesis Outline"),
  P("The remainder of the thesis is organised as follows. Chapter 2 reviews the relevant literature on SVMs and kernel methods for high-dimensional biomedical data, Multiple Kernel Learning, noise handling and feature weighting in IoT healthcare, and adaptive and hybrid kernel SVMs, and locates the research gap. Chapter 3 sets out the methodology, including the mathematical formulation of the composite kernel, the feature weight computation, the optimization objective, the algorithm, the datasets, and the evaluation protocol. Chapter 4 describes the implementation and experimental setup. Chapter 5 reports the experimental results, covering classification performance, the noise robustness analysis, the ablation study, and statistical significance testing. Chapter 6 discusses the findings, their limitations, and their practical implications. Chapter 7 concludes and outlines directions for future work."),
];

const doc = new Document({
  styles: {
    default: { document: { run: { font: "Times New Roman", size: 24 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 32, bold: true, font: "Times New Roman" },
        paragraph: { spacing: { before: 300, after: 200 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 28, bold: true, font: "Times New Roman" },
        paragraph: { spacing: { before: 220, after: 140 }, outlineLevel: 1 } },
    ],
  },
  numbering: {
    config: [
      { reference: "obj", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
      { reference: "hyp", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
      { reference: "con", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
    ],
  },
  sections: [{
    properties: { page: { size: { width: 12240, height: 15840 }, margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } } },
    children,
  }],
});

Packer.toBuffer(doc).then((b) => { fs.writeFileSync("Chapter1_Introduction.docx", b); console.log("written"); });
