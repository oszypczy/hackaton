# Prompty do Deep Research (Claude.ai)

Poniższe prompty są zaprojektowane do użycia w Claude.ai Deep Research,
aby zebrać pogłębioną wiedzę na potrzeby CISPA European Cybersecurity & AI Hackathon Championship (Warszawa, 9-10 maja 2026).

---

## PROMPT 1: Adversarial Attacks na modele klasyfikacji obrazów

```
I'm preparing for a 24-hour AI security hackathon where one challenge involves generating adversarial perturbations to fool image classifiers. I need a comprehensive, practical guide covering:

1. All major white-box attack methods (FGSM, PGD, C&W, DeepFool, AutoAttack) - how they work mathematically, their strengths/weaknesses, and when to use each
2. Black-box attack methods (transfer attacks, query-based attacks, score-based vs decision-based)
3. Targeted vs untargeted attacks - implementation differences
4. Practical tricks to maximize attack success rate while minimizing perturbation magnitude (Lp norms: L0, L2, Linf)
5. Python code examples using PyTorch (preferably) for each method
6. How to attack common architectures: ResNet, VGG, Vision Transformers
7. Defense-aware attacks (attacking models with adversarial training or input preprocessing)

Focus on actionable, competition-ready knowledge. Include specific hyperparameter recommendations and common pitfalls.
```

---

## PROMPT 2: Model Inversion i ekstrakcja danych treningowych

```
I'm competing in an AI security hackathon where a challenge involves reconstructing training data from trained ML models (model inversion attacks). I need comprehensive research on:

1. Model inversion attacks on image classifiers - how to reconstruct training images from model parameters or API access
2. Model inversion attacks on LLMs - extracting training text/sequences from language models
3. The difference between membership inference, dataset inference, and model inversion
4. Gradient-based inversion techniques (optimization in input space)
5. Training data extraction from diffusion models (Carlini et al. 2023 approach: generate-and-filter pipeline)
6. Practical attack implementations in PyTorch - step-by-step code
7. Metrics for measuring reconstruction quality (FID, SSIM, exact match, etc.)
8. How differential privacy (DP-SGD) affects these attacks and how to attack DP-protected models
9. The CoDeC method for detecting data contamination in LLMs via in-context learning
10. Dataset inference approach by Maini et al. (2024) - aggregating weak MIA signals into strong dataset-level statistical tests

Include recent papers from 2024-2025, practical Python code, and competition-ready strategies.
```

---

## PROMPT 3: Watermarking i watermark removal w AI

```
I'm preparing for an AI security hackathon with challenges on watermark detection and removal. I need deep research on:

1. LLM text watermarking:
   - Kirchenbauer et al. (2023) "green list" method - full technical details and how to detect it
   - Other LLM watermarking schemes (KGW, Unigram, EXP-edit, etc.)
   - How to remove/attack text watermarks (paraphrasing, token substitution, emoji attacks)
   - Statistical tests for watermark detection (z-scores, p-values)

2. Image watermarking:
   - Invisible watermark embedding techniques (spatial domain, frequency domain, DNN-based)
   - Tree-ring watermarks for diffusion models
   - Stable Signature and other generation-time watermarking
   - Watermark removal attacks: compression, noise, diffusion-based regeneration, adversarial removal
   - The NeurIPS 2024 result proving invisible watermarks are provably removable

3. Practical attack strategies:
   - Given a watermarked image, what are the most effective removal techniques?
   - Given LLM-generated text, how to detect if it's watermarked?
   - Code implementations in Python

Focus on attack techniques (removal/detection) since the hackathon challenges involve breaking watermarks.
```

---

## PROMPT 4: Model Stealing / Model Extraction przez API

```
I need comprehensive research on model stealing/extraction attacks for an AI security hackathon:

1. Theoretical foundations: What can be learned from API queries? (decision boundaries, model architecture, hyperparameters)
2. Query-efficient extraction methods:
   - Active learning-based approaches
   - Knockoff Nets (Orekondy et al.)
   - JBDA (Jacobian-Based Data Augmentation)
   - Prediction-based vs logit-based extraction
3. Extracting different model types:
   - Image classifiers (CNNs, ViTs)
   - Language models
   - Decision trees / linear models (Tramèr et al. 2016)
4. How to train a surrogate/shadow model that mimics the target
5. Knowledge distillation as an extraction technique
6. Measuring extraction success (fidelity, accuracy agreement, task accuracy)
7. Defense mechanisms and how to bypass them (watermarking, PATE, output perturbation)
8. Complete Python implementation: from querying an API to training a stolen model

The hackathon provides model access via API only (black-box). I need practical, working code and strategies to maximize model fidelity with minimum queries.
```

---

## PROMPT 5: AI Image Attribution / Source Model Detection

```
I need research on detecting which AI model generated a specific image, for a cybersecurity hackathon:

1. What traces/artifacts do different generative models leave in their outputs?
   - GAN fingerprints (spectral analysis, upsampling artifacts)
   - Diffusion model fingerprints
   - Model-specific patterns in frequency domain
2. Passive fingerprinting techniques:
   - Noise residual analysis
   - GAN-specific frequency artifacts
   - PRNU-like approaches for generative models
3. Active fingerprinting:
   - Model watermarking / fingerprinting during training
   - FI-LoRA and similar fine-tuning based approaches
4. Classification approaches:
   - Training a classifier on outputs from multiple generators
   - Few-shot attribution
   - Transfer learning for model attribution
5. Robustness of attribution:
   - How post-processing (JPEG compression, resizing, cropping) affects detection
   - Adversarial attacks against attribution systems
6. Practical implementation: given a set of images, how to build a model that identifies the source generator

Include Python code, relevant datasets, and recent papers (2024-2025).
```

---

## PROMPT 6: Fairness Auditing w modelach ML (Black-box)

```
For an AI security hackathon, I need research on inferring training data properties from model predictions (black-box setting):

1. Property inference attacks:
   - Detecting demographic bias in training data from API queries
   - Determining if training data was balanced/imbalanced
   - Shadow model approach for property inference
2. Fairness auditing techniques:
   - Statistical parity, equalized odds, calibration
   - Testing for bias without access to training data
   - Adversarial fairness testing
3. Distribution inference:
   - Inferring properties of training distribution from model behavior
   - Confidence-based analysis
   - Calibration-based inference
4. Practical black-box approach:
   - How to design queries that reveal training data properties
   - Building shadow models to benchmark expected behavior
   - Statistical tests for detecting imbalance

The Barcelona hackathon had a challenge where participants had to determine if medical AI models were trained on biased patient populations using only API access. I need strategies for this type of task.
```

---

## PROMPT 7: Ogólne przygotowanie - narzędzia i szybkie prototypowanie

```
I'm competing in a 24-hour AI security hackathon. I need a practical toolkit guide:

1. Essential Python libraries for ML security research:
   - adversarial attacks (Foolbox, ART, CleverHans, Torchattacks)
   - privacy attacks (ML Privacy Meter, Opacus)
   - model interpretability (Captum, SHAP)
2. Quick-start templates for common tasks:
   - Loading and attacking pre-trained models (torchvision, HuggingFace)
   - Setting up model inversion optimization loops
   - API query wrappers for model stealing
   - Watermark detection statistical tests
3. GPU optimization tricks for fast iteration during hackathon
4. How to efficiently use AI assistants (Claude, ChatGPT) during the hackathon for:
   - Debugging ML code
   - Understanding unfamiliar papers quickly
   - Generating boilerplate code
   - Analyzing model outputs
5. Common pitfalls in ML security competitions and how to avoid them

Focus on speed and practicality - everything should be ready to copy-paste and adapt.
```
