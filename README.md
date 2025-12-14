# Sliding‑Window Network Anomaly Detection — BGP & IODA

Short summary  
This repository benchmarks eight anomaly detection methods on time‑series telemetry from BGP routing (RouteViews) and IODA outage signals using a sliding‑window representation. The goal is to compare detection quality and runtime across sequence models (CNN1D, GRU, LSTM, Bi‑GRU, Bi‑LSTM), analytic random‑feature learners (BLS, VFBLS), and a tree‑based baseline (LightGBM).

Highlights
- End‑to‑end pipeline: median imputation, z‑score normalization, optional feature selection.
- Sliding windows (configurable; default = 10) convert time series into fixed‑length sequences.
- Eight-model benchmark suite with consistent preprocessing, evaluation, and timing.
- Fast closed‑form learners (BLS, VFBLS) vs. iterative deep sequence models (PyTorch).
- Built‑in evaluation: ROC‑AUC, PR‑AUC, Accuracy, F1, confusion matrix, and timing.

Datasets
- BGP (RouteViews): Slammer, Moscow_blackout, WannaCrypt — features include announcements, withdrawals, AS‑path statistics, duplicate announcements, and related routing metrics.
- IODA (CAIDA): Zhytomyr, Iraq, Gaza — signals include BGP visible prefixes, active probing reachability, and darknet traffic patterns.
Datasets are labeled for known events and included as CSV files in the repository.

Sliding‑Window Technique
- Windows of length W (default 10) are created by sliding across the time series (stride = 1).
- Each window is labeled using the label at the final timestep of the window (this is configurable).
- Windowing captures temporal patterns and augments samples, but must be handled carefully to avoid train/test leakage (chronological splits recommended).

Models Implemented
- CNN1D: Temporal convolutional network for local pattern extraction.
- GRU / LSTM: Unidirectional recurrent models for sequence modeling.
- Bi‑GRU / Bi‑LSTM: Bidirectional recurrent models capturing forward and backward context.
- BLS / VFBLS: Broad Learning System variants — random feature mapping + closed‑form ridge regression (very fast).
- LightGBM: Gradient boosted decision trees on flattened windows as a non‑sequence baseline.

Evaluation & Thresholding
- Metrics: ROC‑AUC, PR‑AUC (average precision), Accuracy, F1, confusion matrix, and training time.
- Research Mode (default in scripts): selects the probability threshold that maximizes F1 on the evaluated set. Note: selecting thresholds using the test set leaks information and inflates reported metrics. For robust evaluation:
  - Use a validation set to select thresholds, or
  - Use cross‑validation and keep the test set strictly held out.

Quickstart (Windows)
1. Create & activate a virtual environment:
   - python -m venv .venv
   - .venv\Scripts\activate
2. Install dependencies:
   - pip install -r requirements.txt
   - or: pip install lightgbm torch scikit-learn pandas numpy
   Note: scripts attempt to auto-install missing packages if needed.
3. Run experiments:
   - python "routeviews-sliding windows.py"
   - python "ioda-sliding windows.py"

Usage Notes & Configuration
- Edit window_size at the top of each script to change window length (paper tested 1–300).
- For reproducibility set random seeds (numpy, torch, LightGBM).
- Scripts are CPU‑only by default. Add device handling in PyTorch models to use CUDA.
- Consider chronological splitting to avoid temporal leakage.

Reproducibility & Best Practices
- Fix seeds: np.random.seed(...), torch.manual_seed(...), and LightGBM seed params.
- Threshold selection: pick thresholds on validation set, not test.
- Handle class imbalance: report PR‑AUC and consider class weighting, oversampling, or cost‑sensitive losses.
- Save model artifacts and predictions for post‑hoc analysis.

Suggested Improvements
- Add validation split and early stopping for deep models.
- Add GPU support and data loaders that preserve chronological order.
- Save detailed logs, model checkpoints, and per‑run seeds for reproducibility.
- Add unit tests for preprocessing, windowing, and threshold selection logic.

Repository Files
- routeviews-sliding windows.py — BGP experiments and models.
- ioda-sliding windows.py — IODA experiments and models (manual implementations included).
- requirements.txt — recommended packages (create if missing).
- README.md — this file.

License & Contributing
- Add your preferred license (e.g., MIT) in the repository root.
- Contributions, bug reports, and improvements are welcome via issues and pull requests.

References
1. Zhida Li and Ljiljana Trajković, “Enhancing Cyber Defense: Using Machine Learning Algorithms for Detection of Network Anomalies,” IEEE SMC, 2023.  
2. Luiz Felipe Oliveira, Rob Ballantyne, Jano Souza, and Ljiljana Trajković, “Internet Outages During Times of Conflict,” IEEE SMC, 2024.  
3. Zhida Li, Ana Laura Gonzalez Rios, and Ljiljana Trajković, “Machine Learning for Detecting Anomalies and Intrusions in Communication Networks,” IEEE J‑SAC.  
4. Zhida Li, CyberDefense GitHub repository: https://github.com/zhida-li/CyberDefense  
5. RouteViews Project, University of Oregon, RouteViews Archive (accessed 2025).  
6. Internet Outage Detection and Analysis (IODA), Georgia Tech Internet Intelligence Lab (accessed 2025).

Contact
Include maintainer GitHub handle or email here for questions and contributions.
