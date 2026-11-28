# Evaluation metrics

| Model | Task | Metric |
| --- | --- | --- |
| Demucs `htdemucs` | Separation | SDR / SI-SDR |
| DeepFilterNet3 | Cleaning | PESQ / STOI |
| PANNs CNN14 | Classification | Precision / Recall / F1 |

- **SDR (Signal-to-Distortion Ratio):** how much louder the wanted stem is than
  the error left behind, in dB — higher is better.
- **SI-SDR (Scale-Invariant SDR):** the same idea but ignores overall volume, so
  a quiet or loud output is scored fairly.
- **PESQ (Perceptual Evaluation of Speech Quality):** an estimate of how good the
  cleaned speech sounds to a listener, roughly on a 1–4.5 scale — higher is better.
- **STOI (Short-Time Objective Intelligibility):** how many words a listener
  would still understand, from 0 to 1 — higher is better.
- **Precision:** of the instruments we predicted, how many were actually there.
- **Recall:** of the instruments actually present, how many we found.
- **F1:** the balance of precision and recall in one number.
