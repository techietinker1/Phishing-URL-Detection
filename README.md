<div align="center">

# 🛡️ Phishing URL Detector

Lightweight Flask web app + RandomForest model + heuristic rules to classify a URL as **Safe** or **Unsafe** and explain why.

</div>

## ✨ Features
- Hybrid decision: ML model vote + explicit heuristic triggers
- Probabilistic gating (threshold + margin) to reduce noise
- Human-readable reasons listed for every Unsafe result
- Debug API with raw features & class probabilities
- Two training modes: simple baseline vs enhanced feature payload

## 🚀 Quick Start
```powershell
python -m venv .venv
./.venv/Scripts/Activate.ps1
pip install -r requirements.txt
python app.py
```
Open: http://127.0.0.1:5000/

## 🧪 Training
Baseline (4 URL features):
```powershell
python phishing.py
```
Enhanced (50+ features, saved with feature list metadata):
```powershell
python train_enhanced.py
```
Resulting file: `phishing_model.pkl` (either a pure model or a payload dict with `model` + `feature_columns`).

## ⚙️ Environment Variables
| Name | Default | Description |
|------|---------|-------------|
| PHISHING_CLASS | 0 | Label representing phishing in the dataset. |
| PHISHING_PROB_THRESHOLD | 0.70 | Minimum phishing probability to consider model vote. |
| PHISHING_PROB_MARGIN | 0.10 | Required gap between phishing prob and next class. |

Example:
```powershell
set PHISHING_PROB_THRESHOLD=0.80
set PHISHING_PROB_MARGIN=0.15
python app.py
```

## 🌐 Endpoints
| Method | Path | Description |
|--------|------|-------------|
| GET | / | Web form UI |
| POST | /predict | Form submission (HTML result) |
| POST | /api/predict | JSON classification (label + reasons) |
| POST | /api/debug | JSON: features, probabilities, decision components |

Sample:
```powershell
curl -X POST http://127.0.0.1:5000/api/predict -H "Content-Type: application/json" -d '{"url":"https://example.com"}'
```

## 🔍 Heuristic Rules (summary)
- Raw IP address as host
- URL length > 120
- Hyphen count ≥ 4
- Special character cluster (@ ? = % total ≥ 5)
- Suspicious keywords (login, verify, update, secure, account, wallet, etc.)
- Risky TLD (zip, mov, country, click, link, work, fit, tokyo, rest)
- '@' present in URL
- HTTP scheme + credential keyword

If any trigger fires, URL can be marked Unsafe even if model is uncertain.

## 🧠 Decision Logic
```
unsafe_final = heuristic_unsafe OR (
	model_pred == PHISHING_CLASS AND
	phishing_prob >= PHISHING_PROB_THRESHOLD AND
	(phishing_prob - second_best_prob) >= PHISHING_PROB_MARGIN
)
```

## 🛠 Debug Endpoint
```powershell
curl -X POST http://127.0.0.1:5000/api/debug -H "Content-Type: application/json" -d '{"url":"https://www.wikipedia.org"}'
```
Returns: feature dict, ordered feature vector, class probabilities, reason list, and final decision flags.

## ⚠️ Distribution Shift Note
The enhanced model was trained with many content / HTML features you are not extracting live (they default to 0). For better calibration:
1. Retrain using URL-only features, OR
2. Implement a fetcher + parser to populate those fields, THEN
3. Recalibrate probabilities (isotonic / Platt) if needed.

## 📈 Possible Improvements
- URL-only retraining script variant (to eliminate zero-fill bias)
- Reputation / blacklist integration (hosts/IPs)
- Add rate limiting or request logging
- Dockerfile for container deployment
- Unit tests for heuristics and feature extraction
- Probability calibration & threshold tuning set via config file

## 🧪 Troubleshooting
| Issue | Cause | Mitigation |
|-------|-------|------------|
| Legit site flagged | Over-aggressive heuristics | Raise thresholds / refine keywords |
| Phish missed | Threshold too high | Lower PHISHING_PROB_THRESHOLD or margin |
| Feature mismatch | Model expects more columns | Ensure `feature_columns` alignment (filled with 0) |

## 🔐 Security Disclaimer
This is an educational prototype, not a full production anti-phishing engine. Always layer with network, reputation, and content analysis defenses.

## 👩‍💻 Author
**Rupam Kumari**  
GitHub: https://github.com/techietinker01

## 📜 License
MIT License © 2025 Rupam Kumari. See `LICENSE` for full text.

---
If you build improvements (URL-only model, Docker, content fetcher), contributions are welcome—adapt freely.

