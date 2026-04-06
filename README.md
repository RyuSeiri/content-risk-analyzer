# Multilingual Content Risk Analyzer

**Multilingual Content Risk Analyzer (Local Model Version)**

English | [简体中文](./README_zh.md) | [日本語](./README_jp.md)

A multilingual content risk analysis tool based on **local pretrained NLP models + rule-based fallback mechanisms**, designed for **TikTok / short video / comments / text moderation** scenarios.

Supports risk assessment across multiple dimensions including **toxicity detection / hate speech / emotional intensity / political sensitivity**, with explainable analysis results.


## ✨ Features

* 🌍 **Multi-language Support**

  * English / Chinese / Japanese / Korean / French / German / Spanish, etc.
* 🤖 **Local Model Inference**

  * No external API dependencies
  * Suitable for enterprise internal deployment and private deployment
* 🧠 **Multi-model Fusion**

  * Sentiment analysis
  * Toxic language detection
  * Hate speech detection
* 🛡️ **Automatic Model Failure Degradation**

  * Automatically switches to keyword rule analysis when models are unavailable
* 📊 **Explainable Risk Scoring**

  * LOW / MODERATE / HIGH / SEVERE
* ⚡ **Batch Analysis Support**
* 🧩 **Modular Design, Easy to Extend**

---

## 📂 Project Structure

```text
.
├── analyzer.py                 # Main program (complete analyzer implementation)
├── requirements.txt            # Dependency list
├── README.md                   # Project documentation
```

---

## 🔍 Risk Analysis Dimensions

| Dimension            | Description                     | Weight |
| -------------------- | ------------------------------- | ------ |
| `toxicity`           | Offensive, aggressive, inappropriate language | 0.35 |
| `hate_targeting`     | Hate speech, group targeting    | 0.35 |
| `emotional_intensity` | Emotional intensity             | 0.20 |
| `political_relevance` | Political sensitive content     | 0.10 |

Outputs a **comprehensive risk score from 0 to 1**, which is then mapped to a risk level.

---

## 🚦 Risk Levels

| Risk Level | Score Range | Recommendation       |
| ---------- | ----------- | -------------------- |
| LOW        | `< 0.2`     | Content is safe      |
| MODERATE   | `0.2 – 0.4` | Requires monitoring  |
| HIGH       | `0.4 – 0.7` | Manual review needed |
| SEVERE     | `> 0.9`     | Likely policy breach |

---

## 🧠 Models Used

### Sentiment Analysis (Multilingual)

* `cardiffnlp/twitter-xlm-roberta-base-sentiment`

### Toxicity Detection

* Primary model: `unitary/toxic-bert`
* Backup model: `distilbert-base-uncased-finetuned-sst-2-english`

### Hate Speech Detection

* `Hate-speech-CNERG/dehatebert-mono-english`
* Automatically degrades to keyword rule detection

---

## 📦 Environment Dependencies

### Python Version

```
Python 3.8+
```

### Required Dependencies

```bash
pip install requirements.txt
```

---

## 🚀 Quick Start

### 1️⃣ Run Dependency Check

```bash
python analyzer.py
```

The program will automatically check for missing dependencies and provide installation prompts.

---

### 2️⃣ Single Text Analysis

```python
from analyzer import analyze_text

result = analyze_text("You're such an IDIOT! I hate you.")

print(result)
```

Example output (simplified):

```json
{
  "risk_level": "HIGH",
  "risk_score": 0.62,
  "dimensions": {
    "toxicity": 0.81,
    "hate_targeting": 0.55,
    "emotional_intensity": 0.74,
    "political_relevance": 0.0
  },
  "detected_language": "en",
  "confidence": 0.85
}
```

---

### 3️⃣ Batch Analysis

```python
from analyzer import batch_analyze

texts = [
    "hello!",
    "バカ！お前が大嫌いだ！",
    "你个二货"
]

results = batch_analyze(texts)
```

---

## 🌐 Language Auto-detection

* Priority use of `langdetect`
* Automatically switches to the following when not installed:

  * Unicode character range detection
  * Common word statistics method

---

## 🧯 Degradation & Fault Tolerance

* ❌ Model loading failure → Automatically enable rule detection
* ❌ Single model error → Does not affect overall analysis
* ❌ Text too short → Automatically adjust confidence

**Ensures operation in production environments“non-interrupted”operation**

---

## 📈 Use Cases

* TikTok / Shorts / Reels comment moderation
* Content compliance pre-screening
* Automated reporting and risk control system
* Community / bullet comments / IM text safety detection
* AI content safety pre-filtering module

---

## ⚠️ Notes

* This project is for **risk assessment** and is not equivalent to the platform's final decision
* Not recommended for direct use as a basis for automatic banning
* Recommended for **filtering and triage before manual review**

---

## 📄 License

[MIT License](./LICENSE)