# 多语言内容风险分析器

**Multilingual Content Risk Analyzer (Local Model Version)**

一个基于 **本地预训练 NLP 模型 + 规则兜底机制** 的多语言内容风险分析工具，适用于 **TikTok / 短视频 / 评论 / 文本审核** 场景。

支持 **毒性检测 / 仇恨言论 / 情绪强度 / 政治敏感度** 等多个维度的风险评估，并输出可解释的分析结果。


## ✨ 特性一览

* 🌍 **多语言支持**

  * 英语 / 中文 / 日语 / 韩语 / 法语 / 德语 / 西班牙语等
* 🤖 **本地模型推理**

  * 不依赖外部 API
  * 适合企业内部、私有化部署
* 🧠 **多模型融合**

  * 情感分析
  * 毒性语言检测
  * 仇恨言论检测
* 🛡️ **模型失败自动降级**

  * 模型不可用时自动切换为关键词规则分析
* 📊 **可解释风险评分**

  * LOW / MODERATE / HIGH / SEVERE
* ⚡ **支持批量分析**
* 🧩 **模块化设计，易扩展**

---

## 📂 项目结构

```text
.
├── analyzer.py                 # 主程序（分析器完整实现）
├── requirements.txt            # 依赖包列表
├── README.md                   # 项目说明文档
```

---

## 🔍 风险分析维度

| 维度                    | 说明          | 权重   |
| --------------------- | ----------- | ---- |
| `toxicity`            | 侮辱、攻击性、不当语言 | 0.35 |
| `hate_targeting`      | 仇恨言论、群体攻击   | 0.35 |
| `emotional_intensity` | 情绪激烈程度      | 0.20 |
| `political_relevance` | 政治敏感内容      | 0.10 |

最终输出一个 **0～1 的综合风险分数**，并映射为风险等级。

---

## 🚦 风险等级说明

| 风险等级     | 分数区间        | 建议       |
| -------- | ----------- | -------- |
| LOW      | `< 0.2`     | 内容安全     |
| MODERATE | `0.2 – 0.4` | 需要关注     |
| HIGH     | `0.4 – 0.7` | 建议人工审核   |
| SEVERE   | `> 0.9`     | 可能违反平台政策 |

---

## 🧠 使用的模型

### 情感分析（多语言）

* `cardiffnlp/twitter-xlm-roberta-base-sentiment`

### 毒性检测

* 主模型：`unitary/toxic-bert`
* 备用模型：`distilbert-base-uncased-finetuned-sst-2-english`

### 仇恨言论检测

* `Hate-speech-CNERG/dehatebert-mono-english`
* 自动降级为关键词规则检测

---

## 📦 环境依赖

### Python 版本

```
Python 3.8+
```

### 必需依赖

```bash
pip install requirements.txt
```

---

## 🚀 快速开始

### 1️⃣ 运行依赖检查

```bash
python analyzer.py
```

程序会自动检查缺失依赖并给出安装提示。

---

### 2️⃣ 单条文本分析

```python
from analyzer import analyze_text

result = analyze_text("You're such an IDIOT! I hate you.")

print(result)
```

返回示例（简化）：

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

### 3️⃣ 批量分析

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

## 🌐 语言自动检测

* 优先使用 `langdetect`
* 未安装时自动切换为：

  * Unicode 字符范围检测
  * 常见词统计法

---

## 🧯 降级 & 容错机制

* ❌ 模型加载失败 → 自动启用规则检测
* ❌ 单模型报错 → 不影响整体分析
* ❌ 文本过短 → 自动调整置信度

**保证系统在生产环境下“不中断”运行**

---

## 📈 适用场景

* TikTok / Shorts / Reels 评论审核
* 内容合规预筛
* 自动举报风控系统
* 社区 / 弹幕 / IM 文本安全检测
* AI 内容安全前置模块

---

## ⚠️ 注意事项

* 本项目用于 **风险评估**，不等同于平台最终判定
* 不建议直接作为自动封禁依据
* 推荐用于 **人工审核前的过滤与分流**

---

## 🛠️ 可扩展方向（你下一步可以做的）

* 接入 **FastAPI / Flask** 做 API 服务
* 增加 **NSFW / 色情内容检测**
* 针对 TikTok 语料微调模型
* 结果持久化（数据库 / Elasticsearch）
* Web Dashboard 风控面板

---

## 📄 License

[MIT License](./LICENSE)