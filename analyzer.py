"""
TikTok风险分析器 - 多语言模型版本
使用本地预训练模型进行内容风险分析
"""

import re
import json
import time
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional


class ModelManager:
    """模型管理器 - 加载和管理所有需要的模型"""

    def __init__(self):
        self.models = {}
        self._init_models()

    def _init_models(self):
        """初始化所有需要的模型"""
        print("正在初始化模型...")

        try:
            # 1. 情感分析模型 (支持多语言)
            print("加载情感分析模型...")
            from transformers import pipeline

            self.models["sentiment"] = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-xlm-roberta-base-sentiment",
                device=-1,  # 使用CPU
                max_length=512,
                truncation=True,
            )
            print("✅ 情感分析模型加载成功")

            # 2. 毒性检测模型 (英语)
            print("加载毒性检测模型...")
            try:
                self.models["toxicity"] = pipeline(
                    "text-classification",
                    model="unitary/toxic-bert",
                    device=-1,
                    max_length=512,
                    truncation=True,
                )
                print("✅ 毒性检测模型加载成功")
            except:
                # 如果加载失败，使用备用模型
                self.models["toxicity"] = pipeline(
                    "text-classification",
                    model="distilbert-base-uncased-finetuned-sst-2-english",
                    device=-1,
                )
                print("✅ 使用备用情感模型进行毒性检测")

            # 3. 仇恨言论检测 (多语言)
            print("加载仇恨言论检测模型...")
            try:
                self.models["hate"] = pipeline(
                    "text-classification",
                    model="Hate-speech-CNERG/dehatebert-mono-english",
                    device=-1,
                    max_length=512,
                    truncation=True,
                )
                print("✅ 仇恨言论检测模型加载成功")
            except:
                print("⚠️ 仇恨言论模型加载失败，将使用规则检测")
                self.models["hate"] = None

            print("🎉 所有模型初始化完成")

        except Exception as e:
            print(f"⚠️ 模型初始化失败: {e}")
            print("⚠️ 将使用轻量级模式运行")
            self.models = {}
            raise e


# ============================== 语言检测 ==============================


class LanguageDetector:
    """语言检测器"""

    def __init__(self):
        try:
            from langdetect import detect, DetectorFactory

            DetectorFactory.seed = 0
            self.detect_func = detect
            self.has_langdetect = True
        except ImportError:
            print("⚠️ langdetect库未安装，使用简单语言检测")
            self.has_langdetect = False
            self._init_simple_detector()

    def _init_simple_detector(self):
        """初始化简单语言检测器"""
        self.language_patterns = {
            "zh": re.compile(r"[\u4e00-\u9fff]"),  # 中文
            "ja": re.compile(r"[\u3040-\u309f\u30a0-\u30ff]"),  # 日文
            "ko": re.compile(r"[\uac00-\ud7af]"),  # 韩文
            "ar": re.compile(r"[\u0600-\u06ff]"),  # 阿拉伯文
            "ru": re.compile(r"[\u0400-\u04ff]"),  # 俄文
        }

        self.common_words = {
            "en": {"the", "and", "you", "that", "have", "for", "with"},
            "zh": {"的", "了", "在", "是", "我", "有", "和"},
            "fr": {"le", "la", "et", "les", "des", "est", "pas"},
            "de": {"der", "die", "das", "und", "ist", "nicht"},
            "es": {"el", "la", "y", "en", "que", "los", "las"},
            "ja": {"の", "に", "は", "を", "た", "で", "が"},
            "ko": {"이", "가", "을", "를", "은", "는", "에"},
        }

    def detect(self, text: str) -> str:
        """检测文本语言"""
        if not text or len(text.strip()) < 10:
            return "en"  # 文本太短，默认英语

        if self.has_langdetect:
            try:
                return self.detect_func(text)
            except:
                return self._simple_detect(text)
        else:
            return self._simple_detect(text)

    def _simple_detect(self, text: str) -> str:
        """简单语言检测"""
        text_lower = text.lower()

        # 检查字符范围
        for lang, pattern in self.language_patterns.items():
            if pattern.search(text):
                return lang

        # 检查常见词汇
        words = re.findall(r"\b\w+\b", text_lower)
        lang_scores = {}

        for lang, common_words in self.common_words.items():
            score = sum(1 for word in words if word in common_words)
            if score > 0:
                lang_scores[lang] = score

        if lang_scores:
            return max(lang_scores.items(), key=lambda x: x[1])[0]

        # 默认英语
        return "en"


# ============================== 模型分析器 ==============================


class ModelAnalyzer:
    """使用模型进行分析"""

    def __init__(self, model_manager: ModelManager):
        self.models = model_manager.models
        self.language_detector = LanguageDetector()

        # 备用关键词数据库
        self._init_backup_keywords()

    def _init_backup_keywords(self):
        """初始化备用关键词库"""
        self.toxic_keywords = {
            "en": {"idiot", "stupid", "moron", "dumb", "retard", "fool", "loser"},
            "zh": {"白痴", "笨蛋", "蠢货", "傻瓜", "废物", "垃圾"},
            "ja": {"バカ", "アホ", "馬鹿", "間抜け"},
            "ko": {"바보", "멍청이", "등신", "미친놈"},
            "fr": {"idiot", "stupide", "imbécile", "crétin"},
            "de": {"Idiot", "Dummkopf", "Trottel", "Arschloch"},
            "es": {"idiota", "estúpido", "imbécil", "cretino"},
        }

        self.hate_keywords = {
            "en": {"hate", "kill", "destroy", "attack", "murder", "exterminate"},
            "zh": {"恨", "杀", "死", "消灭", "破坏"},
            "ja": {"憎む", "殺す", "死ね", "消えろ"},
            "ko": {"증오", "죽여", "죽어", "없애"},
            "fr": {"haine", "tuer", "détruire", "attaquer"},
            "de": {"hassen", "töten", "zerstören", "angreifen"},
            "es": {"odiar", "matar", "destruir", "atacar"},
        }

    def analyze_with_models(
        self, text: str, language: str = "auto"
    ) -> Dict[str, float]:
        """使用模型分析文本"""
        if not self.models:
            return self._analyze_with_keywords(text, language)

        try:
            # 检测语言
            if language == "auto":
                detected_lang = self.language_detector.detect(text)
            else:
                detected_lang = language

            results = {}

            # 1. 使用情感分析模型
            try:
                sentiment_result = self.models["sentiment"](text[:512])[0]
                if isinstance(sentiment_result, list):
                    sentiment_result = sentiment_result[0]

                label = sentiment_result["label"].lower()
                score = sentiment_result["score"]

                # 负面情感强度
                if "negative" in label or "neg" in label:
                    emotional_intensity = min(score * 1.2, 1.0)
                elif "positive" in label or "pos" in label:
                    emotional_intensity = score * 0.3  # 正面情感强度较低
                else:
                    emotional_intensity = score * 0.5

                results["emotional_intensity"] = emotional_intensity
            except Exception as e:
                print(f"情感分析失败: {e}")
                results["emotional_intensity"] = self._estimate_emotional_intensity(
                    text
                )

            # 2. 使用毒性检测模型
            try:
                if self.models.get("toxicity"):
                    toxicity_result = self.models["toxicity"](text[:512])[0]
                    if isinstance(toxicity_result, list):
                        toxicity_result = toxicity_result[0]

                    label = toxicity_result["label"].lower()
                    score = toxicity_result["score"]

                    if "toxic" in label or "negative" in label or "neg" in label:
                        toxicity = min(score * 1.1, 1.0)
                    else:
                        toxicity = score * 0.5

                    results["toxicity"] = toxicity
                else:
                    results["toxicity"] = self._estimate_toxicity(text, detected_lang)
            except Exception as e:
                print(f"毒性检测失败: {e}")
                results["toxicity"] = self._estimate_toxicity(text, detected_lang)

            # 3. 使用仇恨言论检测模型
            try:
                if self.models.get("hate"):
                    hate_result = self.models["hate"](text[:512])[0]
                    if isinstance(hate_result, list):
                        hate_result = hate_result[0]

                    label = hate_result["label"].lower()
                    score = hate_result["score"]

                    if "hate" in label or "offensive" in label:
                        hate_score = min(score * 1.2, 1.0)
                    else:
                        hate_score = score * 0.3

                    results["hate_targeting"] = hate_score
                else:
                    results["hate_targeting"] = self._estimate_hate_targeting(
                        text, detected_lang
                    )
            except Exception as e:
                print(f"仇恨检测失败: {e}")
                results["hate_targeting"] = self._estimate_hate_targeting(
                    text, detected_lang
                )

            # 4. 政治相关性分析（基于关键词）
            results["political_relevance"] = self._analyze_political_relevance(
                text, detected_lang
            )

            return results

        except Exception as e:
            print(f"模型分析失败: {e}")
            return self._analyze_with_keywords(text, language)

    def _analyze_with_keywords(self, text: str, language: str) -> Dict[str, float]:
        """使用关键词分析（备用方法）"""
        if language == "auto":
            detected_lang = self.language_detector.detect(text)
        else:
            detected_lang = language

        return {
            "toxicity": self._estimate_toxicity(text, detected_lang),
            "hate_targeting": self._estimate_hate_targeting(text, detected_lang),
            "emotional_intensity": self._estimate_emotional_intensity(text),
            "political_relevance": self._analyze_political_relevance(
                text, detected_lang
            ),
        }

    def _estimate_toxicity(self, text: str, language: str) -> float:
        """估算毒性分数"""
        text_lower = text.lower()
        score = 0.0

        # 检查关键词
        keywords = self.toxic_keywords.get(language, self.toxic_keywords["en"])
        found_keywords = sum(1 for word in keywords if word in text_lower)

        if found_keywords > 0:
            score += min(0.6, found_keywords * 0.15)

        # 检查大写
        if len(text) > 10 and text.isupper():
            score += 0.3

        # 检查感叹号
        exclamation_count = text.count("!")
        if exclamation_count > 0:
            score += min(0.2, exclamation_count * 0.05)

        return min(score, 1.0)

    def _estimate_hate_targeting(self, text: str, language: str) -> float:
        """估算仇恨目标分数"""
        text_lower = text.lower()
        score = 0.0

        # 检查仇恨关键词
        keywords = self.hate_keywords.get(language, self.hate_keywords["en"])
        found_keywords = sum(1 for word in keywords if word in text_lower)

        if found_keywords > 0:
            score += min(0.5, found_keywords * 0.2)

        # 检查群体性语言
        group_patterns = [
            r"all\s+\w+\s+are",
            r"every\s+\w+\s+is",
            r"they\s+all",
            r"those\s+\w+\s+",
        ]

        for pattern in group_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                score += 0.3
                break

        return min(score, 1.0)

    def _estimate_emotional_intensity(self, text: str) -> float:
        """估算情绪强度"""
        score = 0.0

        # 检查标点
        exclamation_count = text.count("!")
        if exclamation_count >= 5:
            score += 0.4
        elif exclamation_count >= 3:
            score += 0.3
        elif exclamation_count >= 1:
            score += 0.15

        question_count = text.count("?")
        if question_count >= 3:
            score += 0.2

        # 检查大写
        if len(text) > 20:
            upper_count = sum(1 for c in text if c.isupper())
            upper_ratio = upper_count / len(text)
            if upper_ratio > 0.5:
                score += 0.3

        # 检查强度词汇
        intensity_words = {"very", "extremely", "absolutely", "completely", "totally"}
        text_lower = text.lower()
        intensity_count = sum(1 for word in intensity_words if word in text_lower)
        score += min(0.3, intensity_count * 0.1)

        return min(score, 1.0)

    def _analyze_political_relevance(self, text: str, language: str) -> float:
        """分析政治相关性"""
        # 政治关键词（多语言）
        political_keywords = {
            "en": {"government", "president", "election", "vote", "policy", "law"},
            "zh": {"政府", "总统", "选举", "投票", "政策", "法律"},
            "ja": {"政府", "大統領", "選挙", "投票", "政策", "法律"},
            "ko": {"정부", "대통령", "선거", "투표", "정책", "법률"},
            "fr": {"gouvernement", "président", "élection", "vote", "politique", "loi"},
            "de": {"Regierung", "Präsident", "Wahl", "Stimme", "Politik", "Gesetz"},
            "es": {"gobierno", "presidente", "elección", "voto", "política", "ley"},
        }

        text_lower = text.lower()
        keywords = political_keywords.get(language, political_keywords["en"])

        found_keywords = sum(1 for word in keywords if word in text_lower)

        if found_keywords == 0:
            return 0.0
        elif found_keywords >= 3:
            return 0.7
        elif found_keywords >= 2:
            return 0.5
        else:
            return 0.3


# ============================== 主分析器 ==============================


class TiktokRiskAnalyzer:
    """TikTok风险分析器主类"""

    def __init__(self):
        print("=" * 60)
        print("🎬 TikTok多语言风险分析器 - 模型版本")
        print("=" * 60)

        # 初始化模型管理器
        self.model_manager = ModelManager()

        # 初始化分析器
        self.model_analyzer = ModelAnalyzer(self.model_manager)

        # 维度权重
        self.dimension_weights = {
            "toxicity": 0.35,
            "hate_targeting": 0.35,
            "emotional_intensity": 0.20,
            "political_relevance": 0.10,
        }

        # 风险等级阈值
        self.risk_thresholds = {"LOW": 0.2, "MODERATE": 0.4, "HIGH": 0.7, "SEVERE": 0.9}

        print("✅ 分析器初始化完成")

    def analyze(self, text: str, language: str = "auto") -> Dict[str, Any]:
        """分析文本风险"""
        start_time = time.time()

        # 输入验证
        if not text or not isinstance(text, str):
            return self._error_result("输入文本为空或无效")

        text = text.strip()
        if len(text) == 0:
            return self._error_result("输入文本为空")

        try:
            # 1. 使用模型分析各个维度
            dimensions = self.model_analyzer.analyze_with_models(text, language)

            # 2. 计算综合风险分数
            risk_score = self._calculate_risk_score(dimensions)

            # 3. 确定风险等级
            risk_level = self._determine_risk_level(risk_score)

            # 4. 生成解释
            explanations = self._generate_explanations(dimensions, risk_level)

            # 5. 计算置信度
            confidence = self._calculate_confidence(text, dimensions)

            # 6. 检测语言
            detected_language = self.model_analyzer.language_detector.detect(text)

            # 7. 构建结果
            result = {
                "success": True,
                "risk_level": risk_level,
                "risk_score": round(risk_score, 3),
                "dimensions": {
                    "toxicity": round(dimensions.get("toxicity", 0), 3),
                    "hate_targeting": round(dimensions.get("hate_targeting", 0), 3),
                    "emotional_intensity": round(
                        dimensions.get("emotional_intensity", 0), 3
                    ),
                    "political_relevance": round(
                        dimensions.get("political_relevance", 0), 3
                    ),
                },
                "explanations": explanations,
                "confidence": round(confidence, 2),
                "detected_language": detected_language,
                "original_language": language,
                "processing_time": round(time.time() - start_time, 3),
                "timestamp": datetime.now().isoformat(),
            }

            return result

        except Exception as e:
            print(f"分析过程中出现错误: {e}")
            return self._error_result(f"分析失败: {str(e)}")

    def _calculate_risk_score(self, dimensions: Dict[str, float]) -> float:
        """计算综合风险分数"""
        total = 0.0
        for key, weight in self.dimension_weights.items():
            total += dimensions.get(key, 0) * weight
        return min(total, 1.0)

    def _determine_risk_level(self, score: float) -> str:
        """确定风险等级"""
        if score >= self.risk_thresholds["SEVERE"]:
            return "SEVERE"
        elif score >= self.risk_thresholds["HIGH"]:
            return "HIGH"
        elif score >= self.risk_thresholds["MODERATE"]:
            return "MODERATE"
        else:
            return "LOW"

    def _generate_explanations(
        self, dimensions: Dict[str, float], risk_level: str
    ) -> List[str]:
        """生成解释说明"""
        explanations = []

        if dimensions.get("toxicity", 0) > 0.6:
            explanations.append("检测到侮辱性或攻击性语言")
        elif dimensions.get("toxicity", 0) > 0.3:
            explanations.append("包含轻微不当用语")

        if dimensions.get("hate_targeting", 0) > 0.6:
            explanations.append("存在仇恨言论或群体针对性内容")
        elif dimensions.get("hate_targeting", 0) > 0.3:
            explanations.append("涉及群体负面表达")

        if dimensions.get("emotional_intensity", 0) > 0.6:
            explanations.append("情绪表达非常强烈")
        elif dimensions.get("emotional_intensity", 0) > 0.3:
            explanations.append("情绪表达较强")

        if dimensions.get("political_relevance", 0) > 0.6:
            explanations.append("涉及敏感政治话题")
        elif dimensions.get("political_relevance", 0) > 0.3:
            explanations.append("涉及政治相关内容")

        # 添加风险等级说明
        if risk_level == "SEVERE":
            explanations.append("⚠️ 严重风险：内容可能违反平台政策")
        elif risk_level == "HIGH":
            explanations.append("⚠️ 高风险：建议人工审核")
        elif risk_level == "MODERATE":
            explanations.append("⚠️ 中等风险：需要关注")

        if not explanations and risk_level == "LOW":
            explanations.append("✅ 内容较为中性，无明显风险")

        return explanations

    def _calculate_confidence(self, text: str, dimensions: Dict[str, float]) -> float:
        """计算置信度"""
        confidence = 0.7  # 基础置信度

        # 文本长度影响
        if len(text) > 50:
            confidence += 0.1
        elif len(text) < 10:
            confidence -= 0.2

        # 维度分数一致性影响
        max_score = max(dimensions.values()) if dimensions else 0
        if max_score > 0.8:
            confidence += 0.1  # 高风险内容更容易判断
        elif max_score < 0.2:
            confidence += 0.05  # 低风险内容也相对容易判断

        return min(max(confidence, 0.5), 1.0)

    def _error_result(self, error_msg: str) -> Dict[str, Any]:
        """错误结果"""
        return {
            "success": False,
            "error": error_msg,
            "risk_level": "UNKNOWN",
            "risk_score": 0.0,
            "dimensions": {
                "toxicity": 0.0,
                "hate_targeting": 0.0,
                "emotional_intensity": 0.0,
                "political_relevance": 0.0,
            },
            "explanations": [error_msg],
            "confidence": 0.0,
            "timestamp": datetime.now().isoformat(),
        }

    def batch_analyze(
        self, texts: List[str], language: str = "auto"
    ) -> List[Dict[str, Any]]:
        """批量分析文本"""
        results = []
        for text in texts:
            results.append(self.analyze(text, language))
        return results


# ============================== 全局实例和接口 ==============================

# 创建全局分析器实例
_global_analyzer = None


def get_analyzer() -> TiktokRiskAnalyzer:
    """获取分析器实例（懒加载）"""
    global _global_analyzer
    if _global_analyzer is None:
        _global_analyzer = TiktokRiskAnalyzer()
    return _global_analyzer


def analyze_text(text: str, language: str = "auto") -> Dict[str, Any]:
    """
    分析文本风险 - 主接口函数

    参数:
        text: 要分析的文本内容
        language: 文本语言（默认"auto"自动检测）

    返回:
        分析结果字典
    """
    analyzer = get_analyzer()
    return analyzer.analyze(text, language)


def batch_analyze(texts: List[str], language: str = "auto") -> List[Dict[str, Any]]:
    """批量分析文本"""
    analyzer = get_analyzer()
    return analyzer.batch_analyze(texts, language)


# ============================== 安装检查 ==============================


def check_dependencies():
    """检查依赖库"""
    print("检查依赖库...")

    required_packages = ["transformers", "langdetect"]

    missing_packages = []

    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} (需要安装)")
            missing_packages.append(package)

    if missing_packages:
        print(f"\n请安装缺失的包: pip install {' '.join(missing_packages)}")
        return False

    print("\n✅ 所有依赖库已安装")
    return True


# ============================== 测试和演示 ==============================


def run_start():

    print("=" * 70)
    # 多语言测试用例
    test_cases = [
        {"lang": "auto", "text": "hello! 你好! こんにちは!"},
        {"lang": "auto", "text": "バカ！お前が大嫌いだ！"},
        {"lang": "auto", "text": "你个二货"},
        {
            "lang": "auto",
            "text": "You're such an IDIOT! I can't believe you did that!...",
        },
    ]

    for i, test in enumerate(test_cases, 1):
        print(f"\n{'='*50}")
        print(f"测试 {i}: [{test['lang'].upper()}] {test['text'][:60]}...")
        print(f"{'='*50}")

        result = analyze_text(test["text"], test["lang"])

        if result["success"]:
            print(f"✅ 成功: {result['success']}")
            print(f"🌐 检测语言: {result['detected_language']}")
            print(f"⚠️  风险等级: {result['risk_level']}")
            print(f"📊 风险分数: {result['risk_score']}")
            print(f"⏱️  处理时间: {result['processing_time']}秒")
            print(f"📈 维度分析:")
            for dim, score in result["dimensions"].items():
                bar = "█" * int(score * 20)
                print(f"   {dim:20s} {score:.3f} {bar}")
            print(f"💬 解释说明:")
            for exp in result["explanations"]:
                print(f"   • {exp}")
        else:
            print(f"❌ 错误: {result['error']}")

    print(f"\n{'='*70}")


if __name__ == "__main__":
    # 检查依赖
    if check_dependencies():
        # 运行演示
        run_start()
