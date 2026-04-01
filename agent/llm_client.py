"""
LLM调用模块 — 封装DashScope(通义千问)和火山引擎(豆包)API
"""

from typing import List, Dict, Optional
from config import LLM_CONFIG


def call_dashscope(messages: List[Dict[str, str]],
                   temperature: float = 0.8,
                   max_tokens: int = 256) -> str:
    """
    调用阿里云DashScope API（通义千问）

    messages格式: [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
    """
    import dashscope
    from dashscope import Generation

    dashscope.api_key = LLM_CONFIG["dashscope"]["api_key"]

    response = Generation.call(
        model=LLM_CONFIG["dashscope"]["model"],
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        result_format="message",
    )

    if response.status_code == 200:
        return response.output.choices[0].message.content
    else:
        raise RuntimeError(f"DashScope API error: {response.code} - {response.message}")


def call_volcengine(messages: List[Dict[str, str]],
                    temperature: float = 0.8,
                    max_tokens: int = 256) -> str:
    """
    调用火山引擎API（豆包）
    """
    from volcenginesdkarkruntime import Ark

    client = Ark(api_key=LLM_CONFIG["volcengine"]["api_key"])

    response = client.chat.completions.create(
        model=LLM_CONFIG["volcengine"]["endpoint_id"],
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    return response.choices[0].message.content


class LLMClient:
    """统一的LLM调用入口，支持主备切换"""

    def __init__(self, primary: str = "dashscope", fallback: str = "volcengine"):
        self.primary = primary
        self.fallback = fallback
        self._callers = {
            "dashscope": call_dashscope,
            "volcengine": call_volcengine,
        }

    def chat(self, messages: List[Dict[str, str]],
             temperature: float = 0.8,
             max_tokens: int = 256) -> str:
        """调用LLM，主模型失败时自动切换备用模型"""
        try:
            return self._callers[self.primary](messages, temperature, max_tokens)
        except Exception as e:
            print(f"[LLM] {self.primary} failed: {e}, switching to {self.fallback}")
            return self._callers[self.fallback](messages, temperature, max_tokens)

    def analyze_emotion(self, user_text: str) -> str:
        """用LLM分析用户文本情绪（方案A：零额外开销）"""
        messages = [
            {"role": "system", "content": (
                "你是一个情绪分析助手。请分析用户这句话的情绪，"
                "只返回一个词：开心/悲伤/愤怒/疲惫/平静/兴奋。不要解释。"
            )},
            {"role": "user", "content": user_text},
        ]
        result = self.chat(messages, temperature=0.1, max_tokens=10)
        # 兜底
        valid = {"开心", "悲伤", "愤怒", "疲惫", "平静", "兴奋"}
        return result.strip() if result.strip() in valid else "平静"

    def evaluate_memory_importance(self, dialogue: str) -> int:
        """用LLM评估对话的记忆重要性（0-10）"""
        messages = [
            {"role": "system", "content": (
                "你是一个记忆重要性评估助手。请评估以下对话内容对于长期记忆的重要性，"
                "返回0-10的整数。10=非常重要（重大事件、强烈情感），0=完全不重要（闲聊寒暄）。"
                "只返回数字，不要解释。"
            )},
            {"role": "user", "content": dialogue},
        ]
        result = self.chat(messages, temperature=0.1, max_tokens=5)
        try:
            score = int(result.strip())
            return max(0, min(10, score))
        except ValueError:
            return 5  # 默认中等重要性

    def generate_daily_summary(self, dialogues: List[str],
                               personality_desc: str) -> Dict[str, str]:
        """生成每日总结和内心OS"""
        dialogue_text = "\n".join(f"- {d}" for d in dialogues[-20:])  # 取最近20条

        messages = [
            {"role": "system", "content": (
                f"你是一个智能家居伙伴，{personality_desc}\n"
                "请根据今天的对话记录，生成：\n"
                "1. 【今日总结】：用第三人称简要概括今天和主人的互动（50字以内）\n"
                "2. 【内心OS】：用第一人称写你的内心独白，要符合你的性格（80字以内）\n"
                "3. 【互动特征】：用JSON格式输出以下四个0-1的数值：\n"
                '   {"emotional_depth": 0.x, "initiative_ratio": 0.x, '
                '"creative_topics": 0.x, "structured_requests": 0.x}\n'
                "请严格按照上述格式输出。"
            )},
            {"role": "user", "content": f"今天的对话记录：\n{dialogue_text}"},
        ]

        result = self.chat(messages, temperature=0.7, max_tokens=500)
        return {"raw": result}
