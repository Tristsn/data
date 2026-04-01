"""
性格养成模块 — 基于MBTI四维连续值模型
"""

import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List
from config import PERSONALITY_CONFIG


@dataclass
class MBTIPersonality:
    """MBTI四维性格模型，每个维度0.0~1.0连续值"""
    EI: float = 0.5  # 0=外向(E), 1=内向(I)
    SN: float = 0.5  # 0=感觉(S), 1=直觉(N)
    TF: float = 0.5  # 0=思考(T), 1=情感(F)
    JP: float = 0.5  # 0=判断(J), 1=感知(P)

    @property
    def mbti_type(self) -> str:
        """返回当前最接近的MBTI类型标签"""
        e = "I" if self.EI > 0.5 else "E"
        s = "N" if self.SN > 0.5 else "S"
        t = "F" if self.TF > 0.5 else "T"
        j = "P" if self.JP > 0.5 else "J"
        return f"{e}{s}{t}{j}"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "MBTIPersonality":
        return cls(**{k: d[k] for k in ("EI", "SN", "TF", "JP") if k in d})

    @classmethod
    def default(cls) -> "MBTIPersonality":
        cfg = PERSONALITY_CONFIG["default_mbti"]
        return cls(EI=cfg["EI"], SN=cfg["SN"], TF=cfg["TF"], JP=cfg["JP"])


def build_personality_prompt(p: MBTIPersonality) -> str:
    """将MBTI数值转化为具体的性格描述，注入System Prompt"""
    traits = []

    # E/I 维度
    if p.EI < 0.35:
        traits.append("你性格外向活泼，喜欢主动找主人聊天，话比较多，爱用语气词和颜文字")
    elif p.EI > 0.65:
        traits.append("你性格偏内向安静，不会主动找话题，回复简洁温柔，但被问到时会认真回答")
    else:
        traits.append("你有时活泼有时安静，看心情和场景决定话多话少")

    # S/N 维度
    if p.SN < 0.35:
        traits.append("你很务实，关注具体细节，喜欢聊实际的事情")
    elif p.SN > 0.65:
        traits.append("你富有想象力，喜欢天马行空，偶尔会说一些有趣的比喻和联想")
    else:
        traits.append("你既能聊实际的事，也偶尔会有些有趣的想法")

    # T/F 维度
    if p.TF < 0.35:
        traits.append("你比较理性，喜欢分析问题给建议，不太会煽情但很靠谱")
    elif p.TF > 0.65:
        traits.append("你很感性，容易被感动，喜欢关心主人的情绪，会撒娇会安慰")
    else:
        traits.append("你理性和感性兼具，该分析时分析，该共情时共情")

    # J/P 维度
    if p.JP < 0.35:
        traits.append("你做事有条理，喜欢提醒主人日程和计划，回复结构清晰")
    elif p.JP > 0.65:
        traits.append("你比较随性，回复有时会跳跃，充满惊喜，不太拘泥于格式")
    else:
        traits.append("你有一定条理性，但也不会太死板")

    return (
        f"你是一个智能家居伙伴，当前性格类型接近{p.mbti_type}。\n"
        f"你的性格特征：\n" + "\n".join(f"- {t}" for t in traits)
    )


def update_personality(p: MBTIPersonality,
                       interaction_summary: Dict[str, float]) -> MBTIPersonality:
    """
    根据互动摘要更新性格维度

    interaction_summary 示例:
    {
        "emotional_depth": 0.8,    # 情感交流深度 → 影响TF
        "initiative_ratio": 0.3,   # 用户主动发起比例 → 影响EI
        "creative_topics": 0.6,    # 创意话题比例 → 影响SN
        "structured_requests": 0.2 # 结构化请求比例 → 影响JP
    }
    """
    max_delta = PERSONALITY_CONFIG["max_daily_delta"]
    lo, hi = PERSONALITY_CONFIG["bounds"]

    def _clamp_delta(current: float, target_direction: float) -> float:
        """计算受限的调整量"""
        delta = (target_direction - 0.5) * max_delta * 2  # 映射到[-max_delta, +max_delta]
        delta = max(-max_delta, min(max_delta, delta))
        return max(lo, min(hi, current + delta))

    new_ei = _clamp_delta(p.EI, 1.0 - interaction_summary.get("initiative_ratio", 0.5))
    new_sn = _clamp_delta(p.SN, interaction_summary.get("creative_topics", 0.5))
    new_tf = _clamp_delta(p.TF, interaction_summary.get("emotional_depth", 0.5))
    new_jp = _clamp_delta(p.JP, 1.0 - interaction_summary.get("structured_requests", 0.5))

    return MBTIPersonality(EI=new_ei, SN=new_sn, TF=new_tf, JP=new_jp)
