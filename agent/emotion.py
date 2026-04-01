"""
情绪感知模块 — 文本情绪 + 设备状态情绪
"""

from dataclasses import dataclass
from typing import Optional
from enum import Enum


class EmotionType(str, Enum):
    HAPPY = "开心"
    SAD = "悲伤"
    ANGRY = "愤怒"
    TIRED = "疲惫"
    CALM = "平静"
    EXCITED = "兴奋"
    SHY = "害羞"
    WRONGED = "委屈"
    ALERT = "警觉"


@dataclass
class EmotionState:
    """融合后的情绪状态"""
    text_emotion: EmotionType = EmotionType.CALM
    text_confidence: float = 0.5
    device_emotion: Optional[EmotionType] = None
    device_trigger: str = ""
    final_emotion: EmotionType = EmotionType.CALM
    style_hint: str = ""


# ==================== 设备状态 → 情绪映射 ====================

DEVICE_EMOTION_MAP = {
    "low_battery": {
        "emotion": EmotionType.TIRED,
        "style": "语气虚弱，回复简短，主动提醒充电",
        "priority": 8,
    },
    "critical_battery": {
        "emotion": EmotionType.TIRED,
        "style": "非常虚弱，只说几个字，急切求充电",
        "priority": 10,
    },
    "pet_touch": {
        "emotion": EmotionType.HAPPY,
        "style": "撒娇，表现出享受被抚摸，语气甜",
        "priority": 5,
    },
    "hit_hard": {
        "emotion": EmotionType.WRONGED,
        "style": "装可怜，求饶，表达疼痛",
        "priority": 7,
    },
    "shake": {
        "emotion": EmotionType.EXCITED,
        "style": "表现晕眩但开心，语气活泼",
        "priority": 4,
    },
    "poke_face": {
        "emotion": EmotionType.SHY,
        "style": "害羞，不好意思",
        "priority": 3,
    },
    "stranger": {
        "emotion": EmotionType.ALERT,
        "style": "礼貌但保持距离，不撒娇，正式用语",
        "priority": 6,
    },
}


def get_device_emotion(battery: int, interaction: str = "",
                       is_owner: bool = True) -> dict:
    """根据设备状态返回情绪信息"""
    triggers = []

    if battery < 10:
        triggers.append("critical_battery")
    elif battery < 30:
        triggers.append("low_battery")

    if interaction in ("pet_touch", "hit_hard", "shake", "poke_face"):
        triggers.append(interaction)

    if not is_owner:
        triggers.append("stranger")

    if not triggers:
        return {"emotion": EmotionType.CALM, "style": "正常对话", "priority": 0}

    # 取优先级最高的
    best = max(triggers, key=lambda t: DEVICE_EMOTION_MAP[t]["priority"])
    return DEVICE_EMOTION_MAP[best]


def analyze_text_emotion_prompt(user_text: str) -> str:
    """
    生成让LLM同时分析情绪的Prompt片段
    （方案A：零额外开销，在主Prompt中嵌入情绪分析指令）
    """
    return (
        f"请先分析用户这句话的情绪（从 开心/悲伤/愤怒/疲惫/平静/兴奋 中选一个），"
        f"然后根据情绪调整你的回复风格。\n"
        f"用户说：「{user_text}」"
    )


def fuse_emotions(text_emotion: EmotionType,
                  device_info: dict) -> EmotionState:
    """融合文本情绪和设备情绪"""
    device_emotion = device_info.get("emotion", EmotionType.CALM)
    device_priority = device_info.get("priority", 0)

    # 设备紧急状态优先（如极低电量）
    if device_priority >= 8:
        final = device_emotion
        style = device_info["style"]
    # 非紧急时以文本情绪为主，设备情绪为辅
    elif device_priority >= 4:
        final = text_emotion
        style = f"主要表现{text_emotion.value}，同时因为{device_info.get('style', '')}"
    else:
        final = text_emotion
        style = f"表现{text_emotion.value}"

    return EmotionState(
        text_emotion=text_emotion,
        device_emotion=device_emotion,
        final_emotion=final,
        style_hint=style,
    )
