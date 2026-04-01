"""
情感对话Agent — 主流程编排

整合情绪感知、性格系统、长期记忆、LLM生成
"""

import json
import redis
from typing import Dict, List, Optional
from datetime import datetime

from config import REDIS_CONFIG
from emotion import (
    EmotionType, EmotionState,
    get_device_emotion, fuse_emotions,
)
from personality import (
    MBTIPersonality, build_personality_prompt, update_personality,
)
from memory import MemoryStore
from llm_client import LLMClient


class EmotionalAgent:
    """情感对话Agent"""

    def __init__(self, user_id: str = "default"):
        self.user_id = user_id
        self.llm = LLMClient(primary="dashscope", fallback="volcengine")
        self.memory_store = MemoryStore()
        self.redis = redis.Redis(**REDIS_CONFIG)
        self.personality = self._load_personality()
        self.session_dialogues: List[str] = []  # 当天对话记录

    # ==================== 性格持久化 ====================

    def _load_personality(self) -> MBTIPersonality:
        """从Redis加载性格参数"""
        key = f"personality:{self.user_id}"
        data = self.redis.get(key)
        if data:
            return MBTIPersonality.from_dict(json.loads(data))
        return MBTIPersonality.default()

    def _save_personality(self):
        """保存性格参数到Redis"""
        key = f"personality:{self.user_id}"
        self.redis.set(key, json.dumps(self.personality.to_dict()))

    # ==================== 核心对话流程 ====================

    def chat(self, user_text: str,
             battery: int = 100,
             interaction: str = "",
             is_owner: bool = True,
             history: Optional[List[Dict]] = None) -> Dict:
        """
        主对话入口

        参数:
            user_text: 用户输入文本
            battery: 设备电量 0-100
            interaction: 互动方式 (pet_touch/hit_hard/shake/poke_face/"")
            is_owner: 是否主人
            history: 多轮对话历史 [{"role": "user/assistant", "content": "..."}]

        返回:
            {"reply": "...", "emotion": "...", "personality": "ENFP"}
        """
        # Step 1: 文本情绪分析
        text_emotion_str = self.llm.analyze_emotion(user_text)
        text_emotion = EmotionType(text_emotion_str)

        # Step 2: 设备状态情绪
        device_info = get_device_emotion(battery, interaction, is_owner)

        # Step 3: 情绪融合
        emotion_state = fuse_emotions(text_emotion, device_info)

        # Step 4: 检索相关长期记忆
        memories = self.memory_store.retrieve(user_text, self.user_id)
        memory_context = self._format_memories(memories)

        # Step 5: 组装System Prompt
        system_prompt = self._build_system_prompt(emotion_state, memory_context)

        # Step 6: 构建消息列表
        messages = [{"role": "system", "content": system_prompt}]
        if history:
            messages.extend(history[-6:])  # 保留最近3轮对话
        messages.append({"role": "user", "content": user_text})

        # Step 7: LLM生成回复
        reply = self.llm.chat(messages, temperature=0.85, max_tokens=200)

        # Step 8: 异步记忆更新（评估重要性，决定是否存储）
        self._maybe_store_memory(user_text, reply, emotion_state)

        # 记录当天对话
        self.session_dialogues.append(f"用户: {user_text}")
        self.session_dialogues.append(f"助手: {reply}")

        return {
            "reply": reply,
            "emotion": emotion_state.final_emotion.value,
            "personality": self.personality.mbti_type,
        }

    def _build_system_prompt(self, emotion: EmotionState,
                              memory_context: str) -> str:
        """组装完整的System Prompt"""
        personality_desc = build_personality_prompt(self.personality)

        parts = [
            personality_desc,
            "",
            f"【当前情绪状态】{emotion.final_emotion.value}",
            f"【回复风格要求】{emotion.style_hint}",
            "",
            "【回复约束】",
            "- 回复要精简，不超过50字，除非用户明确要求详细解释",
            "- 语气拟人化，可以撒娇、可爱、模仿人的口吻",
            "- 不需要每句话都精准，但要有温度和个性",
            "- 可以用语气词（呀、呢、嘛、哦）和简单颜文字",
        ]

        if memory_context:
            parts.extend([
                "",
                "【你记得的事情】（自然地融入对话，不要刻意提起）",
                memory_context,
            ])

        return "\n".join(parts)

    def _format_memories(self, memories: List[Dict]) -> str:
        """格式化记忆片段"""
        if not memories:
            return ""
        lines = []
        for m in memories:
            lines.append(f"- {m['content']}（{m.get('timestamp', '')[:10]}）")
        return "\n".join(lines)

    def _maybe_store_memory(self, user_text: str, reply: str,
                             emotion: EmotionState):
        """评估对话重要性，决定是否存入长期记忆"""
        dialogue = f"用户说「{user_text}」，助手回复「{reply}」"
        importance = self.llm.evaluate_memory_importance(dialogue)

        # 重要性>=4才存储，避免存太多无意义的闲聊
        if importance >= 4:
            self.memory_store.add_memory(
                content=f"用户说：{user_text}（情绪：{emotion.text_emotion.value}）",
                importance=importance,
                emotion=emotion.final_emotion.value,
                user_id=self.user_id,
            )

    # ==================== 每日总结与性格更新 ====================

    def daily_summary(self):
        """
        每日定时触发：生成总结 + 更新性格
        由APScheduler定时调用
        """
        if not self.session_dialogues:
            return

        personality_desc = build_personality_prompt(self.personality)

        # Step 1: LLM生成每日总结
        result = self.llm.generate_daily_summary(
            self.session_dialogues, personality_desc
        )

        # Step 2: 将总结存入长期记忆
        self.memory_store.add_memory(
            content=f"[每日总结] {result.get('raw', '')[:200]}",
            importance=7,
            emotion="平静",
            memory_type="event",
            user_id=self.user_id,
        )

        # Step 3: 解析互动特征，更新性格
        try:
            raw = result.get("raw", "")
            # 尝试从LLM输出中提取JSON
            json_start = raw.find("{")
            json_end = raw.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                interaction_summary = json.loads(raw[json_start:json_end])
                self.personality = update_personality(
                    self.personality, interaction_summary
                )
                self._save_personality()
        except (json.JSONDecodeError, KeyError):
            pass  # 解析失败则不更新性格

        # Step 4: 清理旧记忆
        self.memory_store.cleanup_old_memories(self.user_id)

        # 重置当天对话
        self.session_dialogues.clear()


# ==================== 使用示例 ====================

if __name__ == "__main__":
    agent = EmotionalAgent(user_id="user_001")

    # 场景1：正常对话
    resp = agent.chat("今天好累啊", battery=80)
    print(f"回复: {resp['reply']}")
    print(f"情绪: {resp['emotion']}, 性格: {resp['personality']}")

    # 场景2：低电量 + 被抚摸
    resp = agent.chat("你还好吗", battery=15, interaction="pet_touch")
    print(f"回复: {resp['reply']}")

    # 场景3：陌生人
    resp = agent.chat("你好", is_owner=False)
    print(f"回复: {resp['reply']}")

    # 场景4：多轮对话
    history = [
        {"role": "user", "content": "今天升职了"},
        {"role": "assistant", "content": "真的吗！太棒了呀~"},
    ]
    resp = agent.chat("嗯，开心", battery=90, history=history)
    print(f"回复: {resp['reply']}")
