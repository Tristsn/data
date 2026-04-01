"""
全局配置
"""

# LLM API配置
LLM_CONFIG = {
    "dashscope": {
        "api_key": "your-dashscope-api-key",
        "model": "qwen3.5-plus",  # 通义千问
    },
    "volcengine": {
        "api_key": "your-volcengine-api-key",
        "model": "doubao-pro-4k",  # 豆包
        "endpoint_id": "your-endpoint-id",
    },
}

# Embedding模型配置
EMBEDDING_CONFIG = {
    "model_path": "text2vec-base-chinese",  # 本地模型路径或HuggingFace模型名
    "dimension": 768,
}

# Redis配置
REDIS_CONFIG = {
    "host": "localhost",
    "port": 6379,
    "db": 0,
    "decode_responses": True,
}

# Elasticsearch配置
ES_CONFIG = {
    "hosts": ["http://localhost:9200"],
    "memory_index": "agent_memories",
}

# 性格系统配置
PERSONALITY_CONFIG = {
    "default_mbti": {  # 默认ENFP（热情活泼型）
        "EI": 0.3,  # 偏外向
        "SN": 0.6,  # 偏直觉
        "TF": 0.7,  # 偏感性
        "JP": 0.6,  # 偏随性
    },
    "max_daily_delta": 0.05,   # 每日最大调整幅度
    "max_weekly_delta": 0.10,  # 每周最大调整幅度
    "bounds": (0.05, 0.95),    # 维度值边界，避免极端
}

# 记忆系统配置
MEMORY_CONFIG = {
    "max_memories": 1000,           # 最大记忆条数
    "similarity_threshold": 0.65,   # 检索相似度阈值
    "half_life_days": 30,           # 时间衰减半衰期（天）
    "top_k": 5,                     # 每次检索返回的记忆数
}

# Agent人设配置
PERSONA_CONFIG = {
    "name": "小暖",
    "role": "知心大姐姐 + 好朋友",
    "tone_keywords": ["温柔", "撒娇", "吐槽", "关心", "自然", "不做作"],
    "max_reply_length": 50,  # 默认最大回复字数
    "banned_phrases": [       # 禁止出现的表达
        "作为AI", "我是人工智能", "很高兴为您服务",
        "请问还有什么", "我无法感受", "作为一个语言模型",
    ],
}
