"""
长期记忆模块 — 基于Elasticsearch向量检索

选用Elasticsearch而非Milvus/FAISS的理由：
1. 混合检索能力：ES同时支持向量检索(kNN)和关键词检索(BM25)，
   记忆场景需要语义相似+关键词精确匹配的混合能力
   （如用户提到"猫"时，既要语义匹配"宠物"相关记忆，也要精确匹配提过"猫"的记忆）
2. 结构化过滤：ES原生支持对timestamp、importance、type等字段的过滤和排序，
   记忆检索需要时间衰减加权、重要性过滤等复杂条件，纯向量库需要额外处理
3. 运维成本低：项目已有ES基础设施（NLU系统在用），无需额外引入新组件
4. 聚合分析：ES的聚合能力可用于每日总结（统计当天情绪分布、话题频次等）
"""

import time
import math
import json
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict
from datetime import datetime

from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer

from config import ES_CONFIG, EMBEDDING_CONFIG, MEMORY_CONFIG


@dataclass
class Memory:
    """记忆条目"""
    id: str
    content: str                    # 记忆摘要
    embedding: List[float] = field(default_factory=list)
    importance: int = 5             # 重要性 0-10
    emotion: str = "平静"           # 关联情绪
    memory_type: str = "event"      # fact / event / emotion
    timestamp: str = ""             # ISO格式时间戳
    user_id: str = "default"

    def to_dict(self) -> dict:
        return asdict(self)


class MemoryStore:
    """基于Elasticsearch的长期记忆存储与检索"""

    def __init__(self):
        self.es = Elasticsearch(ES_CONFIG["hosts"])
        self.index = ES_CONFIG["memory_index"]
        self.embed_model = SentenceTransformer(EMBEDDING_CONFIG["model_path"])
        self._ensure_index()

    def _ensure_index(self):
        """创建ES索引（含向量字段）"""
        if self.es.indices.exists(index=self.index):
            return

        mappings = {
            "mappings": {
                "properties": {
                    "content": {"type": "text", "analyzer": "ik_max_word"},
                    "embedding": {
                        "type": "dense_vector",
                        "dims": EMBEDDING_CONFIG["dimension"],
                        "index": True,
                        "similarity": "cosine",
                    },
                    "importance": {"type": "integer"},
                    "emotion": {"type": "keyword"},
                    "memory_type": {"type": "keyword"},
                    "timestamp": {"type": "date"},
                    "user_id": {"type": "keyword"},
                }
            }
        }
        self.es.indices.create(index=self.index, body=mappings)

    def add_memory(self, content: str, importance: int, emotion: str,
                   memory_type: str = "event", user_id: str = "default") -> str:
        """添加一条记忆"""
        embedding = self.embed_model.encode(content).tolist()
        mem_id = f"mem_{user_id}_{int(time.time() * 1000)}"

        doc = {
            "content": content,
            "embedding": embedding,
            "importance": importance,
            "emotion": emotion,
            "memory_type": memory_type,
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id,
        }
        self.es.index(index=self.index, id=mem_id, body=doc)
        return mem_id

    def retrieve(self, query: str, user_id: str = "default",
                 top_k: int = None) -> List[Dict]:
        """
        混合检索：向量相似度 + 关键词匹配 + 时间衰减 + 重要性加权

        这是选用ES的核心优势：一次查询同时利用语义和关键词能力
        """
        top_k = top_k or MEMORY_CONFIG["top_k"]
        threshold = MEMORY_CONFIG["similarity_threshold"]
        half_life = MEMORY_CONFIG["half_life_days"]

        query_embedding = self.embed_model.encode(query).tolist()

        # ES混合查询：kNN向量检索 + BM25关键词检索
        search_body = {
            "size": top_k * 3,  # 多取一些，后面用时间衰减重排
            "query": {
                "bool": {
                    "must": [
                        {"term": {"user_id": user_id}},
                    ],
                    "should": [
                        # 关键词匹配（BM25）
                        {"match": {"content": {"query": query, "boost": 0.3}}},
                    ],
                }
            },
            "knn": {
                "field": "embedding",
                "query_vector": query_embedding,
                "k": top_k * 3,
                "num_candidates": 100,
                "filter": {"term": {"user_id": user_id}},
            },
        }

        resp = self.es.search(index=self.index, body=search_body)
        hits = resp["hits"]["hits"]

        # 时间衰减 + 重要性加权重排
        now_ts = time.time()
        scored = []
        for hit in hits:
            src = hit["_source"]
            es_score = hit["_score"]  # ES的综合相似度分

            # 时间衰减
            mem_time = datetime.fromisoformat(src["timestamp"]).timestamp()
            days_ago = (now_ts - mem_time) / 86400
            time_weight = 0.5 ** (days_ago / half_life)

            # 重要性权重
            imp_weight = src.get("importance", 5) / 10.0

            # 综合评分
            final_score = es_score * time_weight * (0.5 + 0.5 * imp_weight)

            scored.append({
                "content": src["content"],
                "emotion": src.get("emotion", ""),
                "importance": src.get("importance", 5),
                "timestamp": src.get("timestamp", ""),
                "score": round(final_score, 4),
            })

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def cleanup_old_memories(self, user_id: str = "default",
                             max_count: int = None):
        """清理低权重的旧记忆，保持容量可控"""
        max_count = max_count or MEMORY_CONFIG["max_memories"]

        # 查询该用户的记忆总数
        count = self.es.count(
            index=self.index,
            body={"query": {"term": {"user_id": user_id}}}
        )["count"]

        if count <= max_count:
            return

        # 删除最旧且重要性最低的记忆
        delete_count = count - max_count
        old_mems = self.es.search(
            index=self.index,
            body={
                "size": delete_count,
                "query": {"term": {"user_id": user_id}},
                "sort": [
                    {"importance": "asc"},
                    {"timestamp": "asc"},
                ],
            },
        )

        for hit in old_mems["hits"]["hits"]:
            self.es.delete(index=self.index, id=hit["_id"])
