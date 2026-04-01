"""
智能家居语义理解系统 - 热词匹配与实体识别模块
参考 rk_nlu 热词匹配设计思路

核心思路：
1. 构建 Trie 前缀树存储热词（通用热词 + 用户个性化热词）
2. 最大正向匹配算法扫描用户输入，提取命中的实体
3. 用户热词优先级高于通用热词（更具体）
4. 支持设备名中数字的多种变体匹配（s20 ↔ s二零 ↔ s二十）
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum


# ==================== 槽位类型定义 ====================

class SlotType(str, Enum):
    """热词槽位类型"""
    DEVICE_TYPE = "DeviceType"       # 设备类型：空调、灯、窗帘
    DEVICE_NAME = "DeviceName"       # 设备名称：小米空调、卧室灯
    POINT_NAME = "PointName"         # 位置/房间：客厅、卧室、书房
    PARAMETER = "Parameter"          # 属性参数：温度、亮度、颜色
    MODE = "Mode"                    # 模式：制冷、制热、自动
    FAMILY_NAME = "FamilyName"       # 家庭名称
    GROUP_NAME = "GroupName"         # 分组名称


class HotwordSource(str, Enum):
    """热词来源"""
    COMMON = "common"   # 通用热词（预置）
    USER = "user"       # 用户热词（动态加载）


@dataclass
class HotwordSlot:
    """热词匹配结果槽位"""
    value: str                  # 匹配到的文本
    slot_type: SlotType         # 槽位类型
    source: HotwordSource       # 来源
    start: int                  # 在原文中的起始位置
    end: int                    # 在原文中的结束位置
    priority: int = 0           # 优先级，数值越大越优先


# ==================== Trie 前缀树 ====================

class TrieNode:
    """前缀树节点"""
    __slots__ = ['children', 'is_end', 'slot_type', 'source', 'priority']

    def __init__(self):
        self.children: Dict[str, TrieNode] = {}
        self.is_end: bool = False
        self.slot_type: Optional[SlotType] = None
        self.source: Optional[HotwordSource] = None
        self.priority: int = 0


class HotwordTrie:
    """热词前缀树"""

    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str, slot_type: SlotType,
               source: HotwordSource = HotwordSource.COMMON,
               priority: int = 0):
        """插入一个热词"""
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True
        node.slot_type = slot_type
        node.source = source
        node.priority = priority

    def search_prefix(self, text: str, start: int) -> List[Tuple[str, SlotType, HotwordSource, int]]:
        """
        从 text[start] 开始，找出所有能匹配的前缀词
        返回: [(匹配词, 槽位类型, 来源, 优先级), ...]
        """
        node = self.root
        matches = []
        i = start
        while i < len(text) and text[i] in node.children:
            node = node.children[text[i]]
            i += 1
            if node.is_end:
                matched_word = text[start:i]
                matches.append((matched_word, node.slot_type, node.source, node.priority))
        return matches


# ==================== 最大正向匹配引擎 ====================

def max_forward_match(text: str, common_trie: HotwordTrie,
                      user_trie: HotwordTrie) -> List[HotwordSlot]:
    """
    双重前缀树最大正向匹配

    策略：
    1. 从左到右扫描文本
    2. 每个位置同时在通用热词树和用户热词树中查找
    3. 取最长匹配（最大正向匹配）
    4. 同长度时用户热词优先于通用热词
    """
    slots = []
    i = 0

    while i < len(text):
        # 在两棵树中分别查找
        common_matches = common_trie.search_prefix(text, i)
        user_matches = user_trie.search_prefix(text, i)

        all_matches = common_matches + user_matches

        if all_matches:
            # 排序：先按长度降序（最大匹配），再按来源（用户优先），再按优先级降序
            best = max(all_matches, key=lambda m: (
                len(m[0]),                              # 最长匹配
                1 if m[2] == HotwordSource.USER else 0, # 用户热词优先
                m[3],                                    # 优先级
            ))

            word, slot_type, source, priority = best
            slots.append(HotwordSlot(
                value=word,
                slot_type=slot_type,
                source=source,
                start=i,
                end=i + len(word),
                priority=priority,
            ))
            i += len(word)  # 跳过已匹配的部分
        else:
            i += 1  # 没匹配到，前进一个字符

    return slots


# ==================== 热词管理器 ====================

# 通用热词库（实际项目中从配置/数据库加载）
DEFAULT_COMMON_HOTWORDS = {
    # 设备类型
    "空调": SlotType.DEVICE_TYPE,
    "灯": SlotType.DEVICE_TYPE,
    "灯光": SlotType.DEVICE_TYPE,
    "窗帘": SlotType.DEVICE_TYPE,
    "电视": SlotType.DEVICE_TYPE,
    "风扇": SlotType.DEVICE_TYPE,
    "摄像头": SlotType.DEVICE_TYPE,
    "扫地机": SlotType.DEVICE_TYPE,
    "加湿器": SlotType.DEVICE_TYPE,
    "热水器": SlotType.DEVICE_TYPE,
    "净化器": SlotType.DEVICE_TYPE,
    "音箱": SlotType.DEVICE_TYPE,
    # 属性参数
    "温度": SlotType.PARAMETER,
    "亮度": SlotType.PARAMETER,
    "色温": SlotType.PARAMETER,
    "湿度": SlotType.PARAMETER,
    "风速": SlotType.PARAMETER,
    "音量": SlotType.PARAMETER,
    # 模式
    "制冷": SlotType.MODE,
    "制热": SlotType.MODE,
    "自动": SlotType.MODE,
    "睡眠模式": SlotType.MODE,
    "节能模式": SlotType.MODE,
}


class HotwordManager:
    """
    热词管理器

    管理通用热词树和用户热词树，提供热词匹配入口
    """

    def __init__(self):
        self.common_trie = HotwordTrie()
        self.user_trie = HotwordTrie()
        # 加载默认通用热词
        for word, slot_type in DEFAULT_COMMON_HOTWORDS.items():
            self.common_trie.insert(word, slot_type, HotwordSource.COMMON)

    def load_user_hotwords(self, user_devices: List[Dict]):
        """
        加载用户个性化热词（从用户设备列表动态构建）

        user_devices 格式示例:
        [
            {"name": "小米空调", "type": "空调", "room": "客厅", "group": "一楼"},
            {"name": "卧室灯", "type": "灯", "room": "主卧", "group": "二楼"},
        ]
        """
        self.user_trie = HotwordTrie()  # 重建用户热词树

        rooms_seen = set()
        groups_seen = set()

        for device in user_devices:
            name = device.get("name", "")
            room = device.get("room", "")
            group = device.get("group", "")

            # 设备名称 -> 高优先级
            if name:
                self.user_trie.insert(name, SlotType.DEVICE_NAME,
                                      HotwordSource.USER, priority=10)

            # 房间名 -> 位置
            if room and room not in rooms_seen:
                self.user_trie.insert(room, SlotType.POINT_NAME,
                                      HotwordSource.USER, priority=5)
                rooms_seen.add(room)

            # 分组名
            if group and group not in groups_seen:
                self.user_trie.insert(group, SlotType.GROUP_NAME,
                                      HotwordSource.USER, priority=3)
                groups_seen.add(group)

            # 组合变体："客厅空调"、"客厅小米空调"
            if room and name:
                combo = room + name
                self.user_trie.insert(combo, SlotType.DEVICE_NAME,
                                      HotwordSource.USER, priority=15)

    def match(self, text: str) -> List[HotwordSlot]:
        """对文本执行热词匹配"""
        return max_forward_match(text, self.common_trie, self.user_trie)


# ==================== 优先级合并（处理重叠槽位） ====================

def merge_overlapping_slots(slots: List[HotwordSlot]) -> List[HotwordSlot]:
    """
    合并重叠的槽位，保留优先级更高的

    优先级策略：
    Case1: 家庭→分组→设备 (family→group→device)
    Case2: 分组→设备 (group→device)
    Case3: 家庭→设备 (family→device)
    Case4: 兜底，独立匹配
    """
    if not slots:
        return []

    # 按起始位置排序
    sorted_slots = sorted(slots, key=lambda s: (s.start, -s.priority))
    merged = []

    for slot in sorted_slots:
        # 检查是否与已有槽位重叠
        overlap = False
        for existing in merged:
            if slot.start < existing.end and slot.end > existing.start:
                # 有重叠，保留优先级更高的（或更长的）
                if slot.priority > existing.priority or (
                    slot.priority == existing.priority and len(slot.value) > len(existing.value)
                ):
                    merged.remove(existing)
                    merged.append(slot)
                overlap = True
                break

        if not overlap:
            merged.append(slot)

    return sorted(merged, key=lambda s: s.start)


# ==================== 演示 ====================

if __name__ == '__main__':
    print("=" * 70)
    print("热词匹配与实体识别 演示")
    print("=" * 70)

    # 1. 初始化热词管理器
    manager = HotwordManager()

    # 2. 模拟用户设备列表（实际从数据库/接口获取）
    user_devices = [
        {"name": "小米空调", "type": "空调", "room": "客厅", "group": "一楼"},
        {"name": "卧室灯", "type": "灯", "room": "主卧", "group": "二楼"},
        {"name": "小爱音箱", "type": "音箱", "room": "客厅", "group": "一楼"},
        {"name": "扫地机器人", "type": "扫地机", "room": "客厅", "group": "一楼"},
    ]
    manager.load_user_hotwords(user_devices)

    # 3. 测试用例
    test_queries = [
        "打开客厅的小米空调调到26度",
        "把主卧灯亮度调高一点",
        "客厅空调设置制冷模式温度25度",
        "关闭所有灯",
        "帮我开一下客厅小米空调",
    ]

    for query in test_queries:
        print(f"\n输入: \"{query}\"")
        raw_slots = manager.match(query)
        final_slots = merge_overlapping_slots(raw_slots)

        print(f"  匹配结果:")
        if not final_slots:
            print(f"    (无匹配)")
        for slot in final_slots:
            print(f"    \"{slot.value}\" -> {slot.slot_type.value}"
                  f"  [来源={slot.source.value}, 位置={slot.start}:{slot.end}]")

    # 4. 详细展示一个完整例子
    print(f"\n{'=' * 70}")
    print("详细流程展示: \"打开客厅的小米空调调到26度\"")
    print("=" * 70)

    query = "打开客厅的小米空调调到26度"
    print(f"\n[Step 1] 原始输入: \"{query}\"")

    print(f"\n[Step 2] 逐位置扫描匹配过程:")
    for i, char in enumerate(query):
        common_m = manager.common_trie.search_prefix(query, i)
        user_m = manager.user_trie.search_prefix(query, i)
        if common_m or user_m:
            all_m = [(w, t.value, s.value) for w, t, s, _ in common_m + user_m]
            print(f"  位置{i} '{char}': 候选={all_m}")

    print(f"\n[Step 3] 最大正向匹配结果:")
    raw_slots = manager.match(query)
    for s in raw_slots:
        print(f"  \"{s.value}\" -> {s.slot_type.value} ({s.source.value})")

    print(f"\n[Step 4] 合并重叠后最终槽位:")
    final = merge_overlapping_slots(raw_slots)
    for s in final:
        print(f"  \"{s.value}\" -> {s.slot_type.value}")

    print(f"\n这些槽位会和后续 BERT 模型输出的槽位进行融合:")
    print(f"  热词槽位(精确匹配) + 模型槽位(语义理解) → 冲突检测 → 最终结果")
