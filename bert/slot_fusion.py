"""
智能家居语义理解系统 - 槽位融合模块

热词槽位（精确匹配）与 BERT 模型槽位（语义理解）的融合策略：
1. 位置重叠 → 热词优先（热词边界更准）
2. 热词未覆盖 → 模型结果补充（模型能发现新实体）
3. 两者一致 → 直接保留
"""

from dataclasses import dataclass
from typing import List, Optional
from enum import Enum


# ==================== 数据结构 ====================

class SlotType(str, Enum):
    DEVICE_TYPE = "DeviceType"
    DEVICE_NAME = "DeviceName"
    POINT_NAME = "PointName"
    PARAMETER = "Parameter"
    MODE = "Mode"
    LEVEL = "Level"             # 数值参数（26度、50%）
    GROUP_NAME = "GroupName"
    FAMILY_NAME = "FamilyName"
    ROTATE_DIR = "RotateDirection"


class SlotSource(str, Enum):
    HOTWORD = "hotword"   # 来自热词匹配
    MODEL = "model"       # 来自 BERT 模型
    FUSED = "fused"       # 融合后


@dataclass
class Slot:
    """统一槽位结构"""
    value: str
    slot_type: SlotType
    start: int              # 在原文中的起始字符位置
    end: int                # 结束位置
    source: SlotSource
    confidence: float = 1.0  # 置信度，热词默认1.0，模型为softmax概率


# ==================== BIO 解码（模型输出 → 槽位列表） ====================

def decode_bio_tags(tokens: List[str], tags: List[str],
                    confidences: List[float] = None) -> List[Slot]:
    """
    将 BERT 模型输出的 BIO 标签序列解码为槽位列表

    BIO 标签体系：
    - B-SlotType: 槽位开始
    - I-SlotType: 槽位内部（延续）
    - O: 非槽位

    示例：
    tokens:      [把, 客, 厅, 小, 米, 空, 调, 温, 度, 调, 到, 2, 6, 度]
    tags:        [O,  B-PointName, I-PointName, B-Device, I-Device, I-Device, I-Device, B-Parameter, I-Parameter, O, O, B-Level, I-Level, I-Level]
    解码结果:    [("客厅", PointName), ("小米空调", Device), ("温度", Parameter), ("26度", Level)]
    """
    if confidences is None:
        confidences = [0.9] * len(tags)

    slots = []
    current_value = ""
    current_type = None
    current_start = -1
    current_conf_sum = 0.0
    current_conf_count = 0
    char_pos = 0  # 追踪在原文中的字符位置

    for i, (token, tag) in enumerate(zip(tokens, tags)):
        if tag.startswith("B-"):
            # 先保存上一个槽位
            if current_type is not None:
                slots.append(Slot(
                    value=current_value,
                    slot_type=SlotType(current_type),
                    start=current_start,
                    end=char_pos,
                    source=SlotSource.MODEL,
                    confidence=current_conf_sum / max(current_conf_count, 1),
                ))
            # 开始新槽位
            current_type = tag[2:]
            current_value = token
            current_start = char_pos
            current_conf_sum = confidences[i]
            current_conf_count = 1

        elif tag.startswith("I-") and current_type == tag[2:]:
            # 延续当前槽位
            current_value += token
            current_conf_sum += confidences[i]
            current_conf_count += 1

        else:
            # O 标签或类型不匹配，结束当前槽位
            if current_type is not None:
                slots.append(Slot(
                    value=current_value,
                    slot_type=SlotType(current_type),
                    start=current_start,
                    end=char_pos,
                    source=SlotSource.MODEL,
                    confidence=current_conf_sum / max(current_conf_count, 1),
                ))
                current_type = None
                current_value = ""

        char_pos += len(token)

    # 处理最后一个槽位
    if current_type is not None:
        slots.append(Slot(
            value=current_value,
            slot_type=SlotType(current_type),
            start=current_start,
            end=char_pos,
            source=SlotSource.MODEL,
            confidence=current_conf_sum / max(current_conf_count, 1),
        ))

    return slots


# ==================== 核心：槽位融合 ====================

def is_overlapping(slot_a: Slot, slot_b: Slot) -> bool:
    """判断两个槽位在原文中是否有位置重叠"""
    return slot_a.start < slot_b.end and slot_b.start < slot_a.end


def fuse_slots(hotword_slots: List[Slot],
               model_slots: List[Slot]) -> List[Slot]:
    """
    融合热词槽位和模型槽位

    策略：
    1. 热词槽位全部保留（置信度高，边界准确）
    2. 模型槽位逐个检查：
       - 与热词无重叠 → 补充进来（模型发现了热词没覆盖的实体）
       - 与热词有重叠且类型一致 → 丢弃（热词已覆盖）
       - 与热词有重叠但类型不同 → 丢弃模型的（热词优先）
    3. 按位置排序输出
    """
    fused = []

    # Step 1: 热词槽位全部保留
    for hw_slot in hotword_slots:
        fused.append(Slot(
            value=hw_slot.value,
            slot_type=hw_slot.slot_type,
            start=hw_slot.start,
            end=hw_slot.end,
            source=SlotSource.HOTWORD,
            confidence=1.0,
        ))

    # Step 2: 模型槽位逐个检查是否与热词重叠
    for m_slot in model_slots:
        has_overlap = False
        for hw_slot in hotword_slots:
            if is_overlapping(m_slot, hw_slot):
                has_overlap = True
                break

        if not has_overlap:
            # 无重叠，模型发现了新实体，补充进来
            fused.append(Slot(
                value=m_slot.value,
                slot_type=m_slot.slot_type,
                start=m_slot.start,
                end=m_slot.end,
                source=SlotSource.MODEL,
                confidence=m_slot.confidence,
            ))

    # Step 3: 按位置排序
    fused.sort(key=lambda s: s.start)
    return fused


# ==================== 演示 ====================

def print_slot(slot: Slot, indent: str = "  "):
    src_tag = {"hotword": "热词", "model": "模型", "fused": "融合"}
    print(f"{indent}\"{slot.value}\" → {slot.slot_type.value}"
          f"  [来源={src_tag[slot.source]}, 置信度={slot.confidence:.2f},"
          f" 位置={slot.start}:{slot.end}]")


if __name__ == '__main__':
    query = "把客厅小米空调温度调到26度"
    print("=" * 70)
    print(f"融合演示: \"{query}\"")
    print("=" * 70)

    # ---- 热词匹配结果（来自 hotword_match.py）----
    hotword_slots = [
        Slot("客厅", SlotType.POINT_NAME, start=1, end=3,
             source=SlotSource.HOTWORD, confidence=1.0),
        Slot("小米空调", SlotType.DEVICE_NAME, start=3, end=7,
             source=SlotSource.HOTWORD, confidence=1.0),
        Slot("温度", SlotType.PARAMETER, start=7, end=9,
             source=SlotSource.HOTWORD, confidence=1.0),
    ]

    print(f"\n[1] 热词匹配结果（精确匹配，边界准确）:")
    for s in hotword_slots:
        print_slot(s)

    # ---- BERT 模型输出（BIO 序列标注）----
    # 模拟 BERT 对 "把客厅小米空调温度调到26度" 的逐字标注
    tokens = list("把客厅小米空调温度调到26度")
    tags = [
        "O",              # 把
        "B-PointName",    # 客
        "I-PointName",    # 厅
        "B-DeviceName",   # 小  ← 注意：模型可能标成 DeviceName 或拆开
        "I-DeviceName",   # 米
        "I-DeviceName",   # 空
        "I-DeviceName",   # 调
        "B-Parameter",    # 温
        "I-Parameter",    # 度
        "O",              # 调
        "O",              # 到
        "B-Level",        # 2  ← 热词没匹配到，但模型识别出来了
        "I-Level",        # 6
        "I-Level",        # 度
    ]
    confidences = [
        0.98,  # 把
        0.95,  # 客 B-PointName
        0.93,  # 厅 I-PointName
        0.88,  # 小 B-DeviceName
        0.85,  # 米 I-DeviceName
        0.90,  # 空 I-DeviceName
        0.87,  # 调 I-DeviceName
        0.92,  # 温 B-Parameter
        0.94,  # 度 I-Parameter
        0.97,  # 调 O
        0.96,  # 到 O
        0.91,  # 2  B-Level
        0.89,  # 6  I-Level
        0.93,  # 度 I-Level
    ]

    print(f"\n[2] BERT 模型 BIO 标注:")
    print(f"  字符: {' '.join(tokens)}")
    print(f"  标签: {' '.join(tags)}")

    model_slots = decode_bio_tags(tokens, tags, confidences)
    print(f"\n[3] BIO 解码后的模型槽位:")
    for s in model_slots:
        print_slot(s)

    # ---- 融合 ----
    print(f"\n[4] 融合过程:")
    for m_slot in model_slots:
        overlaps = [h for h in hotword_slots if is_overlapping(m_slot, h)]
        if overlaps:
            print(f"  模型 \"{m_slot.value}\"({m_slot.slot_type.value})"
                  f" ↔ 热词 \"{overlaps[0].value}\"({overlaps[0].slot_type.value})"
                  f" → 位置重叠，保留热词")
        else:
            print(f"  模型 \"{m_slot.value}\"({m_slot.slot_type.value})"
                  f" → 无重叠，补充到结果中")

    fused = fuse_slots(hotword_slots, model_slots)
    print(f"\n[5] 最终融合结果:")
    for s in fused:
        print_slot(s)

    # ---- 第二个例子：模型发现热词没有的实体 ----
    print(f"\n{'=' * 70}")
    query2 = "三分钟后关闭卧室灯"
    print(f"融合演示2: \"{query2}\"")
    print("=" * 70)

    hotword_slots2 = [
        Slot("卧室灯", SlotType.DEVICE_NAME, start=5, end=8,
             source=SlotSource.HOTWORD, confidence=1.0),
    ]
    print(f"\n[1] 热词匹配: 只认出了 \"卧室灯\"")
    for s in hotword_slots2:
        print_slot(s)

    # 模型能理解 "三分钟后" 是延时参数
    tokens2 = list("三分钟后关闭卧室灯")
    tags2 = [
        "B-Level", "I-Level", "I-Level", "I-Level",  # 三分钟后
        "O", "O",                                      # 关闭
        "B-DeviceName", "I-DeviceName", "I-DeviceName", # 卧室灯
    ]
    model_slots2 = decode_bio_tags(tokens2, tags2)

    print(f"\n[2] 模型槽位: 识别出了 \"三分钟后\" 和 \"卧室灯\"")
    for s in model_slots2:
        print_slot(s)

    fused2 = fuse_slots(hotword_slots2, model_slots2)
    print(f"\n[3] 融合结果: 热词的 \"卧室灯\" + 模型补充的 \"三分钟后\"")
    for s in fused2:
        print_slot(s)
