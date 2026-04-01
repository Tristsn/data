"""
智能家居语义理解系统 - 后处理与规则修正模块

三件事：
1. 规则修正 - 修正模型判断不准的 badcase（色温、充电等）
2. 特殊逻辑 - 处理延时/定时控制、数值标准化
3. 协议输出 - 用 Pydantic 校验并格式化为标准设备控制指令
"""

import re
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, field_validator


# ==================== 基础类型定义 ====================

class Skill(str, Enum):
    SMART_HOME = "SmartHome"
    APP_OPERATE = "AppOperate"
    PLAY = "Play"
    CHAT = "Chat"


class DeviceIntent(str, Enum):
    OPEN = "Open"           # 打开
    CLOSE = "Close"         # 关闭
    SET = "Set"             # 设置
    QUERY = "Query"         # 查询
    ADJUST_UP = "AdjustUp"  # 调高
    ADJUST_DOWN = "AdjustDown"  # 调低
    CHARGE = "Charge"       # 充电


class SlotType(str, Enum):
    DEVICE_TYPE = "DeviceType"
    DEVICE_NAME = "DeviceName"
    POINT_NAME = "PointName"
    PARAMETER = "Parameter"
    MODE = "Mode"
    LEVEL = "Level"
    DELAY = "Delay"
    ROTATE_DIR = "RotateDirection"


# ==================== Pydantic 数据模型（协议定义）====================
# Pydantic 的作用：自动校验字段类型、必填项、值范围
# 如果数据不合法，会直接抛出清晰的错误信息

class SlotItem(BaseModel):
    """单个槽位"""
    slot_type: SlotType
    value: str

    def __repr__(self):
        return f"{self.slot_type.value}={self.value}"


class NLURes(BaseModel):
    """
    NLU 最终输出结果（标准协议格式）

    这就是 Pydantic 的核心用法：
    - 声明字段和类型，Pydantic 自动校验
    - skill 必须是 Skill 枚举值，传个乱七八糟的字符串会报错
    - slots 必须是 SlotItem 列表，格式不对也会报错
    """
    skill: Skill
    intent: DeviceIntent
    slots: List[SlotItem] = []
    delay_seconds: Optional[int] = None  # 延时秒数（可选）

    # Pydantic 的 validator：自定义校验逻辑
    @field_validator('delay_seconds')
    @classmethod
    def check_delay(cls, v):
        if v is not None and v < 0:
            raise ValueError('延时秒数不能为负数')
        return v


# ==================== 规则修正引擎 ====================

def fix_color_temperature(intent: DeviceIntent, slots: List[dict]) -> List[dict]:
    """
    规则1：色温修正
    用户说"把灯的温度调高" → 灯没有温度，实际是色温

    如果 设备是灯/灯光 且 参数是温度 → 改成色温
    """
    device_type = None
    for s in slots:
        if s["slot_type"] == "DeviceType":
            device_type = s["value"]

    if device_type in ("灯", "灯光", "台灯", "吸顶灯"):
        for s in slots:
            if s["slot_type"] == "Parameter" and s["value"] == "温度":
                s["value"] = "色温"
    return slots


def fix_charge_intent(intent: DeviceIntent, slots: List[dict]):
    """
    规则2：意图转换 - 电量调高→充电
    用户说"把扫地机电量调高" → 电量不能调高，实际是要充电
    """
    for s in slots:
        if s["slot_type"] == "Parameter" and s["value"] == "电量":
            if intent in (DeviceIntent.ADJUST_UP, DeviceIntent.SET):
                return DeviceIntent.CHARGE
    return intent


def fix_camera_rotation(intent: DeviceIntent, slots: List[dict]) -> List[dict]:
    """
    规则3：摄像头旋转映射
    用户说"摄像头抬头" → 转换为旋转方向=上
    """
    device_type = None
    for s in slots:
        if s["slot_type"] == "DeviceType":
            device_type = s["value"]

    if device_type in ("摄像头", "监控"):
        direction_map = {"抬头": "上", "低头": "下", "左转": "左", "右转": "右"}
        for s in slots:
            if s["value"] in direction_map:
                s["slot_type"] = "RotateDirection"
                s["value"] = direction_map[s["value"]]
                # 同时补充参数槽位
                slots.append({"slot_type": "Parameter", "value": "旋转"})
                break
    return slots


# ==================== 延时/定时处理 ====================

def parse_delay(slots: List[dict]):
    """
    解析延时表达，如"三分钟后"→180秒

    返回 (delay_seconds, 清理后的slots)
    """
    delay_seconds = None
    cleaned_slots = []

    for s in slots:
        if s["slot_type"] == "Level" and _is_time_expression(s["value"]):
            delay_seconds = _time_to_seconds(s["value"])
            # 转换为 Delay 类型槽位
            cleaned_slots.append({
                "slot_type": "Delay",
                "value": s["value"],
            })
        else:
            cleaned_slots.append(s)

    return delay_seconds, cleaned_slots


def _is_time_expression(value: str) -> bool:
    """判断是否是时间表达"""
    time_keywords = ["秒", "分钟", "分", "小时", "后"]
    return any(kw in value for kw in time_keywords)


def _time_to_seconds(value: str) -> int:
    """简单的时间表达转秒数"""
    value = value.replace("后", "")

    # 匹配 "X分钟" / "X秒" / "X小时"
    m = re.search(r'(\d+)\s*小时', value)
    if m:
        return int(m.group(1)) * 3600

    m = re.search(r'(\d+)\s*分钟?', value)
    if m:
        return int(m.group(1)) * 60

    m = re.search(r'(\d+)\s*秒', value)
    if m:
        return int(m.group(1))

    return 0


# ==================== 后处理主流程 ====================

def post_process(intent: DeviceIntent, slots: List[dict],
                 skill: Skill = Skill.SMART_HOME) -> NLURes:
    """
    后处理主入口

    流程：
    1. 规则修正（色温、充电、旋转等 badcase）
    2. 延时/定时解析
    3. Pydantic 协议封装与校验
    """
    # Step 1: 规则修正
    slots = fix_color_temperature(intent, slots)
    intent = fix_charge_intent(intent, slots)
    slots = fix_camera_rotation(intent, slots)

    # Step 2: 延时处理
    delay_seconds, slots = parse_delay(slots)

    # Step 3: Pydantic 封装（自动校验字段类型和合法性）
    slot_items = [SlotItem(slot_type=SlotType(s["slot_type"]), value=s["value"])
                  for s in slots]

    result = NLURes(
        skill=skill,
        intent=intent,
        slots=slot_items,
        delay_seconds=delay_seconds,
    )

    return result


# ==================== 演示 ====================

if __name__ == '__main__':
    print("=" * 60)
    print("后处理与规则修正 演示")
    print("=" * 60)

    # ---- 场景1：色温修正 ----
    print("\n场景1: 用户说\"把灯的温度调高\"")
    print("  模型输出: intent=AdjustUp, slots=[灯/DeviceType, 温度/Parameter]")
    result = post_process(
        intent=DeviceIntent.ADJUST_UP,
        slots=[
            {"slot_type": "DeviceType", "value": "灯"},
            {"slot_type": "Parameter", "value": "温度"},
        ],
    )
    print(f"  修正后: intent={result.intent.value}, slots={result.slots}")
    print(f"  → 温度被修正为色温")

    # ---- 场景2：充电意图转换 ----
    print("\n场景2: 用户说\"把扫地机电量调高\"")
    print("  模型输出: intent=AdjustUp, slots=[扫地机/DeviceType, 电量/Parameter]")
    result = post_process(
        intent=DeviceIntent.ADJUST_UP,
        slots=[
            {"slot_type": "DeviceType", "value": "扫地机"},
            {"slot_type": "Parameter", "value": "电量"},
        ],
    )
    print(f"  修正后: intent={result.intent.value}, slots={result.slots}")
    print(f"  → 意图从 AdjustUp 变成了 Charge")

    # ---- 场景3：延时控制 ----
    print("\n场景3: 用户说\"3分钟后关灯\"")
    print("  模型输出: intent=Close, slots=[灯/DeviceType, 3分钟后/Level]")
    result = post_process(
        intent=DeviceIntent.CLOSE,
        slots=[
            {"slot_type": "DeviceType", "value": "灯"},
            {"slot_type": "Level", "value": "3分钟后"},
        ],
    )
    print(f"  修正后: intent={result.intent.value}, delay={result.delay_seconds}秒")
    print(f"  slots={result.slots}")
    print(f"  → \"3分钟后\"被解析为延时180秒")

    # ---- 场景4：正常情况，无需修正 ----
    print("\n场景4: 用户说\"把空调温度调到26度\"（正常，无需修正）")
    result = post_process(
        intent=DeviceIntent.SET,
        slots=[
            {"slot_type": "DeviceType", "value": "空调"},
            {"slot_type": "PointName", "value": "客厅"},
            {"slot_type": "Parameter", "value": "温度"},
            {"slot_type": "Level", "value": "26"},
        ],
    )
    print(f"  输出: {result.model_dump_json(ensure_ascii=False, indent=2)}")

    # ---- Pydantic 校验演示 ----
    print(f"\n{'=' * 60}")
    print("Pydantic 校验演示")
    print("=" * 60)

    print("\n传入合法数据:")
    valid = NLURes(skill="SmartHome", intent="Open",
                   slots=[SlotItem(slot_type="DeviceType", value="灯")])
    print(f"  ✓ 通过: {valid}")

    print("\n传入非法数据（delay_seconds=-10）:")
    try:
        bad = NLURes(skill="SmartHome", intent="Open",
                     slots=[], delay_seconds=-10)
    except Exception as e:
        print(f"  ✗ Pydantic 报错: {e}")

    print("\n传入非法数据（intent='飞翔'，不在枚举里）:")
    try:
        bad = NLURes(skill="SmartHome", intent="飞翔", slots=[])
    except Exception as e:
        print(f"  ✗ Pydantic 报错: {e}")
