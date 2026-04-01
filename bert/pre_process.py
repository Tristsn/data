"""
智能家居语义理解系统 - 预处理模块
参考 rk_nlu/process/func_pos/pre_process.py 的设计思路

功能：
1. 文本清洗 - 去除特殊符号、多余空格
2. 标准化处理 - 数字转换（中文数字↔阿拉伯数字）、标点统一
3. 上下文提取 - 历史对话信息整合
"""

import re
import unicodedata
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, field


# ==================== 数字转换工具 ====================

# 中文数字映射
CN_NUM_MAP = {
    '零': 0, '一': 1, '二': 2, '三': 3, '四': 4,
    '五': 5, '六': 6, '七': 7, '八': 8, '九': 9,
    '两': 2, '〇': 0, '壹': 1, '贰': 2, '叁': 3,
    '肆': 4, '伍': 5, '陆': 6, '柒': 7, '捌': 8, '玖': 9,
}

CN_UNIT_MAP = {
    '十': 10, '拾': 10, '百': 100, '佰': 100,
    '千': 1000, '仟': 1000, '万': 10000, '亿': 100000000,
}

ARAB_TO_CN = {str(i): cn for cn, i in CN_NUM_MAP.items() if cn in '零一二三四五六七八九'}


def chinese_to_arabic(cn_str: str) -> Optional[int]:
    """中文数字转阿拉伯数字，如 '二十六' -> 26, '二零' -> 20"""
    if not cn_str:
        return None

    # 纯中文数字逐字转换（如 "二零" -> 20, "一零二" -> 102）
    if all(c in CN_NUM_MAP for c in cn_str):
        return int(''.join(str(CN_NUM_MAP[c]) for c in cn_str))

    # 带单位的中文数字（如 "二十六" -> 26）
    result = 0
    current = 0
    for char in cn_str:
        if char in CN_NUM_MAP:
            current = CN_NUM_MAP[char]
        elif char in CN_UNIT_MAP:
            unit = CN_UNIT_MAP[char]
            if current == 0 and char == '十':
                current = 1  # 处理 "十六" -> 16
            result += current * unit
            current = 0
        else:
            return None
    result += current
    return result if result > 0 else None


def arabic_to_chinese(num: int, mode: str = 'digit') -> str:
    """
    阿拉伯数字转中文数字
    mode='digit': 逐位转换，如 20 -> '二零'
    mode='unit':  带单位转换，如 20 -> '二十'
    """
    if mode == 'digit':
        return ''.join(ARAB_TO_CN.get(d, d) for d in str(num))

    if num < 0:
        return '负' + arabic_to_chinese(-num, mode)
    if num == 0:
        return '零'

    units = ['', '十', '百', '千']
    result = ''
    s = str(num)
    length = len(s)
    for i, digit in enumerate(s):
        d = int(digit)
        pos = length - 1 - i
        if d != 0:
            result += ARAB_TO_CN[digit] + (units[pos] if pos < len(units) else '')
        elif not result.endswith('零') and i < length - 1:
            result += '零'

    # "一十" -> "十"
    if result.startswith('一十'):
        result = result[1:]
    return result.rstrip('零') or '零'


# ==================== 文本清洗 ====================

# 需要移除的特殊字符模式
SPECIAL_CHARS_PATTERN = re.compile(r'[【】\[\]{}()（）<>《》「」『』\u200b\ufeff]')
# 多余空白
MULTI_SPACE_PATTERN = re.compile(r'\s+')
# 全角转半角映射
FULLWIDTH_OFFSET = 0xFEE0


def normalize_unicode(text: str) -> str:
    """Unicode标准化，全角转半角"""
    result = []
    for char in text:
        code = ord(char)
        # 全角字母/数字 -> 半角
        if 0xFF01 <= code <= 0xFF5E:
            result.append(chr(code - FULLWIDTH_OFFSET))
        elif code == 0x3000:  # 全角空格
            result.append(' ')
        else:
            result.append(char)
    return ''.join(result)


def remove_special_chars(text: str) -> str:
    """移除特殊符号，保留中文、字母、数字和基本标点"""
    text = SPECIAL_CHARS_PATTERN.sub('', text)
    text = MULTI_SPACE_PATTERN.sub('', text)
    return text.strip()


def normalize_punctuation(text: str) -> str:
    """标点符号统一化"""
    mapping = {
        '，': ',', '。': '.', '！': '!', '？': '?',
        '；': ';', '：': ':', '"': '"', '"': '"',
        ''': "'", ''': "'", '、': ',',
    }
    for cn, en in mapping.items():
        text = text.replace(cn, en)
    return text


# ==================== 数字标准化 ====================

# 匹配文本中的中文数字串（如 "二十六度" 中的 "二十六"）
CN_NUM_PATTERN = re.compile(
    r'[零一二三四五六七八九两〇壹贰叁肆伍陆柒捌玖十拾百佰千仟万亿]+'
)

# 匹配设备名中的阿拉伯数字（如 "s20" 中的 "20"）
DEVICE_NUM_PATTERN = re.compile(r'([a-zA-Z])(\d+)')


def expand_device_number_variants(text: str) -> List[str]:
    """
    为设备名中的数字生成多种变体，用于热词匹配
    如 "s20" -> ["s20", "s二零", "s二十"]
    """
    variants = [text]
    match = DEVICE_NUM_PATTERN.search(text)
    if match:
        prefix = text[:match.start(2)]
        num_str = match.group(2)
        suffix = text[match.end(2):]
        num = int(num_str)

        # 逐位变体: 20 -> 二零
        digit_variant = prefix + arabic_to_chinese(num, 'digit') + suffix
        variants.append(digit_variant)

        # 带单位变体: 20 -> 二十
        unit_variant = prefix + arabic_to_chinese(num, 'unit') + suffix
        if unit_variant != digit_variant:
            variants.append(unit_variant)

    return variants


def normalize_numbers_in_text(text: str) -> str:
    """
    标准化文本中的数字表达
    - 将中文数字转为阿拉伯数字（用于数值参数提取）
    - 保留原始文本用于热词匹配
    """
    def replace_cn_num(match):
        cn_str = match.group()
        # 跳过单独的单位词
        if cn_str in ('十', '百', '千', '万', '亿'):
            return cn_str
        num = chinese_to_arabic(cn_str)
        if num is not None:
            return str(num)
        return cn_str

    return CN_NUM_PATTERN.sub(replace_cn_num, text)


# ==================== 上下文处理 ====================

@dataclass
class History:
    """对话历史"""
    turns: List[Dict] = field(default_factory=list)  # [{"query": ..., "intent": ..., "slots": ...}]
    max_turns: int = 5

    def add_turn(self, query: str, intent: str = '', slots: List[Dict] = None):
        self.turns.append({
            'query': query,
            'intent': intent,
            'slots': slots or [],
        })
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns:]

    def get_last_device(self) -> Optional[str]:
        """从历史中获取最近操作的设备"""
        for turn in reversed(self.turns):
            for slot in turn.get('slots', []):
                if slot.get('name') in ('DeviceType', 'DeviceName'):
                    return slot['value']
        return None

    def get_last_intent(self) -> Optional[str]:
        """获取上一轮意图"""
        if self.turns:
            return self.turns[-1].get('intent')
        return None


@dataclass
class RequestContext:
    """请求上下文，承载用户信息和对话状态"""
    user_id: str = ''
    device_id: str = ''                    # 音箱设备ID
    iot_control_range: int = 0             # 控制范围 0:全部 1:家庭 2:分组
    usr_infos: Dict = field(default_factory=dict)       # 用户设备/家庭/分组信息
    audio_client_info: Dict = field(default_factory=dict)  # 音箱位置和权限
    hotword_slots: List[Dict] = field(default_factory=list)  # 热词匹配结果
    history: Optional[History] = None      # 对话历史


# ==================== 预处理主流程 ====================

@dataclass
class PreProcessResult:
    """预处理结果"""
    raw_text: str           # 原始文本
    cleaned_text: str       # 清洗后文本
    normalized_text: str    # 标准化后文本（数字转换等）
    context: RequestContext  # 请求上下文


def pre_process(
    query: str,
    context: Optional[RequestContext] = None,
) -> PreProcessResult:
    """
    预处理主入口

    流程:
    1. 文本清洗 - 去特殊符号、空白处理
    2. Unicode标准化 - 全角转半角
    3. 标点统一
    4. 数字标准化 - 中文数字转阿拉伯数字
    5. 上下文整合

    Args:
        query: 用户原始输入
        context: 请求上下文（包含用户信息、对话历史等）

    Returns:
        PreProcessResult 预处理结果
    """
    if context is None:
        context = RequestContext()

    # Step 1: 文本清洗
    cleaned = remove_special_chars(query)

    # Step 2: Unicode标准化
    cleaned = normalize_unicode(cleaned)

    # Step 3: 标点统一
    cleaned = normalize_punctuation(cleaned)

    # Step 4: 转小写（英文部分）
    cleaned = cleaned.lower()

    # Step 5: 数字标准化
    normalized = normalize_numbers_in_text(cleaned)

    return PreProcessResult(
        raw_text=query,
        cleaned_text=cleaned,
        normalized_text=normalized,
        context=context,
    )


# ==================== 测试 ====================

if __name__ == '__main__':
    # 基本预处理测试
    test_cases = [
        "打开客厅的小米空调，调到26度",
        "把温度调到二十六度",
        "【打开】灯光",
        "播放Ｓ２０的音乐",       # 全角字符
        "帮我把那个调亮一点",
        "s20空调设置二零度",
    ]

    print("=" * 60)
    print("预处理测试")
    print("=" * 60)

    for text in test_cases:
        result = pre_process(text)
        print(f"\n原始: {result.raw_text}")
        print(f"清洗: {result.cleaned_text}")
        print(f"标准: {result.normalized_text}")

    # 数字变体测试
    print(f"\n{'=' * 60}")
    print("设备名数字变体测试")
    print("=" * 60)
    for name in ["s20", "p10", "a1"]:
        variants = expand_device_number_variants(name)
        print(f"  {name} -> {variants}")

    # 中文数字转换测试
    print(f"\n{'=' * 60}")
    print("中文数字转换测试")
    print("=" * 60)
    cn_tests = ["二十六", "二零", "一百零三", "十六", "三"]
    for cn in cn_tests:
        print(f"  {cn} -> {chinese_to_arabic(cn)}")
