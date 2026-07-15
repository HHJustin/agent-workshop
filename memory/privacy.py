"""
记忆隐私保护 — PII 检测 + 脱敏 + 用户隔离 + Right to be Forgotten

Author: 程响
"""

import re

# PII 检测模式
PII_PATTERNS = {
    "phone": re.compile(r'(?:\+?86)?1[3-9]\d{9}'),
    "email": re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'),
    "id_card": re.compile(r'\d{17}[\dXx]'),
    "bank_card": re.compile(r'\d{16,19}'),
    "ip": re.compile(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}'),
}

PII_MASK = {
    "phone": "***PHONE***",
    "email": "***EMAIL***",
    "id_card": "***ID_CARD***",
    "bank_card": "***BANK_CARD***",
    "ip": "***IP***",
}


def detect_pii(text: str) -> list[dict]:
    """检测文本中的 PII"""
    found = []
    for pii_type, pattern in PII_PATTERNS.items():
        for match in pattern.finditer(text):
            found.append({
                "type": pii_type,
                "value": match.group(),
                "start": match.start(),
                "end": match.end(),
            })
    return found


def mask_pii(text: str) -> str:
    """脱敏：替换 PII 为占位符"""
    result = text
    for pii_type, pattern in PII_PATTERNS.items():
        result = pattern.sub(PII_MASK[pii_type], result)
    return result


def has_pii(text: str) -> bool:
    """是否包含 PII"""
    return any(pattern.search(text) for pattern in PII_PATTERNS.values())
