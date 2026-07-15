"""PII 隐私保护单元测试"""
from memory.privacy import detect_pii, mask_pii, has_pii


def test_detect_phone():
    found = detect_pii("我的电话是13812345678")
    assert any(p["type"] == "phone" for p in found)


def test_detect_email():
    found = detect_pii("邮箱 test@example.com 请联系")
    assert any(p["type"] == "email" for p in found)


def test_no_pii():
    assert not has_pii("今天天气真好")


def test_mask_phone():
    masked = mask_pii("电话: 13812345678")
    assert "13812345678" not in masked
    assert "PHONE" in masked


def test_mask_email():
    masked = mask_pii("邮箱: test@qq.com")
    assert "test@qq.com" not in masked
    assert "EMAIL" in masked


def test_mask_multiple():
    masked = mask_pii("电话13812345678邮箱test@qq.com")
    assert "PHONE" in masked
    assert "EMAIL" in masked
