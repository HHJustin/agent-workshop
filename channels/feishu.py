"""
飞书 Channel — 接收飞书消息 → Agent 处理 → 回复

使用：
    1. 飞书开放平台创建应用 → 启用机器人 → 获取 webhook URL
    2. 配置 .env: FEISHU_WEBHOOK_URL
    3. 配置飞书事件订阅: http://yourserver:9900/api/channels/feishu

API:
    POST /api/channels/feishu   — 接收飞书事件回调
    POST /api/channels/feishu/send — 手动发送飞书消息

Author: 程响
"""

from app.logger import logger


async def send_feishu_message(title: str, content: str) -> dict:
    """发送飞书消息（通过 Bot Webhook）"""
    import json, httpx
    from app.config import config

    webhook = config.feishu_webhook_url
    if not webhook:
        return {"ok": False, "error": "未配置 FEISHU_WEBHOOK_URL"}

    payload = {
        "msg_type": "interactive",
        "card": {
            "header": {
                "title": {"tag": "plain_text", "content": title[:100]},
                "template": "blue",
            },
            "elements": [
                {"tag": "markdown", "content": content[:3000]},
                {"tag": "hr"},
                {"tag": "note", "elements": [
                    {"tag": "plain_text", "content": "Agent Workshop 自动发送"}
                ]},
            ],
        },
    }

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(webhook, json=payload)
        if resp.status_code == 200:
            logger.info(f"[Feishu] 消息已发送: {title[:30]}")
            return {"ok": True}
        else:
            logger.error(f"[Feishu] 发送失败: {resp.status_code} {resp.text[:100]}")
            return {"ok": False, "error": resp.text[:100]}
    except Exception as e:
        logger.error(f"[Feishu] 异常: {e}")
        return {"ok": False, "error": str(e)}


async def send_agent_response(question: str, answer: str) -> dict:
    """把 Agent 的回答推送到飞书"""
    title = f"Agent 回复: {question[:30]}..."
    return await send_feishu_message(title, answer)
