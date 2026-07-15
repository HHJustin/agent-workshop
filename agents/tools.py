"""
共享工具集 — 三种 Agent 模式共用同一套工具

会话隔离：
    通过 ToolContext (contextvars) 传递上下文，
    retrieve_knowledge 检索时自动过滤当前会话的文档。

Author: 程响
"""

from datetime import datetime
from zoneinfo import ZoneInfo

from langchain_core.tools import tool

from app.config import config
from app.logger import logger

# 复用统一的 ToolContext（兼容旧版 API）
from .context import set_current_session, get_current_context as _get_ctx


def get_current_session() -> str:
    """获取当前会话ID（从 ToolContext 中提取）"""
    return _get_ctx().session_id


# ==================== 知识检索工具 ====================

@tool
async def retrieve_knowledge(query: str) -> str:
    """从知识库中检索相关文档和资料。

    适用场景：用户询问专业知识、操作步骤、配置方法、概念解释时使用。
    参数：query - 检索关键词或用户问题
    返回：相关文档片段

    调用示例：retrieve_knowledge("OSPF协议配置步骤")
    """
    try:
        from retrieval.hybrid_search import hybrid_searcher
        from retrieval.query_rewrite import llm_rewrite

        # Query Rewrite：口语转专业术语，提升召回率
        rewritten = await llm_rewrite(query)

        session_id = get_current_session()

        # 两级混合检索：公共库 + 个人库
        all_docs = []

        # 1. 公共知识库（用改写后的 query 检索）
        global_results = await hybrid_searcher.search(
            rewritten, top_k=3, filter_meta={"_scope": "global"},
        )
        all_docs.extend(global_results)

        # 2. 个人知识库（_scope=session）
        if session_id:
            session_results = await hybrid_searcher.search(
                query, top_k=2, filter_meta={"_session_id": session_id},
            )
            # 去重
            seen = {doc.page_content[:100] for doc, _ in all_docs}
            for doc, score in session_results:
                if doc.page_content[:100] not in seen:
                    all_docs.append((doc, score))
                    seen.add(doc.page_content[:100])

        # 取 top-3
        all_docs.sort(key=lambda x: x[1], reverse=True)
        docs = [doc for doc, _ in all_docs[:3]]
        if not docs:
            return "[知识库] 未找到相关文档"

        parts = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("_file_name", "未知来源")
            parts.append(f"【资料{i}】来源：{source}\n{doc.page_content[:500]}")
        return "\n\n".join(parts)
    except Exception as e:
        logger.error(f"[retrieve_knowledge] 检索失败: {e}")
        return f"知识检索失败: {e}。请告知用户当前知识库不可用。"


# ==================== 时间工具 ====================

@tool
def get_current_time(timezone: str = "Asia/Shanghai") -> str:
    """获取当前日期和时间。

    适用场景：用户询问当前时间、日期，或需要基于当前时间做判断时使用。
    参数：timezone - 时区，默认为 Asia/Shanghai（北京时间）
    返回：格式化的日期时间字符串

    调用示例：get_current_time("Asia/Shanghai")
    """
    try:
        tz = ZoneInfo(timezone)
        now = datetime.now(tz)
        return now.strftime("%Y-%m-%d %H:%M:%S %Z")
    except Exception as e:
        return f"时间查询失败: {e}"


# ==================== 日志查询工具（Mock） ====================

@tool
def search_logs(service_name: str, minutes: int = 15) -> str:
    """查询指定服务的最近日志。

    适用场景：排查故障时需要查看服务日志、定位异常、分析错误原因时使用。
    参数：
        service_name - 服务名称，如 "核心交换机"、"防火墙"、"api-gateway"
        minutes - 查询最近多少分钟的日志，默认 15 分钟
    返回：日志条目列表

    调用示例：search_logs("核心交换机", 30)
    """
    import random

    levels = ["INFO", "INFO", "INFO", "WARN", "ERROR"]
    templates = [
        "端口 GigabitEthernet0/{port} 状态变更: {status}",
        "CPU 使用率: {cpu}%",
        "内存使用率: {mem}%",
        "BGP 邻居 {neighbor} 状态: {state}",
        "接收到来自 {src} 的 {count} 个数据包",
        "ACL 规则触发: 拒绝 {src} → {dst}",
    ]

    logs = []
    for i in range(min(5, max(1, minutes // 5))):
        level = random.choice(levels)
        tmpl = random.choice(templates)
        msg = tmpl.format(
            port=random.randint(1, 24),
            status=random.choice(["up", "down"]),
            cpu=random.randint(20, 95),
            mem=random.randint(30, 90),
            neighbor=f"10.0.{random.randint(1, 255)}.{random.randint(1, 255)}",
            state=random.choice(["Established", "Idle", "Active"]),
            src=f"192.168.{random.randint(1, 255)}.{random.randint(1, 255)}",
            dst=f"10.0.{random.randint(1, 255)}.{random.randint(1, 255)}",
            count=random.randint(100, 10000),
        )
        logs.append(f"[{level}] {msg}")

    return f"【{service_name} 最近{minutes}分钟日志】\n" + "\n".join(logs)


# ==================== 告警查询工具（Mock） ====================

@tool
def query_alerts(severity: str = "all") -> str:
    """查询当前系统活动告警。

    适用场景：排查故障时需要了解系统当前告警状态，确认是否有相关告警触发。
    参数：severity - 告警级别，可选 critical/warning/info/all，默认 all
    返回：告警列表

    调用示例：query_alerts("critical")
    """
    alerts = [
        {"name": "HighCPUUsage", "severity": "critical", "target": "核心交换机-01", "duration": "15m"},
        {"name": "MemoryPressure", "severity": "warning", "target": "汇聚交换机-03", "duration": "8m"},
        {"name": "BgpSessionDown", "severity": "critical", "target": "出口路由器-R1", "duration": "3m"},
        {"name": "PortFlapping", "severity": "warning", "target": "接入交换机-A12", "duration": "22m"},
    ]

    if severity != "all":
        alerts = [a for a in alerts if a["severity"] == severity]

    if not alerts:
        return f"当前无 {severity} 级别告警"

    lines = [f"当前{severity}级别告警 ({len(alerts)}条):"]
    for a in alerts:
        lines.append(f"  [{a['severity'].upper()}] {a['name']} — {a['target']} (持续{a['duration']})")
    return "\n".join(lines)


# ==================== 联网搜索工具 ====================

@tool
async def web_search(query: str) -> str:
    """联网搜索最新信息。知识库中没有答案时使用此工具获取互联网上的实时信息。

    适用场景：知识库检索不到、需要最新资讯、查实时数据、通用知识查询。
    参数：query - 搜索关键词
    返回：搜索结果的摘要

    调用示例：web_search("BGP 路由泄露最新漏洞")
    """
    try:
        api_key = config.tavily_api_key
        if not api_key:
            return "[联网搜索] 未配置 TAVILY_API_KEY，请设置环境变量后重试"

        from tavily import TavilyClient
        client = TavilyClient(api_key=api_key)
        response = client.search(query, search_depth="basic", max_results=3)

        if not response.get("results"):
            return "[联网搜索] 未找到相关信息"

        parts = []
        for i, r in enumerate(response["results"][:3], 1):
            title = r.get("title", "无标题")
            content = r.get("content", "")[:300]
            url = r.get("url", "")
            parts.append(f"【搜索结果{i}】{title}\n{content}\n来源：{url}")

        return "\n\n".join(parts)
    except ImportError:
        return "[联网搜索] tavily-python 未安装，请执行 pip install tavily-python"
    except Exception as e:
        logger.error(f"[web_search] 失败: {e}")
        return f"联网搜索失败: {e}"


# ==================== MySQL 查询工具 ====================

@tool
async def mysql_query(sql: str) -> str:
    """执行 MySQL 查询并返回结果。当用户提到数据库、表、字段、SQL时使用此工具。

    ⚠️ 仅支持 SELECT 查询，禁止 INSERT/UPDATE/DELETE/DROP 等写操作。

    适用场景：需要查询数据库中的运维数据、用户记录、配置信息时使用。
    参数：sql - 完整的 SELECT 查询语句
    返回：查询结果的 JSON 格式

    调用示例：mysql_query("SELECT * FROM employee LIMIT 5")
    """
    logger.info(f"[mysql_query] 收到SQL: {sql[:100]}")
    try:
        import json

        host = config.mysql_host
        port = config.mysql_port
        user = config.mysql_user
        password = config.mysql_password
        database = config.mysql_database

        sql_upper = sql.strip().upper()
        forbidden = ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "TRUNCATE", "CREATE"]
        for word in forbidden:
            if sql_upper.startswith(word) or f" {word} " in f" {sql_upper} ":
                return f"[MySQL] 禁止执行写操作: {word}。仅允许 SELECT 查询。"

        import pymysql
        conn = pymysql.connect(host=host, port=port, user=user, password=password,
                               database=database, charset="utf8mb4", connect_timeout=5)
        with conn.cursor() as cursor:
            cursor.execute(sql)
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
        conn.close()

        if not rows:
            return "[MySQL] 查询结果为空"

        result = []
        for row in rows[:20]:
            result.append(dict(zip(columns, [str(v) if v is not None else "" for v in row])))

        return f"[MySQL] 返回 {len(result)} 行:\n{json.dumps(result, ensure_ascii=False, indent=2)}"
    except ImportError:
        return "[MySQL] pymysql 未安装，请执行 pip install pymysql"
    except Exception as e:
        logger.error(f"[mysql_query] 失败: {e}")
        return f"MySQL 查询失败: {e}"


# ==================== Prometheus 指标查询 ====================

@tool
async def prometheus_query(promql: str) -> str:
    """查询 Prometheus 监控指标。

    适用场景：需要获取 CPU、内存、网络流量、HTTP 请求数等实时监控数据时使用。
    参数：promql - PromQL 查询语句
    返回：查询结果的 JSON 格式

    常用查询示例：
    - CPU使用率: rate(node_cpu_seconds_total{mode!='idle'}[5m])
    - 内存使用: node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes
    - 网络流量: rate(node_network_receive_bytes_total[5m])
    调用示例：prometheus_query("rate(node_cpu_seconds_total{mode='idle'}[5m])")
    """
    try:
        import json
        import httpx

        base_url = config.prometheus_url
        api_url = f"{base_url.rstrip('/')}/api/v1/query"

        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(api_url, params={"query": promql})
            resp.raise_for_status()
            data = resp.json()

        if data.get("status") != "success":
            return f"[Prometheus] 查询失败: {data}"

        results = data.get("data", {}).get("result", [])
        if not results:
            return "[Prometheus] 查询结果为空"

        simplified = []
        for r in results[:10]:
            metric = r.get("metric", {})
            value = r.get("value", [])
            simplified.append({
                "metric": str(metric),
                "value": value[1] if len(value) > 1 else str(value),
            })

        return f"[Prometheus] 返回 {len(results)} 条（显示前10条）:\n{json.dumps(simplified, ensure_ascii=False, indent=2)}"
    except ImportError:
        return "[Prometheus] httpx 未安装，请执行 pip install httpx"
    except Exception as e:
        logger.error(f"[prometheus_query] 失败: {e}")
        return f"Prometheus 查询失败: {e}"


# ==================== 通知工具 ====================

@tool
async def send_notification(title: str, content: str, channel: str = "feishu") -> str:
    """发送通知到指定渠道（飞书/钉钉）。

    适用场景：诊断完成后自动通知运维人员、告警升级时发送紧急通知、生成报告后推送摘要。
    参数：
        title - 通知标题
        content - 通知内容（支持 Markdown）
        channel - 通知渠道，可选 feishu/dingtalk，默认 feishu
    返回：发送结果

    调用示例：send_notification("CPU告警诊断完成", "核心交换机根因为连接池耗尽，建议扩容", "feishu")
    """
    try:
        import json
        import httpx

        webhook_url = ""
        if channel == "feishu":
            webhook_url = config.feishu_webhook_url
        elif channel == "dingtalk":
            webhook_url = config.dingtalk_webhook_url
        else:
            return f"[通知] 不支持的渠道: {channel}，可选 feishu/dingtalk"

        if not webhook_url:
            return f"[通知] 未配置 {channel.upper()}_WEBHOOK_URL 环境变量，跳过发送。通知内容预览：\n### {title}\n{content[:300]}"

        # 飞书消息格式
        if channel == "feishu":
            payload = {
                "msg_type": "interactive",
                "card": {
                    "header": {"title": {"tag": "plain_text", "content": title}},
                    "elements": [{"tag": "markdown", "content": content[:2000]}],
                },
            }
        else:
            payload = {"msgtype": "markdown", "markdown": {"title": title, "text": f"### {title}\n{content[:2000]}"}}

        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.post(webhook_url, json=payload)
            if resp.status_code == 200:
                return f"[通知] 已通过 {channel} 发送成功"
            return f"[通知] 发送失败: {resp.status_code} {resp.text[:100]}"
    except ImportError:
        return "[通知] httpx 未安装，请执行 pip install httpx"
    except Exception as e:
        logger.error(f"[send_notification] 失败: {e}")
        return f"通知发送失败: {e}"


# ==================== 默认工具集 ====================

from mcp_tools.network_server import MCP_NETWORK_TOOLS

DEFAULT_TOOLS = (
    retrieve_knowledge,
    get_current_time,
    search_logs,
    query_alerts,
    web_search,
    mysql_query,
    prometheus_query,
    send_notification,
) + MCP_NETWORK_TOOLS
