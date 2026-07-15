"""
FastAPI 应用入口 — Agent Workshop 项目（Boss Agent 自动调度）
"""

from __future__ import annotations

import asyncio
import json
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.config import config
from app.llm_factory import get_llm_info, get_chat_model
from app.logger import logger
from app.resilience import rate_limited, circuit_breakers
from app.audit import AuditTrace


# ============================================================
# Lifespan
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 50)
    logger.info(f"[启动] {config.app_name} v{config.app_version}")
    logger.info(f"[启动] LLM: {config.llm_provider}/{config.llm_model}")
    logger.info(f"[启动] Vector: {config.vector_store}")
    logger.info(f"[启动] http://{config.host}:{config.port}")
    logger.info("=" * 50)
    yield
    logger.info(f"[关闭] {config.app_name} 已停止")


app = FastAPI(
    title=config.app_name, version=config.app_version,
    description="Boss Agent 智能调度平台", lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
if os.path.isdir(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


# ============================================================
# 模型
# ============================================================

class ChatRequest(BaseModel):
    question: str = "你好"

class AgentRequest(BaseModel):
    question: str = "你好"
    agent_mode: str = "auto"
    session_id: str = "default"
    web_search: bool = True

class FilePathRequest(BaseModel):
    path: str = ""

class CompactionTestRequest(BaseModel):
    message_count: int = 30
    topic: str = "排查核心交换机 CPU 告警"


# ============================================================
# 活跃流取消事件（主动停止机制）
# ============================================================

_active_streams: dict[str, asyncio.Event] = {}


# ============================================================
# 路由
# ============================================================

@app.get("/")
async def root():
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": f"Welcome to {config.app_name}"}


@app.get("/health")
async def health():
    return JSONResponse(content={
        "status": "healthy", "service": config.app_name,
        "version": config.app_version, "llm": get_llm_info(),
    })


@app.post("/api/chat/test")
async def chat_test(req: ChatRequest):
    """LLM 连通性测试"""
    try:
        llm = get_chat_model(streaming=False)
        response = llm.invoke(req.question)
        return {"status": "ok", "question": req.question, "answer": response.content}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/api/agent")
async def agent_chat(req: AgentRequest):
    """非流式 Agent 对话"""
    from agents.react_agent import react_agent
    answer = await react_agent.ainvoke(req.question, req.session_id)
    react_agent._save_messages(req.session_id, req.question, answer)
    return {"status": "ok", "question": req.question, "answer": answer}


@app.post("/api/agent/stream")
@rate_limited
async def agent_chat_stream(req: AgentRequest, request: Request):
    """流式 Agent 对话 — SSE（支持主动停止，有限流保护）"""
    from agents import intent_router
    from agents.react_agent import react_agent

    # 始终先检测用户意图
    route_result = await intent_router.route(req.question)
    actual_intent = route_result.intent
    logger.info(f"[IntentRouter] {req.question[:30]}... → intent={actual_intent}")

    # Agent 模式选择
    # PlanExecute/Supervisor 只在诊断场景生效，QA/Report 强制用 ReAct
    req.agent_mode = actual_intent

    if req.agent_mode == "supervisor":
        from agents.supervisor import supervisor_agent
        agent = supervisor_agent
    elif req.agent_mode == "plan_execute":
        from agents.plan_execute import plan_execute_agent
        agent = plan_execute_agent
    elif req.agent_mode == "boss":
        from agents.boss_agent import boss_agent
        agent = boss_agent
    elif actual_intent == "diagnosis":
        # diagnosis → PlanExecute（诊断场景直接用深度模式）
        complex_keywords = ["排查", "诊断", "分析", "告警", "故障", "异常", "报错",
                            "CPU", "内存", "磁盘", "网络", "重启", "宕机",
                            "全面排查", "根因分析", "综合分析", "彻底排查"]
        if any(kw in req.question for kw in complex_keywords):
            from agents.plan_execute import plan_execute_agent
            agent = plan_execute_agent
            req.agent_mode = "plan_execute"
            logger.info(f"[AutoRoute] 复杂诊断 → PlanExecute")
        else:
            agent = react_agent
    else:
        agent = react_agent  # qa/report → ReAct

    logger.info(f"[Route] {req.question[:30]}... → agent={type(agent).__name__}, intent={actual_intent}")

    # 审计追踪 + 日志 Trace ID 注入（Loguru contextualize 包裹全链路）
    trace = AuditTrace(session_id=req.session_id, user_query=req.question, intent=req.agent_mode)
    trace_logger = logger.bind(trace_id=trace.trace_id)
    trace_logger.info(f"[请求开始] {req.question[:50]}...")
    intent_span = trace.start_span("intent", "IntentRouter", req.question[:200])
    intent_span.finish(f"intent={req.agent_mode}")

    # 联网开关（仅 ReactAgent 支持）
    if hasattr(agent, "rebuild_with_web_search"):
        agent.rebuild_with_web_search(req.web_search)

    # 注册取消事件
    cancel_event = asyncio.Event()
    _active_streams[req.session_id] = cancel_event

    async def generate():
        full_answer = []
        llm_span = trace.start_span("llm", "agent_execution", req.question[:200])
        try:
            async for chunk in agent.astream(req.question, req.session_id, intent=req.agent_mode):
                if cancel_event.is_set():
                    logger.info("[Stream] 已取消")
                    break
                if chunk.get("content"):
                    full_answer.append(chunk["content"])
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
            if full_answer and hasattr(agent, "_save_messages"):
                agent._save_messages(req.session_id, req.question, "".join(full_answer))
            llm_span.finish("".join(full_answer)[:500] if full_answer else "")
            trace.finish("success")
            trace.save()
            trace_logger.info(f"[请求完成] {trace.total_latency_ms}ms, {len(trace.spans)} spans")
        except Exception as e:
            trace_logger.error(f"[请求异常] {e}")
            yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"
            try:
                llm_span.finish(error=str(e)[:200])
                trace.finish("error")
                trace.save()
            except Exception:
                pass
        finally:
            _active_streams.pop(req.session_id, None)

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/api/clean/test")
async def test_cleaner():
    """数据清洗测试 — 模拟脏数据，展示6步管道效果"""
    from langchain_core.documents import Document
    from documents.cleaner import full_clean

    dirty_docs = [
        Document(page_content="版权所有 © 2024 思创网络", metadata={"_source": "manual.pdf"}),
        Document(page_content="", metadata={"_source": "manual.pdf"}),
        Document(page_content="   \n\n\n   ", metadata={"_source": "manual.pdf"}),
        Document(page_content="OSPF 协议配置步骤：1. 进入配置模式 2. 设置 router-id 3. 宣告网络", metadata={"_source": "manual.pdf"}),
        Document(page_content="OSPF 协议配置步骤：1. 进入配置模式 2. 设置 router-id 3. 宣告网络", metadata={"_source": "manual.pdf"}),
        Document(page_content="公司招聘：诚聘网络工程师，要求CCIE证书", metadata={"_source": "manual.pdf"}),
        Document(page_content="BGP 邻居状态异常排查：检查 TCP 179 端口连通性，查看邻居表 show ip bgp summary", metadata={"_source": "manual.pdf"}),
        Document(page_content="xy", metadata={"_source": "manual.pdf"}),
        Document(page_content="########", metadata={"_source": "manual.pdf"}),
    ]

    result = full_clean(dirty_docs)

    return {
        "status": "ok",
        "total_input": result.total_input,
        "total_output": result.total_output,
        "pass_rate": f"{result.pass_rate:.1%}",
        "stats": {step: {"removed_format": s.removed_format, "removed_duplicate": s.removed_duplicate,
                         "removed_irrelevant": s.removed_irrelevant, "flagged_anomaly": s.flagged_anomaly}
                  for step, s in result.stats.items()},
        "anomalies": [{"issue": a.issue, "severity": a.severity, "preview": a.preview}
                      for a in result.anomalies],
        "kept_content": [doc.page_content[:100] for doc in result.documents],
    }


@app.post("/api/tools/test")
async def test_tools():
    """测试所有工具 — 逐个调用看哪些可用"""
    from agents.tools import (
        get_current_time, query_alerts, search_logs,
        web_search, mysql_query, prometheus_query, send_notification,
    )

    results = {}

    # 1. 时间（一定可用）
    results["get_current_time"] = get_current_time.invoke({})

    # 2. 告警（Mock，一定可用）
    results["query_alerts"] = query_alerts.invoke({"severity": "all"})

    # 3. 日志（Mock，一定可用）
    results["search_logs"] = search_logs.invoke({"service_name": "核心交换机", "minutes": 15})

    # 4. 联网搜索（需要 TAVILY_API_KEY）
    try:
        results["web_search"] = await web_search.ainvoke({"query": "OSPF 协议"})
    except Exception as e:
        results["web_search"] = f"失败: {e}"

    # 5. MySQL（需要数据库连接）
    try:
        results["mysql_query"] = await mysql_query.ainvoke({"sql": "SELECT 1"})
    except Exception as e:
        results["mysql_query"] = f"失败: {e}"

    # 6. Prometheus（需要 Prometheus 服务）
    try:
        results["prometheus_query"] = await prometheus_query.ainvoke({"promql": "up"})
    except Exception as e:
        results["prometheus_query"] = f"失败: {e}"

    # 7. 通知（需要 Webhook URL）
    try:
        results["send_notification"] = await send_notification.ainvoke({
            "title": "工具测试",
            "content": "Agent Workshop 工具测试消息",
            "channel": "feishu",
        })
    except Exception as e:
        results["send_notification"] = f"失败: {e}"

    return {"status": "ok", "results": results}


@app.get("/api/audit/traces")
async def list_traces(limit: int = 20):
    """审计追踪列表 — 最近的 LLM 调用记录"""
    from app.audit import list_recent_traces
    traces = list_recent_traces(limit)
    return {"status": "ok", "traces": traces, "total": len(traces)}


@app.get("/api/audit/trace/{trace_id}")
async def get_trace_detail(trace_id: str):
    """单条 Trace 的完整链路（意图→检索→LLM→工具）"""
    from app.audit import get_trace
    result = get_trace(trace_id)
    if not result:
        return {"status": "error", "message": "Trace 不存在"}
    return {"status": "ok", **result}


@app.get("/api/sessions")
async def list_sessions():
    """获取历史会话列表"""
    from agents.react_agent import react_agent
    sessions = await react_agent.list_sessions()
    return {"status": "ok", "sessions": sessions}


@app.get("/api/session/{session_id}")
async def get_session(session_id: str):
    """获取指定会话的历史消息"""
    from agents.react_agent import react_agent
    history = await react_agent.get_session_history(session_id)
    return {"status": "ok", "session_id": session_id, "messages": history, "count": len(history)}


# ==================== 文档管理接口 ====================

@app.get("/api/documents")
async def list_documents():
    """列出所有已索引文档"""
    from documents.index_manager import incremental_indexer
    docs = []
    for rd in incremental_indexer.list_docs():
        fp = rd.get("file_path", "")
        name = fp.split("\\")[-1] if "\\" in fp else fp.split("/")[-1] if "/" in fp else fp
        if name:
            docs.append({"file_name": name, "chunk_count": rd.get("chunk_count", 0)})
    return {"status": "ok", "documents": docs, "total": len(docs)}


@app.delete("/api/session/{session_id}")
async def delete_session(session_id: str):
    """删除指定会话历史"""
    from agents.react_agent import react_agent
    if hasattr(react_agent, "_sessions"):
        react_agent._sessions.pop(session_id, None)
        react_agent._save_sessions()
    # 同时删除该会话上传的文档索引
    from retrieval.vector_store import vector_store_manager
    vector_store_manager.delete_by_source(f"session:{session_id}")
    return {"status": "ok", "session_id": session_id, "message": "会话已删除"}


@app.delete("/api/documents/{file_name}")
async def delete_document(file_name: str):
    """删除指定文档的索引"""
    from retrieval.vector_store import vector_store_manager
    deleted = vector_store_manager.delete_by_source(file_name)
    return {"status": "ok", "file_name": file_name, "deleted_chunks": deleted}


@app.post("/api/agent/stop")
async def stop_agent(session_id: str = "default"):
    """主动停止流式输出"""
    event = _active_streams.get(session_id)
    if event:
        event.set()
        return {"status": "ok"}
    return {"status": "ok", "message": "无活跃流"}


@app.post("/api/intent")
async def detect_intent(req: ChatRequest):
    """意图识别测试"""
    from agents.intent_router import intent_router
    result = await intent_router.route(req.question)
    return {"question": req.question, "intent": result.intent, "confidence": result.confidence,
            "matched_by": result.matched_by, "reason": result.reason}


@app.post("/api/compaction/test")
async def test_compaction(req: CompactionTestRequest):
    """上下文压缩测试"""
    from langchain_core.messages import AIMessage, SystemMessage
    from langchain_core.messages import HumanMessage as LCHumanMessage
    from memory.compaction import ContextCompactor

    messages = [SystemMessage(content="你是一个网络运维助手。中文回答，不编造数据。")]
    for i in range(req.message_count):
        messages.append(LCHumanMessage(content=f"第{i}轮：{req.topic}，步骤{i}的结果？" + "详细" * 30))
        messages.append(AIMessage(content=f"第{i}轮回复：CPU{50+i}%，需排查步骤{i+1}。" + "..." * 50))

    compactor = ContextCompactor()
    result = await compactor.compact(messages, task_state={
        "goal": req.topic, "files_read": ["switch.yaml"], "findings": ["CPU持续上升"],
        "modified": [], "to_verify": "pytest test_network.py",
        "blockers": "缺少root密码", "next_step": "联系网络组",
    })
    return {"status": "ok", "simulated_rounds": req.message_count,
            "token_before": result.token_before, "token_after": result.token_after,
            "layers_applied": result.layers_applied, "trigger": result.trigger.value,
            "summary_preview": result.summary[:300] if result.summary else "（无需压缩）"}


@app.post("/api/upload/global")
async def upload_global(
    file: UploadFile = File(...),
):
    """上传公共知识库文档 → 所有对话共享，不隔离"""
    return await _do_upload(file, session_id="global", scope="global")


@app.post("/api/upload")
async def upload_file(
    file: UploadFile = File(...),
    session_id: str = Form("default"),
):
    """上传个人文档 → 会话隔离"""
    return await _do_upload(file, session_id=session_id, scope="session")


async def _do_upload(
    file: UploadFile,
    session_id: str,
    scope: str,
):
    import tempfile
    from pathlib import Path
    from fastapi import HTTPException

    suffix = Path(file.filename).suffix if file.filename else ".txt"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        from documents.loader import load_document_with_info
        from documents.splitter import split_documents
        from documents.cleaner import full_clean
        from retrieval.vector_store import vector_store_manager

        # 自动判断 PDF 复杂度 → 逐级降级（PyPDF → pdfplumber → MinerU → OCR）
        docs, info = load_document_with_info(tmp_path, pdf_fallback="auto")
        chunks = split_documents(docs)

        # 知识库分级：global（公共库，所有对话共享）/ session（个人库，仅当前对话）
        for c in chunks:
            c.metadata["_session_id"] = session_id
            c.metadata["_scope"] = scope  # global=公共库 / session=个人库
            c.metadata["_source"] = file.filename  # 统一用原始文件名，方便删除
            c.metadata["_file_name"] = file.filename  # 覆盖 loader 写入的临时文件名

        # 6步清洗管道
        clean_result = full_clean(chunks, skip_l3=True, skip_l6=True)
        chunks = clean_result.documents
        if not chunks:
            raise HTTPException(status_code=400, detail="文档解析后无有效内容")

        # 增量索引：用文件名做 doc_id，用临时文件路径算 hash
        from documents.index_manager import incremental_indexer, _generate_doc_id, _file_hash, _generate_version
        doc_id = _generate_doc_id(file.filename)
        fhash = _file_hash(tmp_path)
        version = _generate_version(fhash)
        diff = incremental_indexer.update(tmp_path, chunks, doc_id=doc_id, force_version=version, display_name=file.filename)

        chunks_to_index = getattr(diff, "_to_index", chunks)
        deleted = vector_store_manager.delete_by_source(file.filename)
        ids = vector_store_manager.add_documents(chunks_to_index) if chunks_to_index else []

        # 重建 BM25 索引（新增文档后必须重建）
        from retrieval.hybrid_search import hybrid_searcher
        from retrieval.vector_store import vector_store_manager as vsm
        try:
            all_docs = vsm.similarity_search("__all__", k=1000)
            hybrid_searcher.index(all_docs)
        except Exception:
            pass  # BM25 重建失败不影响主流程

        return {"status": "ok", "file_name": file.filename, "info": info,
                "chunk_count": len(chunks), "deleted_old": deleted, "stored_ids": ids[:5],
                "incremental": {"old_version": diff.old_version[:8],
                                "new_version": diff.new_version[:8],
                                "unchanged": len(diff.unchanged),
                                "added": len(diff.added),
                                "deleted": len(diff.deleted),
                                "changed": len(diff.changed),
                                "summary": diff.summary},
                "cleaning": {"total_input": clean_result.total_input,
                             "total_output": clean_result.total_output,
                             "pass_rate": f"{clean_result.pass_rate:.1%}",
                             "anomalies": len(clean_result.anomalies)},
                "anomaly_details": [{"issue": a.issue, "severity": a.severity, "preview": a.preview}
                                    for a in clean_result.anomalies[:5]],
                "sample_chunks": [{"index": i, "length": len(c.page_content), "preview": c.page_content[:150]}
                                  for i, c in enumerate(chunks[:3])]}
    except Exception as e:
        logger.error(f"[Upload] 失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


# ============================================================
# 启动
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=config.host, port=config.port, reload=config.debug, log_level="info")
