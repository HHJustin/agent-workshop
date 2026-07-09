# Agent Workshop 项目面试准备

> 30 分钟逐字稿 + 常见问题速查

---

## 一、30 秒项目介绍

> "Agent Workshop 是一个一站式 AI 知识库与智能诊断平台。核心能力是文档管理+多模式Agent+混合检索+工具生态。技术栈是 LangChain + LangGraph + Milvus + FastAPI + MCP，14 个工具包括知识检索、联网搜索、MySQL NL2SQL、Prometheus 监控、飞书通知和 MCP 网络诊断。"

---

## 二、3 分钟详细版（面试标准回答）

> "这个项目解决的是企业知识库的工程化落地问题。我做了四个模块：
>
> **第一，文档处理管道。** 支持 PDF/TXT/Markdown，自动判复杂度选 PyPDFLoader 或 MinerU 做解析，然后三阶段语义分块、六步清洗、四层去重，最后增量索引到 Milvus。改了文档只重建变化的部分，不动已有的。
>
> **第二，多模式 Agent。** 默认是 ReAct Agent，14 个工具自主调用。复杂诊断自动走 Plan-Execute-Replan——Planner 先查历史案例制定排查计划，Executor 逐步执行，Replanner 每步评估要不要调整。还实现了 Supervisor 和 Boss Agent 作为对比模式，面试时可以通过 URL 参数现场切换。
>
> **第三，混合检索。** 不是单路 Embedding，是 BM25 + Embedding 双路召回，RRF 融合，最后 Reranker 精排。同时支持会话隔离——A 对话上传的文档 B 对话搜不到。
>
> **第四，工程化。** 熔断限流降级、Hook 中间件体系、SQLite 审计追踪带 trace_id、RAGAS 评估框架、MCP 协议集成。14 个工具里 8 个是真实数据源——Tavily 联网搜索、MySQL NL2SQL、Prometheus 监控、飞书通知、MCP 网络诊断。"

---

## 三、项目亮点速查表

| 亮点 | 一句话 | 面试怎么讲 |
|------|--------|-----------|
| 文档管道的工程化 | 解析→分块→清洗→去重→增量索引 | "不是用 LangChain 默认参数，是做了 6 步清洗 + 4 层去重 + 增量 diff" |
| 四种 Agent 模式 | ReAct/Supervisor/PlanExecute/Boss | "简单问题 ReAct 秒答，复杂问题 PlanExecute 深度排查" |
| 混合检索 | BM25+Embedding+RRF+Reranker | "BM25 捕获精确关键词，Embedding 捕获语义，RRF 统一量纲，Reranker 精排" |
| 会话隔离 | 两级知识库（global/session） | "A 对话上传的文档 B 对话搜不到，生产级的权限隔离" |
| 14 个工具 | 8 真实 + 3 Mock + 3 MCP | "不是调 API 玩，是真实对接了 Tavily、MySQL、Prometheus、飞书" |
| MCP 协议 | 独立进程 + 动态发现 + 降级 | "MCP Server 跑在 8005 端口，Agent 通过 get_tools() 自动发现" |
| 熔断限流 | 指数退避 + 熔断器 + 降级 | "同一工具调 3 次失败熔断 30 秒，防止级联故障" |
| Query Rewrite | 口语→专业术语改写 | "'CPU飙了'→'CPU使用率过高 根因分析'，提升召回率" |

---

## 四、高频面试问题速答

### Q1: "为什么用 Milvus 不用 Chroma？"

> Chroma 是嵌入式，适合原型。Milvus 是独立服务——支持分布式、多索引（IVF_FLAT/HNSW/DiskANN）、健康检查、Prometheus 监控。生产环境 7×24 运行需要独立部署和水平扩展能力。

### Q2: "14 个工具 Agent 怎么选？"

> Agent 通过 Function Calling 的 JSON Schema 判断。我做了三层保障：1）系统提示词明确说什么时候用什么工具；2）每个工具的 description 写清楚触发条件和参数意思；3）Hook 体系做审计追踪，万一选错了日志里有记录。

### Q3: "Plan-Execute-Replan 和 ReAct 什么区别？"

> ReAct 是边走边看——思考→行动→观察→再思考，适合 1-2 步能完成的任务。P-E-R 是先规划后执行——Planner 查历史案例制定全局计划，Executor 逐步执行，Replanner 每步评估要不要调整或停止。我用复杂度检测自动选择：短问题走 ReAct，带"全面排查"等关键词走 P-E-R。

### Q4: "混合检索怎么融合的？"

> BM25 和 Embedding 的分数不在一个量纲上，不能直接加权。我用 RRF（Reciprocal Rank Fusion）——不管原始分数多少，只看排名：排第一得 1/61 分，排第二得 1/62 分，排第 N 得 1/(60+N) 分。两路的 RRF 分数加起来排序。这个算法在 TREC 评测中验证过，简单且有效。

### Q5: "增量索引怎么做的？"

> doc_id + chunk_id + version 三级管理。上传文档时算文件 MD5——没变就跳过。变了就重新分块，算每个 chunk 的 hash，和旧版 diff 对比：相同不动、新增写入、删除软标记、变化重 Embedding。旧版本不物理删除——出问题可以回滚。

### Q6: "系统最大的挑战是什么？"

> langchain_milvus 和 pymilvus 新版本不兼容导致检索返回 0 条。排查了两天才发现是 langchain_milvus 的 Collection.search 和 pymilvus ORM 连接别名冲突。最后绕过 langchain_milvus，用 pymilvus 原生 API 直接搜引擎。这个过程中学到了两个框架版本兼容的调试方法论。

### Q7: "你觉得还差什么？"

> 三个：1）0 个单元测试，这是最应该补的；2）多用户认证和 RBAC 权限；3）K8s 部署和 CI/CD。功能层面已经对齐生产需求，工程化层面还需要测试和部署。

---

## 五、工程落地 — 面试官追问"上线了吗"

### Q8: "这个系统上线了吗？如果没上线，差什么？"

> 功能层面已经对齐生产需求。差的是工程化——测试、部署、监控。
> 
> 已做的：熔断限流降级、Hook 审计、SQLite Trace、日志 trace_id 注入。
> 待做的：pytest 单元测试、Dockerfile、K8s、CI/CD、认证鉴权。

### Q9: "如果日活 1000 人，系统哪里会先崩？怎么解决？"

> 瓶颈在 LLM API 调用——DashScope 有 QPS 限制。我已经加了全局限流器（Token Bucket，20 QPS + 突发 50），超限直接返回 429。其次是 Milvus 单机检索——万级文档没问题，百万级需要集群模式（读写分离 + Proxy）。最后是会话持久化——当前用 JSON 文件，改 PostgreSQL + Redis 做分布式会话。

### Q10: "日志怎么查问题？出错了怎么快速定位？"

> 每次请求日志都带了 trace_id——grep 同一个 trace_id 能看到完整链路：IntentRouter→LLM→工具调用。同时审计追踪存在 SQLite + 日志文件里，保留 90 天。出错了打开 `logs/app_2026-07-09.log`，搜 trace_id，完整调用链一目了然。

### Q11: "你怎么保证 Agent 不给用户错误建议导致生产事故？"

> 四层防护：1）系统提示词约束"严格基于工具查询结果回答，不要推测"；2）知识库检索不到直接说'未找到'，不编造；3）防幻觉提示词明确标注数据不足时写'待确认'；4）PlanExecute 模式下 Replanner 每步评估——连续 3 步没进展就 respond，不继续排查。最重要的——Agent 只给建议不自动执行，Human-in-the-loop 是最终兜底。

### Q12: "换了 Embedding 模型怎么办？历史向量全废了？"

> 不会。Milvus 初始化时自动检查 Collection 的向量维度——如果和配置不一致（比如从 1536 维换到 1024 维），自动重建 Collection。同时增量索引的软删除机制保护旧数据——标记 archived 不物理删除，出问题可以回滚。

### Q13: "配置管理怎么做的？生产环境和开发环境怎么区分？"

> Pydantic Settings + `.env` 文件。所有配置类型安全——端口是 int、debug 是 bool、CORS origins 是 list。`.env` 文件不提交 Git，`.env.example` 是模板。ChatQwen/ChatOpenAI/DeepSeek/vLLM 四种模型后端通过 `LLM_PROVIDER` 一行切换。

### Q14: "怎么评估系统好不好用？有没有量化指标？"

> RAG 阶段用 RAGAS 风格评估——Recall、Precision、Faithfulness、Correctness 四个指标。Agent 阶段用审计 Trace 追踪工具调用链和 Token 消耗。任务完成率目前人工抽检——50 条典型问题，人工打分 1-5 分。下一步要接 Langfuse 做自动化 Agent Eval。

---

## 六、架构图（面试时在白板上画）

```
用户 → FastAPI + SSE
         │
    IntentRouter（三层：关键词→轻量LLM→大模型）
         │
    ┌────┴────┐
    │ ReAct   │ ← 默认，14 工具
    │ Boss    │ ← Supervisor + PlanExecute
    │ PlanExe │ ← Planner→Executor→Replanner
    └────┬────┘
         │
    ┌────┴────────────────────┐
    │ 文档管道                 │ 检索 + 工具
    │ PDF→分块→清洗→增量索引   │ BM25+Embedding+RRF
    │ Milvus                    │ MySQL/Prometheus/飞书
    └──────────────────────────┘
```

---

## 六、面试时的现场演示

| 演示什么 | 怎么做 |
|---------|--------|
| 默认模式 | `http://localhost:9900` 问一个问题 |
| Supervisor 模式 | `http://localhost:9900/?mode=supervisor` |
| PlanExecute 模式 | `http://localhost:9900/?mode=plan_execute` |
| 审计追踪 | `http://localhost:9900/api/audit/traces` |
| 评估报告 | `python -m retrieval.evaluation` |
| Milvus 直查 | `python check_milvus.py` |
| MCP Server | 另一个终端 `python -m mcp_tools.mcp_server` → Agent 日志 `[MCP] 成功加载 3 个工具` |
| 熔断测试 | `python -m app.resilience` |
