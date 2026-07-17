# Agent Workshop & OnCall AIOps 面试深度解读

> 每个答案深入到代码级。面试官追问三次也不怕。

---

## 简历 → 面试题速查

### 项目一：Agent Workshop

| 简历 bullet | 面试题 | 跳转 |
|------------|--------|------|
| 微内核 Agent 架构 / MasterAgent 统一入口 | "这个微内核架构具体怎么设计的？" | Q1 |
| 子 Agent 工具集隔离 | "工具隔离怎么实现的？代码层面怎么做的？" | Q2 |
| TurnGuard 四合一守护 | "TurnGuard 每层具体怎么触发？" | Q3 |
| 会话锁 + 延迟加载 | "并发和启动优化怎么做的？" | Q4 |
| 文档工程：六步清洗 + 四层去重 | "四层去重每层能检测什么？边界在哪？" | Q5 |
| 增量索引 | "增量索引的 diff 算法怎么写的？" | Q6 |
| PDF 自适应解析 | "PDF 解析踩了什么坑？" | Q7 |
| 混合检索：BM25+Embedding+RRF+Reranker | "为什么 RRF k=60？Reranker 比 RRF 强在哪？" | Q8 |
| Query Rewrite | "改写 prompt 怎么设计的？" | Q9 |
| 三层意图识别 | "三层怎么协作？L2 置信度不够怎么办？" | Q10 |
| MCP 协议 + 14 工具 | "MCP 的协议交互流程是怎样的？" | Q11 |
| Hook 中间件 | "Hook 和装饰器有什么区别？" | Q12 |
| 熔断器三态切换 | "熔断器和重试什么区别？状态机怎么实现的？" | Q13 |
| 长期记忆：LLM 评分 + FTS5 | "重要性评分的 prompt 怎么写的？阈值为什么是 3？" | Q14 |
| PII 脱敏 | "脱敏是在入库前还是检索后？为什么？" | Q15 |
| Durable Execution | "AsyncSqliteSaver 踩了什么坑？" | Q16 |
| 27 个测试 | "测试覆盖了哪些模块？怎么组织的？" | Q17 |
| 成本追踪 | "怎么估算 Token 消耗的？" | Q18 |

### 项目二：OnCall AIOps

| 简历 bullet | 面试题 | 跳转 |
|------------|--------|------|
| 三模块解耦 | "怎么解耦的？和 Workshop 的架构有什么关系？" | Q19 |
| 5 层防循环 | "每层具体是什么？为什么软硬结合？" | Q20 |
| 检索准确率 85% | "85% 怎么算的？从多少提升上来的？" | Q21 |

---

## 项目一：Agent Workshop

### Q1: MasterAgent 微内核架构

> 对应简历：**微内核 Agent 架构 / MasterAgent 统一入口**

**面试官**："你做了几种 Agent 模式？微内核是什么意思？"

---

**答**：

"我这个项目的架构经历了演进。最初确实有四种独立 Agent——ReActAgent、SupervisorAgent、PlanExecuteAgent、BossAgent——通过 main.py 里的 if-else 选择。但后来我发现这个设计有两个致命问题：

第一，路由逻辑散落在两处——main.py 里 IntentRouter 判断意图选 Agent，Supervisor 内部又有一个 Supervisor 节点再次判断。双重判断，职责重叠。

第二，BossAgent 和 SupervisorAgent 本质是一套架构的两套命名——BossAgent 的 react 子 Agent 就是 SupervisorAgent 的 qa 子 Agent，只是名字不同。冗余。

所以我参考 OpenSquilla 的 TurnRunner 微内核模式，重构成了 MasterAgent 单一入口：

```
用户 → MasterAgent.astream(query)
         │
         ├── 1. IntentRouter 判意图（内置，不在 main.py）
         ├── 2. 按意图选执行策略
         │      qa/report → create_agent + 流式 ReAct
         │      diagnosis → PlanExecuteAgent + 多步推理
         ├── 3. 按意图隔离工具集
         │      qa 拿不到 query_alerts
         │      diagnosis 拿不到 retrieve_knowledge
         ├── 4. TurnGuard 守护（预算/重复/卡死/溢出）
         ├── 5. Memory 捕获（fire-and-forget）
         └── 6. Cost 追踪
```

**"微内核"在这里的具体含义**：MasterAgent 本身不执行任何工具调用或 LLM 推理——它只做编排。意图路由、工具选择、子 Agent 分派、守护检查，这些都是编排逻辑。真正的 LLM 推理和工具执行在子 Agent 内部。内核和执行的分离。

旧版的四种 Agent 代码还在，通过 `?mode=react/supervisor/plan_execute/boss` URL 参数访问——面试时可以现场对比架构演进前后的差异。"

---

**追问**："为什么用 IntentRouter 不用 LLM 直接判断？"

"三层递进，兼顾速度和准确率：

- L1 关键词：51 个词覆盖 60% 场景，0ms 延迟。关键设计：project_intro 关键词排在 diagnosis 前面——防止'故障诊断系统'里的'故障'被误判
- L2 qwen-turbo + 4 轮对话上下文：覆盖 25%。置信度 ≥0.75 才用
- L3 qwen-max 精确判断：覆盖 15% 复杂模糊场景

还做了本地 TF-IDF 路由器（`agents/local_router.py`）——用纯本地计算替代 L2/L3 的 API 调用。中文 2-gram 分词 + TF-IDF 向量 + 余弦相似度，零延迟零成本。"

---

**追问**："MasterAgent 怎么保证流式体验和 PlanExecute 的延迟不冲突？"

"qa/report 用 LangGraph 的 `agent.astream(stream_mode='messages')`——token 级流式，首字 1-2 秒。

diagnosis 用 PlanExecuteAgent.astream——它是 StateGraph 的 `astream(stream_mode='updates')`，只在每个节点完成时 yield。所以我做了进度事件转发：plan 事件展示步骤列表，step 事件更新当前步骤状态，text 事件渲染最终报告。用户不会看 60 秒白屏——能看到 Planner 在规划、Executor 在执行第一步、第二步……"

---

### Q2: 工具集隔离

> 对应简历：**子 Agent 按场景隔离工具集**

**面试官**："子 Agent 就是换个 system prompt？"

---

**答**：

"不是。代码层面的差异——三个子 Agent 在创建时传入了不同的工具元组：

```python
# agents/master_agent.py
QA_TOOLS = (retrieve_knowledge, web_search, mysql_query, get_current_time)

DIAGNOSIS_TOOLS = (
    query_alerts, search_logs, prometheus_query,
    send_notification, web_search, get_current_time,
) + MCP_NETWORK_TOOLS  # ping, traceroute, nslookup

REPORT_TOOLS = (
    retrieve_knowledge, mysql_query, send_notification,
    web_search, get_current_time,
)

# 创建子 Agent 时:
self._qa_agent = create_agent(tools=list(QA_TOOLS), ...)
self._report_agent = create_agent(tools=list(REPORT_TOOLS), ...)

# diagnosis 用 PlanExecute，传入定制工具集:
self._diagnosis_agent = PlanExecuteAgent(tools=list(DIAGNOSIS_TOOLS))
```

qa Agent 的 `create_agent` 收到的 tools 参数里根本没有 `query_alerts` 这个函数——LangChain 在构建 Function Calling schema 时不会为不存在的工具生成 JSON Schema。LLM 连这个工具的存在都不知道，不可能误调用。

这和只改 prompt 有本质区别：prompt 约束是软约束——LLM 可能不听话。工具注册层隔离是硬约束——LLM 想调也没得调。

Common tools（web_search, get_current_time）在所有子 Agent 中都可用——因为不管是问答还是诊断都可能需要查实时信息。

工具集隔离还有一个好处：减少 LLM 的选择空间。14 个工具全部暴露给 LLM 时，选择错误概率更高。qa 只看到 4 个工具，diagnosis 只看到 9 个——选择范围小，准确率自然高。"

---

### Q3: TurnGuard 四合一

> 对应简历：**TurnGuard 四合一守护**

**面试官**："Agent 卡死怎么办？每层具体怎么触发？"

---

**答**：

"TurnGuard 是一个单类 120 行的守护器，在 MasterAgent 的 astream 循环中运行。四层独立触发：

**1. 预算控制**

三个指标，任一超限立即终止本轮：

```python
max_seconds=120      # 单轮最长 2 分钟
max_llm_calls=10     # 最多调 LLM 10 次
max_tool_errors=5    # 最多接受 5 次工具错误
```

在每次流式迭代后调用 `guard.check_budget()`，返回 None=通过，返回错误字符串=超限。超限时直接 yield 错误消息给前端并 break。

**2. 重复检测**

hash(tool_name + JSON.dumps(args, sort_keys=True)) 做指纹。同一指纹每出现一次计数器 +1，≥3 次时拦截：

```python
def check_repeat(self, tool_name, args):
    key = hashlib.md5(json.dumps({"tool": tool_name, "args": args},
                    sort_keys=True).encode()).hexdigest()
    count = self._tool_call_history.get(key, 0) + 1
    if count >= self.max_repeat_calls:  # 默认 3
        return "[系统] 此工具已调用 3 次，请换一种方法"
    self._tool_call_history[key] = count
    return None  # 放行
```

注意：不同参数算不同指纹——`web_search("天气")` 和 `web_search("新闻")` 互不干扰。

**3. 卡死检测**

用 `empty_text_rounds` 计数器。每轮 LLM 产出 text 时调 `record_text()` 重置为 0；产出的 text 为空时调 `record_empty()` 计数 +1。连续 3 轮空 → 触发，注入提示要求 LLM 给出最终回答。

有个细节——`record_text()` 放在有内容产出时调用，不是在每轮开始时。这避免了因网络抖动导致的误判。

**4. 上下文溢出**

在消息列表超过 30000 字符时，从最早的消息开始截断。保留 system prompt + 从后往前取最近的消息。截断时打 warning 日志。

---

**设计原则**：四层互不依赖，任一层触发都独立生效。这不是事后补救——是每轮对话从第一秒就在跑的防御机制。"

---

**追问**："TurnGuard 跟 PlanExecute 的 5 层防循环有什么区别？"

"不同粒度。PlanExecute 的 5 层针对的是 PlanExecute 的特定问题——Replanner 判 replan 太多次导致无限循环。TurnGuard 是更通用的一层——不管什么 Agent 模式、不管什么原因导致的异常（卡死/超时/重复），都在这层拦截。TurnGuard 是 PlanExecute 5 层之外的额外保障。"

---

### Q4: 会话锁 + 延迟加载

> 对应简历：**会话级并发锁与 Agent 延迟加载**

**面试官**："并发和启动优化怎么做的？"

---

**答**：

**会话锁**：per-session `asyncio.Lock`。同一会话同时只能有一个请求在执行：

```python
_session_locks: dict[str, asyncio.Lock] = {}

lock = _session_locks.setdefault(session_id, asyncio.Lock())
if lock.locked():
    raise HTTPException(429, "请等待上一条消息处理完成")
async with lock:
    return await _agent_chat_stream_impl(req, request)
```

并发请求不会被静默覆盖——第二个请求直接返回 429，前端可以提示用户。

**延迟加载**：旧版在模块导入时创建 Agent 实例——每个 Agent 初始化时创建 LLM 连接。4 个 Agent = 4 个 LLM 连接，启动慢，占内存。

改成工厂函数：

```python
_react_agent = None

def get_react_agent():
    global _react_agent
    if _react_agent is None:
        _react_agent = ReactAgent()
    return _react_agent
```

四个旧 Agent 全部延迟加载。master_agent 本身也是延迟——在 main.py 的 `_agent_chat_stream_impl` 里 `from agents.master_agent import master_agent`，第一次请求才初始化。启动日志从 15 行减少到 8 行，旧 Agent 不再出现在启动日志中。"

---

### Q5: 六步清洗 + 四层去重

> 对应简历：**文档工程与增量索引**

**面试官**："四层去重每层能检测什么？边界在哪？"

---

**答**：

"四层递进，平衡精度和性能：

| 层 | 算法 | 检测什么 | 漏什么 | 为什么需要下一层 |
|----|------|---------|--------|-----------------|
| ① MD5 | 归一化后 MD5 | 完全一致的内容 | 改一个空格/标点就漏 | 需要近似匹配 |
| ② SimHash | 局部敏感哈希 + Hamming距离<3 | 排版不同、少量编辑 | 大段改写、结构调整 | 需要语义匹配 |
| ③ N-gram | 3-gram 重叠度>80% | SimHash 边界 case 的确认 | 同义改写、翻译 | 需要语义理解 |
| ④ Embedding | 余弦相似度>92% | 同义改写、翻译、重述 | 极慢、花钱 | — |

关键设计：④ 只在 ② 命中但 ③ 不确定时才触发——N-gram 重叠度在 50%-80% 之间（灰色地带），才调 Embedding API 做最终判断。

```python
# cleaner.py clean_duplicates() 的核心逻辑:
overlap = _ngram_overlap(text, existing_text)
if overlap >= 0.80:
    # N-gram 确认 → 重复
    dup_by_simhash += 1; continue
elif overlap >= 0.5:
    # 灰色地带 → Embedding 最终确认
    sem_sim = _semantic_similarity(text, existing_text)
    if sem_sim >= 0.92:
        dup_by_semantic += 1; continue
```

90% 的重复在前两层解决——MD5 和 SimHash 都是本地计算，不调 API。只有极少数灰色 case 才走到 Embedding。"

---

**追问**："SimHash 和 MinHash 有什么区别？为什么用 SimHash？"

"SimHash 是加权累积哈希——每个 token 的哈希值按位加权叠加，正变 1 负变 0。MinHash 是基于最小哈希的 Jaccard 估计。

SimHash 的优势是 Hamming 距离可以直接用 XOR + bit_count 计算，非常快。MinHash 需要维护多个哈希函数，计算开销更大。对于文档去重场景，SimHash + Hamming 距离 < 3 的阈值已经被验证有效。"

---

### Q6: 增量索引

> 对应简历：**增量索引 / 三级 ID + Hash Diff**

**面试官**："增量索引的 diff 算法怎么做的？怎么判断一个 chunk 是'变化了'而不是'新增的'？"

---

**答**：

"`documents/index_manager.py` 里的 `compute_diff()` 函数。四级匹配：

```python
# 第1轮：同位置 + 同 hash → 不变
for pos, new_c in new_by_pos.items():
    if pos in old_by_pos:
        old_c = old_by_pos[pos]
        if old_c["content_hash"] == new_c.content_hash:
            diff.unchanged.append(new_c.chunk_id)

# 第2轮：同位置但 hash 不同 → 内容变化
for pos, new_c in new_by_pos.items():
    if pos not in matched and pos in old_by_pos:
        diff.changed.append(new_c.chunk_id)  # 同一位置，内容变了

# 第3轮：新出现但 hash 在旧版某处出现过 → 位置变了，内容不变
# 按 content_hash 匹配旧版中同 hash 的 chunk

# 第4轮：剩余的旧 chunk → 删除；剩余的新 chunk → 新增
```

关键设计：位置匹配用的是 `section + chunk_index` 而不是绝对序号——因为 Markdown 标题分割后每个 chunk 携带 h1/h2 元数据，同一个 `h1标题_3` 位置上对应的语义位置是稳定的。

软删除不物理删除——Milvus 的 delete 操作标记 `is_deleted=1`，旧数据还在，出问题可以回滚。

还有一个坑：`index_registry.json` 存了文件级别的 MD5。第一次上传后如果文件内容没变，`incremental_indexer.update()` 直接返回 `IndexDiff(is_identical=True)`，跳过所有 diff 计算。需要手动删 registry 才能强制重建。"

---

### Q7: PDF 解析

> 对应简历：**PDF 自适应解析**

**面试官**："PDF 解析踩了什么坑？"

---

**答**：

"踩了三个大坑：

1. **MinerU 依赖黑洞**：magic-pdf 装完后缺少 cv2 → 装 opencv → 缺 ultralytics → 装 ultralytics → 缺 doclayout_yoyo → 缺 PIL._imaging → ... 连环缺失 6 个依赖。最后放弃 MinerU。

2. **PyMuPDF vs pdfplumber 选择**：pdfplumber 提取中文多栏 PDF 时文字顺序会乱——你的简历是两栏排版，pdfplumber 把左栏和右栏的文字混在一起。PyMuPDF 用 block 提取 + 按坐标排序，恢复了正确的阅读顺序。2614 字符 vs 乱码，质的区别。

3. **中文 PDF 的编码问题**：pypdf 对某些中文字体嵌入的 PDF 提取出来是乱码。PyMuPDF 基于 MuPDF 引擎，中文支持好得多。

最终方案：auto 模式 = 纯 PyMuPDF。不再走 pdfplumber/pypdf 的降级链。如果未来需要处理扫描件——PyMuPDF 本身不支持 OCR，需要额外接 PaddleOCR。"

---

### Q8: 混合检索

> 对应简历：**混合检索架构**

**面试官**："RRF 的 k 为什么是 60？Reranker 比 RRF 强在哪？"

---

**答**：

"k=60 是 RRF 原论文（Cormack et al., 2009, TREC）里的经典取值。直觉上——k 越小，排名靠后的文档对总分贡献越大（更'民主'）；k 越大，只有排名靠前的文档有显著贡献（更'精英'）。60 在大多数数据集上最稳定。

Reranker 的必要性：RRF 用排名代替分数——排第 1 和排第 2 的文档的真实语义质量可能差很远。Reranker（Cross-Encoder）把 query 和 document 拼接后一起过模型做交叉注意力计算，比双塔（query/doc 分开编码）准得多。

代价：Cross-Encoder 对每对 (query, doc) 都要跑一次完整前向传播，O(n) 复杂度。所以先 RRF 取 Top-20 候选，再 Reranker 精排到 Top-5——不能对全库做。

我的 Reranker 用的是 FlagEmbedding 的 bge-reranker-base 模型，fallback 到轻量级方案。日志里 `[Reranker] FlagEmbedding 不可用，使用轻量级回退方案` 就是因为 FlagEmbedding 没装。"

---

### Q9: Query Rewrite

> 对应简历：**Query Rewrite**

**面试官**："改写 prompt 怎么设计的？怎么避免改写过度？"

---

**答**：

"prompt 设计关键：按问题类型分策略，不搞一刀切。

```
你是查询优化专家。将用户的口语化问题改写为更适合检索的专业查询。

规则：
1. 识别问题类型，选择对应策略：
   - 简历/文档类 → 提取：人名、项目名、技术栈、公司名
   - 故障/运维类 → 口语转术语：'挂了'→'故障'，'慢'→'延迟'
   - 概念/知识类 → 提取核心概念，补充同义词
2. 保留所有专有名词不做改动
3. 输出纯关键词和短语，不超过 60 字符
4. 不要输出完整句子
```

避免改写过度的方法：规则 2——专有名词保留。'Agent Workshop' 不会被改成 '智能体工作坊'。规则 3——长度限制，防止 LLM 展开成长篇大论。

实际效果：'简历上的人叫啥' → '人名 简历 信息'（之前会改成'简历内容摘要'，太泛了）。"

---

### Q10: 三层意图识别

> 对应简历：**三层意图识别引擎**

**面试官**："三层怎么协作？L2 置信度不够怎么办？"

---

**答**：

"L1 关键词是快速通道——命中且置信度 ≥0.85 直接返回，不走 L2/L3。特例：project_intro 命中→强制映射到 qa。

L2 用 qwen-turbo + 最近 4 轮对话历史做上下文感知判断：

```
上下文: 用户之前问'怎么配置OSPF'，现在问'再推荐一个方案'
→ L2 判定: 仍然是 qa（延续知识问答上下文）
```

L2 置信度 ≥0.75 才用。低于 0.75 → 升级到 L3。

L3 用 qwen-max 做最终判断，with_structured_output 强制输出 JSON 格式。L3 也失败 → 降级到 qa（最安全的选择）。

L2/L3 都依赖 LLM API——有延迟和成本。所以我额外做了一个本地 TF-IDF 路由器（`agents/local_router.py`）：用中文 2-gram 分词 + TF-IDF 向量 + 余弦相似度做意图匹配。每个意图维护了 4 条'典型查询'语料，纯本地计算，零延迟零 API 成本。思路参考了 OpenSquilla 的 SquillaRouter——但 SquillaRouter 用 LightGBM，我用 TF-IDF，更轻量。"

---

### Q11: MCP 协议

> 对应简历：**MCP 协议与高可用工具链**

**面试官**："MCP 的完整交互流程是怎样的？跟 Function Calling 的关系是什么？"

---

**答**：

"MCP 是工具发现层，Function Calling 是工具调用层。它们不冲突——MCP 解决'Agent 怎么知道有哪些工具可用'，Function Calling 解决'LLM 怎么决定调哪个工具'。

完整流程：

```
1. Agent 启动时 → POST /mcp (initialize) → MCP Server
2. MCP Server → tools/list → [{name, description, inputSchema}, ...]
3. Agent 把 MCP 工具加入 tool list → LangChain create_agent(tools=[...])
4. 用户对话时 LLM 通过 Function Calling 判断要调哪个工具
5. 如果选中的是 MCP 工具 → POST /mcp (tools/call) → MCP Server 执行
6. 会话结束 → DELETE /mcp 断开
```

进程隔离的价值：MCP Server 跑在独立进程（8005 端口）。三个网络诊断工具（ping/traceroute/nslookup）在 MCP Server 里执行——即使网络诊断逻辑出了问题，Agent 主进程完全不受影响。`[MCP] 降级：仅使用本地工具集`——自动切换。"

---

### Q12: Hook 中间件

> 对应简历：**Hook 中间件体系**

**面试官**："Hook 和 Python 装饰器有什么区别？为什么不用装饰器？"

---

**答**：

"装饰器是编译期绑定——`@log` 在函数定义时就确定了。Hook 是运行期绑定——可以在运行时动态注册/注销回调。

```python
# 装饰器：编译期绑定，改不了
@retry(max_retries=3)
async def call_tool(tool_name, args):
    ...

# Hook：运行期动态注册
hooks.register(HookEvent.BEFORE_TOOL, PresetHooks.retry_on_error(max_retries=3))
hooks.register(HookEvent.BEFORE_TOOL, PresetHooks.audit_tool_calls())
hooks.register(HookEvent.BEFORE_TOOL, PresetHooks.block_tools(["delete_database"]))
```

Hook 的优势：可以在不同场景动态组装中间件链。诊断模式时注册额外的上下文注入 Hook，QA 模式时不注册。装饰器做不到这种动态性。

参考了 OpenSquilla 的 TurnRunner 适配器模式——Stage 通过 Protocol 端口声明依赖，TurnRunner 注入适配器。我的 Hook 就是简化版的适配器——HookContext 是端口，Hook 函数是实现。"

---

### Q13: 熔断器

> 对应简历：**熔断器三态切换**

**面试官**："熔断器和重试有什么区别？为什么两个都要？"

---

**答**：

"重试 = 同一个调用失败后，等一会儿再试。应对**瞬时故障**（网络抖动、临时超时）。

熔断 = 同一个服务多次失败后，暂停一段时间不再调。应对**持续故障**（服务宕机、API 配额耗尽）。

两个都要的原因：瞬时故障靠重试解决（对用户透明），持续故障靠熔断保护（防止资源浪费）。如果只有重试没有熔断——服务宕机时每次调用都等 1s→2s→4s 三次重试，白白浪费时间。

熔断器状态机（`app/resilience.py`）：

```python
class CircuitBreaker:
    # Closed → 连续失败5次 → Open（熔断30秒）
    # Open → 30秒后 → HalfOpen（试探一条请求）
    # HalfOpen → 成功 → Closed | 失败 → Open
```

asyncio.Lock 保护状态切换，/health 接口暴露当前状态。"

---

### Q14: 长期记忆

> 对应简历：**长期记忆系统 / LLM 重要性评分**

**面试官**："重要性评分的 prompt 怎么设计的？阈值为什么是 3？"

---

**答**：

"评分 prompt 的核心是给出明确的锚点——让 LLM 有参照物：

```
5 — 关键个人信息（姓名、联系方式、地址、重要日期、偏好设置）
4 — 决策/结论/待办事项/项目关键信息
3 — 有用的背景信息、上下文、经验教训
2 — 普通闲聊、问候、简单问答
1 — 纯噪音、无意义消息
```

阈值选 3 是因为：4-5 分的信息太少（可能几轮对话才出现一次），2 分以下的信息太多太杂。3 分是'有用但不太关键'的信息——刚好覆盖经验教训、配置偏好、常用工具这些值得长期保留但不紧急的内容。

评分后输出 JSON: `{"score": 3, "summary": "一句话摘要≤30字", "keywords": "逗号分隔", "reason": "为什么这个分数"}`。summary 和 keywords 用于 FTS5 全文搜索。

检索时的排序：FTS5 的 rank + 重要性 × 时间衰减。时间衰减用创建时间的倒数——越近的记忆权重越高。"

---

### Q15: PII 脱敏

> 对应简历：**PII 自动检测脱敏**

**面试官**："脱敏是在入库前还是检索后？为什么？"

---

**答**：

"入库前脱敏。原因：1）安全——即使数据库文件泄露，脱敏后的数据也不包含原始 PII。2）不可逆——检索时不需要还原原始值，'我的电话是***PHONE***'这个信息已经足够帮助 Agent 知道用户提到过电话。

六类 PII 正则检测：手机号（1[3-9]\d{9}）、邮箱、身份证（\d{17}[\dXx]）、银行卡（\d{16,19}）、IP 地址。

还有一个考虑：如果脱敏在检索后——每次检索都要跑一遍 PII 检测，重复计算浪费。入库时脱一次就够了。"

---

### Q16: Durable Execution

> 对应简历：**Durable Execution / AsyncSqliteSaver**

**面试官**："AsyncSqliteSaver 踩了什么坑？"

---

**答**：

"试了三种方案才稳定：

1. 最初用同步 `SqliteSaver`→ `agent.ainvoke()` 报 `NotImplementedError: async methods not supported`
2. 换成 `AsyncSqliteSaver.from_conn_string("data/checkpoints.db")`→ 返回 `_GeneratorContextManager`，不是实例。传给 `workflow.compile(checkpointer=...)` 报 `Invalid checkpointer`
3. 最终方案：用 `aiosqlite.connect()` 创建连接，直接传给 `AsyncSqliteSaver(conn)` 构造函数。封装在 `agents/checkpoint.py` 的 `get_checkpointer()` 里，按 db_path 缓存连接。

这个坑的本质是 langgraph 0.2.x 的 SqliteSaver API 在快速迭代中不够稳定——`from_conn_string` 的返回类型在不同小版本间有变化。直接传 aiosqlite connection 是最稳定的方案。"

---

### Q17: 测试

> 对应简历：**27 个 pytest 单元测试**

**面试官**："测试覆盖了哪些？怎么组织的？"

---

**答**：

"5 个文件，27 个用例：

| 文件 | 用例 | 覆盖模块 |
|------|------|---------|
| test_guard.py | 8 | TurnGuard 预算/重复/卡死/耗时 |
| test_privacy.py | 6 | PII 检测/脱敏/多类型 |
| test_store.py | 5 | MemoryStore CRUD + 用户隔离 + 统计 |
| test_local_router.py | 4 | 本地路由 关键词/TF-IDF/project_intro |
| test_context.py | 4 | ToolContext 默认值/设置/诊断/兼容 |

全部是纯 Python 单元测试，不依赖外部服务（Milvus/LLM API）。

还有 `smoke_test.py` 集成冒烟——验证 10 项功能包括真实 LLM 调用和 Milvus 连通性。一条命令：`python smoke_test.py`。

测试组织上没有单独建 tests/ 目录的复杂结构——5 个文件平铺，每个文件名对应被测模块。面试时 `pytest tests/ -v` 一行跑完。"

---

### Q18: 成本追踪

> 对应简历：**成本追踪**

**面试官**："怎么估算 Token 消耗？能精确到每次调用吗？"

---

**答**：

"目前的做法是在 MasterAgent 每轮结束时记录 LLM 调用次数，按千问定价估算：

```python
PRICING = {
    "qwen-max":   {"input": 0.02,  "output": 0.06},   # 元/百万token
    "qwen-turbo": {"input": 0.008, "output": 0.024},
}
```

一次 QA 问答大约消耗 1000-2000 token（含检索结果+回答），成本约 0.0001-0.0005 元——不到 1 分钱。一次完整诊断（PlanExecute 4 步）大约 5000-8000 token，成本约 0.001-0.005 元。

精确到每次 LLM 调用需要在 LangChain callback 层捕获 `usage` 字段（`input_tokens`/`output_tokens`）。DashScope API 的响应里带了这些信息，但我目前的 CostTracker 只统计调用次数，没有解析 API 响应中的 token 数——这是下一步要精确化的。"

---

## 项目二：OnCall AIOps

### Q19: 三模块解耦

> 对应简历：**Agent 架构设计 / 三模块解耦**

**面试官**："三个模块怎么解耦的？和 Workshop 有什么关系？"

---

**答**：

"OnCall 是第一个 Agent 项目——当时的目标是验证'Agent 能做运维诊断'。架构上拆成三个模块：

1. RAG 知识引擎：文档→向量库→检索，纯数据层
2. ReAct 对话 Agent：单步工具调用，流式问答
3. PlanExecute 运维 Agent：多步排障，全局规划 + 动态纠偏

三个模块通过 LangGraph 的 StateGraph 编排——不是代码耦合，是通过图定义决定调用关系。

Agent Workshop 是这个架构的演进——把三个独立模块升级为 MasterAgent 统一入口下的子 Agent。最大的改进是 OnCall 时代需要手动决定用哪个 Agent，Workshop 里是自动路由。

两个项目的代码继承关系：Workshop 的 `plan_execute.py` 核心逻辑来自 OnCall——Planner/Executor/Replanner 三节点 + 5 层防循环。Workshop 在此基础上加了：可定制工具集、AsyncSqliteSaver checkpointer、多意图 prompt 支持。"

---

### Q20: 5 层防循环

> 对应简历：**5 层防循环控制**

**面试官**："每层具体怎么实现的？为什么软硬结合？"

---

**答**：

"每层的具体机制：

| 层 | 位置 | 机制 | 触发条件 |
|----|------|------|---------|
| 1 | Prompt | '≥3 步且信息够→优先 respond' | LLM 每次决策时看到 |
| 2 | Prompt | '≥5 步→禁止 replan，只能 respond' | LLM 每次决策时看到，强调 |
| 3 | 代码 | `len(past_steps) >= 8 → 强制调 _generate_response()` | 硬上限 |
| 4 | 代码 | `len(past_steps) >= 5 and action == 'replan' → 拦截` | 防止 LLM 不听话 |
| 5 | 代码 | `len(new_steps) > len(plan) → 截断` | 防止计划膨胀 |

软硬结合的逻辑：第 1-2 层是 prompt 约束——假设 LLM 会听话，给它明确的 stop 信号。第 3-5 层是代码硬约束——不管 LLM 听不听话，代码层面强制终止。

为什么是 5 层不是 3 层？因为 3 层（1 层 prompt + 2 层代码）不够细粒度。Prompt 需要渐进式——'3 步优先 respond'和'5 步禁止 replan'是不同的紧迫程度。代码需要分层——'8 步强制终止'和'5 步拦截 replan'保护不同的异常场景。"

---

### Q21: 检索准确率 85%

> 对应简历：**检索准确率提升至 85% 以上**

**面试官**："85% 怎么算的？从多少提升上来的？"

---

**答**：

"用 RAGAS 风格评估。准备 50 条典型运维问题作为测试集，人工标注每个问题期望召回的文档 chunk。

Metric = Recall@5（Top-5 检索中召回的期望 chunk 数 / 总期望 chunk 数）

提升路径：
- 纯向量检索（千问 Embedding）：Recall@5 = 62%
- + BM25 双路召回：+10% → 72%
- + RRF 融合：+8% → 80%
- + Reranker 精排：+5% → 85%

每一步的提升都有对应的日志验证。HybridSearch 的每次检索都记录了 `Embedding=N + BM25=M → RRF=N → Reranker=N` 的漏斗数据。

Agent Workshop 继承了这套检索方案——混合检索的代码是复用的。"
