# Agent Workshop — 待扩展功能

> 当前已完成 P0+P1，以下为后续规划

---

## 文档处理

| 功能 | 说明 | 面试考点 |
|------|------|---------|
| CSV/TSV 加载 | `CSVLoader`，每行一个 Document | Q: "多格式文档怎么统一处理？" |
| XLSX 加载 | `UnstructuredExcelLoader`，保留行列结构 | — |
| 图片 Caption | qwen-vl 生成描述 → 向量化 | Q: "图片怎么检索？" |
| OCR 扫描件 | MinerU OCR 模式 + 中文校正 | Q: "扫描件怎么处理？" |
| 代码文件 | AST 按函数/类切分，不按字符数 | Q: "代码怎么分块？" |

## Agent

| 功能 | 说明 | 面试考点 |
|------|------|---------|
| 联网搜索工具 | Tavily / SerpAPI | Q: "多工具怎么管理？" |
| MySQL 查询工具 | NL2SQL 自动生成查询 | Q: "怎么让 Agent 操作数据库？" |
| 多用户认证 | JWT + 权限隔离 | — |
| Human-in-the-loop | LangGraph `interrupt()` | Q: "高风险操作怎么兜底？" |
| A/B 测试框架 | 动态提示词切换对比 | — |

## 检索

| 功能 | 说明 | 面试考点 |
|------|------|---------|
| 混合检索 | BM25 + Embedding + RRF 融合 | Q: "向量检索和关键词检索怎么融合？" |
| Reranker | Cross-Encoder 精排 | Q: "Bi-encoder vs Cross-encoder？" |
| RAGAS 评估 | 自动化 RAG 质量评估 | Q: "85% 怎么测的？" |
| 多模态 Embedding | CLIP 图片向量化 | — |

## 工程化

| 功能 | 说明 | 面试考点 |
|------|------|---------|
| Redis 缓存 | Embedding 结果缓存 | Q: "怎么加速检索？" |
| 异步任务队列 | Celery 处理大文件 | — |
| K8s 部署 | Helm Chart | Q: "生产环境怎么部署？" |
| 监控告警 | Prometheus + Grafana | — |
| 离线 LLM | vLLM 本地推理 | Q: "如何支持离线部署？" |
