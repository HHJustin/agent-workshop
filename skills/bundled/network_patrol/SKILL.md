---
name: 网络巡检
description: 自动巡检网络设备健康状态，检查告警、CPU、内存，生成报告并推飞书
trigger_keywords: [巡检, 检查网络, 跑一遍, 健康检查, 网络体检, patrol]
tools: [query_alerts, prometheus_query, search_logs, web_search, send_notification, get_current_time]
---

## 执行流程（按顺序，不可跳过）

### 第1步：获取当前时间
调用 get_current_time，记录巡检开始时间。

### 第2步：查询活动告警
调用 query_alerts(severity="all")，获取当前所有活跃告警。
- 按严重级别分类统计（critical: N条, warning: M条）
- 重点关注 critical 级别

### 第3步：检查核心设备 CPU
调用 prometheus_query(rate(node_cpu_seconds_total{mode!='idle'}[5m])) 获取 CPU 使用率。
- 正常: < 60%
- 警告: 60%-85%
- 异常: > 85%

### 第4步：检查核心设备内存
调用 prometheus_query(node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes) 获取内存可用率。
- 正常: > 30%
- 警告: 15%-30%
- 异常: < 15%

### 第5步：抽样查看最近日志
调用 search_logs("核心交换机", 15)，检查最近15分钟是否有异常日志。
关注关键词：error, critical, failed, down, 故障, 异常

### 第6步：生成巡检报告
汇总以上所有结果，按以下格式输出 Markdown 报告。

### 第7步：推送通知
调用 send_notification 将报告推送到飞书。
title: "网络巡检报告 — {日期}"
channel: feishu

## 输出格式

```
## 🏥 网络巡检报告
**巡检时间**: {当前时间}

### 📊 告警概览
| 级别 | 数量 |
|------|------|
| Critical | X |
| Warning | X |

（如有 critical 告警，逐条列出）

### 💻 CPU 使用率
- 当前值: XX%
- 状态: 正常/警告/异常

### 🧠 内存可用率
- 当前值: XX%
- 状态: 正常/警告/异常

### 📋 日志摘要
- 过去15分钟日志量: X 条
- 异常日志: 有/无

### ✅ 巡检结论
（正常/发现 N 个问题，建议如下）
```

## 约束
1. 不执行任何修改操作（不重启服务、不修改配置、不清理日志）
2. 所有数值必须来自工具实际返回，不编造
3. 工具调用失败时标注"查询失败"，继续下一步
4. 飞书发送失败时告知用户"报告已生成但推送失败"
