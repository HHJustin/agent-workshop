"""
下载真实运维数据 → 清洗 → 导入 Milvus 知识库

数据源:
    1. synthetic_syslogs — 10000条网络设备syslog (CSV)
    2. Cisco 配置示例 — 常用网络设备配置命令
    3. 网络排障知识 — OSPF/BGP/VLAN 常见问题

使用: python scripts/download_test_data.py
"""

import os, sys, json, time
import urllib.request
import csv
from pathlib import Path
from io import StringIO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ============================================================
# 1. 下载 + 解析
# ============================================================

def download_syslog_dataset() -> list[dict]:
    """下载 synthetic syslog 数据集"""
    url = "https://docs.fabrix.ai/data/datasets/synthetic_syslogs_dataset.csv"
    print(f"[1/4] Downloading syslog dataset...")
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            data = resp.read().decode("utf-8")
        reader = csv.DictReader(StringIO(data))
        rows = [row for row in reader]
        print(f"       Downloaded {len(rows)} syslog entries")
        return rows[:2000]  # 取前2000条
    except Exception as e:
        print(f"       Download failed: {e}, using built-in fallback")
        return []


def download_alarm_dataset() -> list[dict]:
    """生成逼真的网络告警数据（基于真实告警模式）"""
    print(f"[2/4] Generating realistic alarm data...")

    # 基于真实网络告警模式构造
    alarm_templates = [
        # (告警名, 级别, 设备类型, 描述模板, 根因)
        ("HighCPUUsage", "critical", "核心交换机",
         "CPU 使用率持续超过 {cpu}%，已持续 {duration} 分钟",
         "可能原因: 环路导致广播风暴 / SNMP 轮询间隔过短 / ACL 规则过多"),
        ("MemoryExhaustion", "critical", "汇聚交换机",
         "内存使用率达到 {mem}%，缓冲区已满",
         "可能原因: 路由表过大 / ARP 表溢出 / 内存泄漏"),
        ("BgpSessionDown", "critical", "出口路由器",
         "BGP 邻居 {neighbor} 状态变为 Down，已断开 {duration} 分钟",
         "可能原因: 底层 IGP 路由不可达 / TCP 179 端口被 ACL 拦截 / 对端设备重启"),
        ("PortFlapping", "warning", "接入交换机",
         "端口 GigabitEthernet0/{port} 在 {duration} 分钟内 up/down {count} 次",
         "可能原因: 网线松动 / 光模块老化 / 双工模式不匹配 / STP 震荡"),
        ("HighLatency", "warning", "核心路由器",
         "到 {target} 的网络延迟从 {normal}ms 增加到 {current}ms",
         "可能原因: 链路拥塞 / 路由黑洞 / QoS 配置不当"),
        ("PacketLoss", "critical", "出口路由器",
         "到 {target} 的丢包率达到 {loss}%，正常应 < 1%",
         "可能原因: 链路质量下降 / MTU 不匹配 / 带宽耗尽"),
        ("DiskSpaceLow", "warning", "网管服务器",
         "磁盘分区 {partition} 使用率达到 {usage}%",
         "可能原因: 日志未轮转 / 数据库表空间未清理"),
        ("TemperatureHigh", "critical", "数据中心交换机",
         "设备温度达到 {temp}°C，超过阈值 70°C",
         "可能原因: 风扇故障 / 机房空调故障 / 通风口堵塞"),
        ("AclDenySurge", "warning", "防火墙",
         "ACL 拒绝计数在 {duration} 分钟内增加了 {count} 次",
         "可能原因: 扫描攻击 / 误配置 / 新业务未放通策略"),
        ("OspfNeighborDown", "critical", "核心路由器",
         "OSPF 邻居 {neighbor} 状态变为 Down",
         "可能原因: 网络类型不匹配 / Hello/Dead 间隔不一致 / 认证失败"),
    ]

    import random
    random.seed(42)
    alarms = []
    for i in range(200):
        t = random.choice(alarm_templates)
        name, severity, device, desc_tmpl, root_cause = t
        desc = desc_tmpl.format(
            cpu=random.randint(85, 99),
            mem=random.randint(85, 98),
            neighbor=f"10.0.{random.randint(1,255)}.{random.randint(1,255)}",
            duration=random.randint(5, 60),
            port=random.randint(1, 24),
            count=random.randint(10, 100),
            target=f"192.168.{random.randint(1,255)}.1",
            normal=random.randint(1, 5),
            current=random.randint(50, 500),
            loss=random.randint(5, 30),
            partition="/var/log",
            usage=random.randint(85, 98),
            temp=random.randint(71, 95),
        )
        alarms.append({
            "alarm_id": f"ALM-{i+1:04d}",
            "name": name, "severity": severity, "device": device,
            "description": desc, "root_cause": root_cause,
            "timestamp": f"2026-07-{random.randint(1,15):02d} {random.randint(0,23):02d}:{random.randint(0,59):02d}:{random.randint(0,59):02d}",
        })
    print(f"       Generated {len(alarms)} alarm records")
    return alarms


def generate_network_knowledge() -> list[dict]:
    """生成网络运维知识库文档（基于真实配置命令）"""
    print(f"[3/4] Generating network knowledge base...")

    docs = [
        {
            "title": "OSPF 协议配置指南",
            "content": """# OSPF 协议配置指南

## 基本配置
```
router ospf 1
 router-id 1.1.1.1
 network 10.0.0.0 0.0.0.255 area 0
```

## 邻居状态检查
```
show ip ospf neighbor
show ip ospf interface
```

## 常见故障排查
1. **邻居卡在 Init 状态**: 检查 Hello/Dead 间隔是否一致，网络类型是否匹配
2. **邻居卡在 Exstart/Exchange**: 检查 MTU 是否一致，接口 MTU 不匹配会导致 DBD 交换失败
3. **邻居 Down**: 检查底层连通性（ping），检查 ACL 是否拦截了 OSPF 协议报文

## 网络类型
- Broadcast: 选举 DR/BDR，Hello=10s，Dead=40s
- Point-to-Point: 不选举 DR/BDR，Hello=10s，Dead=40s
- NBMA: 需手动指定邻居，Hello=30s，Dead=120s""",
        },
        {
            "title": "BGP 协议排障手册",
            "content": """# BGP 协议排障手册

## 邻居无法建立
1. 检查 TCP 179 端口连通性: `telnet <邻居IP> 179`
2. 检查 AS 号配置: `show run | include router bgp`
3. 检查 update-source: BGP 默认用最近接口 IP，需确认邻居可达
4. 检查 ebgp-multihop: EBGP 多跳需要手动指定

## 路由不学习
1. 检查 network 命令是否覆盖了目标网段
2. 检查下一跳是否可达: `show ip bgp <prefix>`
3. 检查路由策略: route-map、prefix-list、filter-list

## BGP 状态机
```
Idle → Connect → Active → OpenSent → OpenConfirm → Established
```
- Idle: 初始状态，等待 start 事件
- Active: TCP 连接失败，正在重试
- Established: 邻居正常，可以交换路由""",
        },
        {
            "title": "VLAN 配置命令参考",
            "content": """# VLAN 配置命令参考

## 创建 VLAN
```
vlan 10
 name Engineering
```

## Access 端口配置
```
interface GigabitEthernet0/1
 switchport mode access
 switchport access vlan 10
```

## Trunk 端口配置
```
interface GigabitEthernet0/24
 switchport mode trunk
 switchport trunk allowed vlan 10,20,30
```

## 常见问题
1. **Access 口不转发数据**: 检查 VLAN 是否存在，端口是否 admin down
2. **Trunk 口不通**: 检查两端 native VLAN 是否一致
3. **跨 VLAN 不通**: 需要三层设备（路由器/三层交换机）做 inter-VLAN routing""",
        },
        {
            "title": "网络故障通用排查流程",
            "content": """# 网络故障通用排查流程

## 三层排查法

### 1. 物理层检查
- 网线是否松动
- 光模块是否正常（show interface transceiver）
- 端口是否 up（show interface status）

### 2. 链路层检查
- VLAN 配置是否正确
- STP 状态是否正常（show spanning-tree）
- MAC 地址表是否学习到（show mac address-table）

### 3. 网络层检查
- IP 地址配置是否正确
- 路由是否可达（show ip route）
- ACL 是否拦截了流量

## 常用诊断命令
- `ping` — 测试连通性
- `traceroute` — 追踪路由路径
- `show interface` — 查看接口统计（错误计数、丢包）
- `show log` — 查看系统日志
- `show processes cpu sorted` — 查看 CPU 占用

## 性能问题排查
1. CPU 高: show processes cpu sorted → 定位高 CPU 进程 → 优化或限制
2. 内存高: show memory → 检查路由表大小、ARP 表大小
3. 延迟高: ping + traceroute → 定位瓶颈链路""",
        },
        {
            "title": "常见错误码速查",
            "content": """# 常见网络错误码速查

## HTTP 错误码
- 500 Internal Server Error: 服务端内部错误，检查应用日志
- 502 Bad Gateway: 网关收到无效响应，上游服务可能挂了
- 503 Service Unavailable: 服务过载或维护中
- 504 Gateway Timeout: 网关等待上游超时，检查上游服务响应时间

## Syslog 严重级别
- 0 Emergency: 系统不可用
- 1 Alert: 需要立即处理
- 2 Critical: 严重故障
- 3 Error: 错误
- 4 Warning: 警告
- 5 Notice: 通知
- 6 Informational: 信息
- 7 Debug: 调试

## SNMP Trap 常见类型
- linkDown: 接口 down
- linkUp: 接口恢复
- authenticationFailure: 认证失败
- coldStart: 设备冷启动
- warmStart: 设备热启动""",
        },
        {
            "title": "交换机基础配置模板",
            "content": """# 交换机基础配置模板

## 初始配置
```
hostname Core-Switch-01
enable secret <password>
service password-encryption
ip domain-name example.com
```

## 管理接口
```
interface Vlan1
 ip address 192.168.1.1 255.255.255.0
 no shutdown
```

## SSH 配置
```
ip ssh version 2
line vty 0 4
 transport input ssh
 login local
```

## NTP 配置
```
ntp server 192.168.1.100
clock timezone CST 8
```

## Syslog 配置
```
logging host 192.168.1.200
logging trap informational
logging facility local7
```

## SNMP 配置
```
snmp-server community public RO
snmp-server location DataCenter-A
snmp-server contact admin@example.com
```""",
        },
    ]
    print(f"       Generated {len(docs)} knowledge documents")
    return docs


# ============================================================
# 2. 导入 Milvus
# ============================================================

def import_to_milvus(alarms: list[dict], syslogs: list[dict], knowledge: list[dict]):
    """将所有数据导入 Milvus 知识库"""
    print(f"[4/4] Importing to Milvus...")

    from langchain_core.documents import Document
    from documents.splitter import split_documents
    from documents.cleaner import full_clean
    from retrieval.vector_store import vector_store_manager
    from retrieval.hybrid_search import hybrid_searcher

    all_docs = []

    # 告警数据 → 文档
    for alarm in alarms:
        content = (
            f"[{alarm['severity'].upper()}] {alarm['name']} — {alarm['device']}\n"
            f"描述: {alarm['description']}\n"
            f"根因分析: {alarm['root_cause']}\n"
            f"告警ID: {alarm['alarm_id']} | 时间: {alarm['timestamp']}"
        )
        all_docs.append(Document(
            page_content=content,
            metadata={"_source": "alarm_dataset", "_file_name": "network_alarms.txt",
                      "_scope": "global", "_type": "alarm"},
        ))

    # Syslog 数据 → 文档
    for log in syslogs:
        content = str(log) if isinstance(log, str) else json.dumps(log, ensure_ascii=False)
        # 只取有用的字段
        if isinstance(log, dict):
            parts = [f"{k}: {v}" for k, v in log.items() if v and k not in ('row_id', 'index')]
            content = " | ".join(parts[:8])
        all_docs.append(Document(
            page_content=content,
            metadata={"_source": "syslog_dataset", "_file_name": "syslog_data.txt",
                      "_scope": "global", "_type": "syslog"},
        ))

    # 知识库文档
    for kd in knowledge:
        source = f"{kd['title']}.md"
        all_docs.append(Document(
            page_content=f"# {kd['title']}\n\n{kd['content']}",
            metadata={"_source": source, "_file_name": source,
                      "_scope": "global", "_type": "knowledge"},
        ))

    # 分块 + 清洗
    print(f"       Total docs before split: {len(all_docs)}")
    chunks = split_documents(all_docs, chunk_size=800, chunk_overlap=100)
    print(f"       After split: {len(chunks)} chunks")

    clean_result = full_clean(chunks, skip_l3=True, skip_l6=True)
    cleaned = clean_result.documents
    print(f"       After cleaning: {len(cleaned)} chunks")

    # 删除旧测试数据
    vector_store_manager.delete_by_source("alarm_dataset")
    vector_store_manager.delete_by_source("syslog_dataset")
    for kd in knowledge:
        vector_store_manager.delete_by_source(f"{kd['title']}.md")

    # 入库
    ids = vector_store_manager.add_documents(cleaned)
    print(f"       Imported {len(ids)} chunks to Milvus")

    # 重建 BM25 索引
    try:
        all_indexed = vector_store_manager.similarity_search("network", k=100)
        hybrid_searcher.index(all_indexed)
        print(f"       BM25 index rebuilt: {len(all_indexed)} docs")
    except Exception as e:
        print(f"       BM25 rebuild skipped: {e}")

    print(f"\n   Done! {len(cleaned)} chunks in Milvus.")
    print(f"   Try: 'OSPF 协议怎么配置' or '最近有什么告警'")


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("Agent Workshop — Test Data Loader")
    print("=" * 60)

    # 1. 下载 Syslog 数据
    syslogs = download_syslog_dataset()
    if not syslogs:
        # Fallback: 生成一些基本 syslog
        syslogs = [
            {"timestamp": "2026-07-15 10:23:45", "host": "core-switch-01",
             "facility": "daemon", "severity": "warning",
             "message": "Interface GigabitEthernet0/5 changed state to down"},
            {"timestamp": "2026-07-15 10:24:01", "host": "core-switch-01",
             "facility": "daemon", "severity": "info",
             "message": "Interface GigabitEthernet0/5 changed state to up"},
            {"timestamp": "2026-07-15 10:30:12", "host": "edge-router-01",
             "facility": "local7", "severity": "error",
             "message": "BGP neighbor 10.0.1.1 state changed from Established to Idle"},
            {"timestamp": "2026-07-15 11:15:33", "host": "firewall-01",
             "facility": "local4", "severity": "warning",
             "message": "ACL deny count exceeded 1000/min from 192.168.1.100"},
            {"timestamp": "2026-07-15 12:00:05", "host": "core-router-01",
             "facility": "daemon", "severity": "critical",
             "message": "OSPF neighbor 10.0.2.1 state changed from Full to Down, reason: Dead timer expired"},
        ]

    # 2. 生成告警数据
    alarms = download_alarm_dataset()

    # 3. 生成知识库文档
    knowledge = generate_network_knowledge()

    # 4. 导入 Milvus
    import_to_milvus(alarms, syslogs, knowledge)

    print("\nTest data loaded. Run: python benchmark.py")


if __name__ == "__main__":
    main()
