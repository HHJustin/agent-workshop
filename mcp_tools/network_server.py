"""
MCP 网络诊断工具 — 作为标准 @tool 注册到 Agent

面试考点：
    Q: "MCP 工具和 @tool 有什么区别？"
    A: 这些工具就是标准 @tool，Agent 直接调用。MCP 架构在 mcp_client.py 里——
       MultiServerMCPClient + 自动发现 + 进程隔离 + 指数退避重试。
       当前 MCP Server 用 @tool 实现（零依赖，即开即用），
       生产环境部署为独立 MCP Server 进程时改一行配置即可。

启动: 无需单独启动，Agent 自动加载
"""

import platform
import socket
import subprocess
from langchain_core.tools import tool


@tool
def ping(host: str, count: int = 4) -> str:
    """对目标主机执行 ping，检测网络连通性和延迟。
    参数 host: 目标主机 IP 或域名，如 '8.8.8.8'
    参数 count: ping 次数，默认 4
    """
    system = platform.system().lower()
    cmd = ["ping", "-n" if "win" in system else "-c", str(count), host]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return r.stdout or r.stderr
    except subprocess.TimeoutExpired:
        return f"ping {host} 超时（30s）"
    except Exception as e:
        return f"ping 失败: {e}"


@tool
def traceroute(host: str, max_hops: int = 15) -> str:
    """路由追踪，显示到目标主机经过的路径。
    参数 host: 目标主机 IP 或域名
    参数 max_hops: 最大跳数，默认 15
    """
    system = platform.system().lower()
    if "win" in system:
        cmd = ["tracert", "-h", str(max_hops), host]
    else:
        cmd = ["traceroute", "-m", str(max_hops), host]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        return r.stdout or r.stderr
    except subprocess.TimeoutExpired:
        return f"traceroute {host} 超时（60s）"
    except Exception as e:
        return f"traceroute 失败: {e}"


@tool
def nslookup(host: str) -> str:
    """DNS 查询，解析域名对应的 IP 地址。
    参数 host: 域名，如 'www.baidu.com'
    """
    try:
        result = socket.getaddrinfo(host, None)
        ips = list(set(addr[4][0] for addr in result))
        return f"{host} → {', '.join(ips)}"
    except socket.gaierror:
        return f"无法解析域名: {host}"
    except Exception as e:
        return f"DNS 查询失败: {e}"


MCP_NETWORK_TOOLS = (ping, traceroute, nslookup)
