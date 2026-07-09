"""
独立 MCP Server — ping / traceroute / nslookup

FastMCP 3.x 实现，独立进程运行。
Agent 通过 MultiServerMCPClient 动态发现工具。

启动: python -m mcp_tools.mcp_server
"""

import platform
import socket
import subprocess
from fastmcp import FastMCP

mcp = FastMCP("NetworkDiagnostics")


@mcp.tool()
def ping(host: str, count: int = 4) -> str:
    """对目标主机执行 ping，检测网络连通性和延迟。host: IP或域名, count: 次数(默认4)"""
    system = platform.system().lower()
    cmd = ["ping", "-n" if "win" in system else "-c", str(count), host]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return r.stdout or r.stderr
    except subprocess.TimeoutExpired:
        return f"ping {host} 超时"
    except Exception as e:
        return f"ping 失败: {e}"


@mcp.tool()
def traceroute(host: str, max_hops: int = 15) -> str:
    """路由追踪，显示到目标主机的路径。host: IP或域名, max_hops: 最大跳数(默认15)"""
    system = platform.system().lower()
    cmd = ["tracert" if "win" in system else "traceroute",
           "-h" if "win" in system else "-m", str(max_hops), host]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        return r.stdout or r.stderr
    except subprocess.TimeoutExpired:
        return f"traceroute {host} 超时"
    except Exception as e:
        return f"traceroute 失败: {e}"


@mcp.tool()
def nslookup(host: str) -> str:
    """DNS 查询，解析域名对应的 IP 地址。host: 域名"""
    try:
        result = socket.getaddrinfo(host, None)
        ips = list(set(addr[4][0] for addr in result))
        return f"{host} → {', '.join(ips)}"
    except socket.gaierror:
        return f"无法解析: {host}"
    except Exception as e:
        return f"DNS 查询失败: {e}"


if __name__ == "__main__":
    print("NetworkDiagnostics MCP Server: http://127.0.0.1:8005")
    mcp.run(transport="streamable-http", host="127.0.0.1", port=8005)
