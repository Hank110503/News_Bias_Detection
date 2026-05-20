import socket
import os
import sys

# 颜色定义，让结果更直观
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
ENDC = '\033[0m'

def print_status(msg, status):
    color = GREEN if status else RED
    symbol = "✅" if status else "❌"
    print(f"{color}{symbol} {msg}{ENDC}")

def check_env_vars():
    """检查代理环境变量"""
    print(f"\n--- 1. 检查环境变量 ---")
    http_proxy = os.getenv('HTTP_PROXY')
    https_proxy = os.getenv('HTTPS_PROXY')
    
    if http_proxy:
        print(f"ℹ️  HTTP_PROXY: {http_proxy}")
    else:
        print(f"⚠️  HTTP_PROXY: 未设置")
        
    if https_proxy:
        print(f"ℹ️  HTTPS_PROXY: {https_proxy}")
    else:
        print(f"⚠️  HTTPS_PROXY: 未设置")

def check_dns(host):
    """检查 DNS 解析"""
    print(f"\n--- 2. 检查 DNS 解析 ({host}) ---")
    try:
        ip = socket.gethostbyname(host)
        print_status(f"域名 {host} 解析成功: {ip}", True)
        return True
    except socket.gaierror:
        print_status(f"域名 {host} 解析失败 (DNS问题)", False)
        return False

def check_tcp_connectivity(host, port, timeout=5):
    """检查 TCP 端口连通性 (模拟 MCP 连接)"""
    print(f"\n--- 3. 检查 TCP 端口 ({host}:{port}) ---")
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()
        
        if result == 0:
            print_status(f"端口 {host}:{port} 连通 (服务正常)", True)
            return True
        else:
            print_status(f"端口 {host}:{port} 无法连接 (服务未启动或防火墙拦截)", False)
            return False
    except Exception as e:
        print_status(f"连接 {host}:{port} 发生错误: {e}", False)
        return False

def check_http_connectivity(url, port=80, timeout=5):
    """检查 HTTP 基础连通性"""
    print(f"\n--- 4. 检查 HTTP 连通性 ({url}) ---")
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        sock.connect((url, port))
        sock.close()
        print_status(f"HTTP 服务 {url}:{port} 可达", True)
        return True
    except socket.timeout:
        print_status(f"连接 {url}:{port} 超时 (可能是代理配置错误或网络阻断)", False)
        return False
    except Exception as e:
        print_status(f"连接 {url}:{port} 失败: {e}", False)
        return False

if __name__ == "__main__":
    print("🚀 开始网络诊断...")
    
    # 1. 检查环境变量
    check_env_vars()
    
    # 2. 检查 DNS (使用 google 和 github 这种通常需要联网的域名)
    check_dns("www.google.com")
    check_dns("github.com")
    
    # 3. 检查常见 MCP/本地服务端口
    # 如果你的 codex_apps 运行在特定端口，可以在这里修改
    check_tcp_connectivity("127.0.0.1", 8080) 
    check_tcp_connectivity("localhost", 3000)

    # 4. 检查公网 HTTP 连通性
    check_http_connectivity("www.google.com", 80)
    
    print("\n🏁 诊断结束。")