#!/usr/bin/env python3
import os
import sys
import socket
import ssl
import time
import requests

def check_dns(hostname):
    print(f"🔍 Checking DNS for {hostname}...")
    try:
        ip = socket.gethostbyname(hostname)
        print(f"   ✅ Resolved to: {ip}")
        return True
    except socket.gaierror:
        print(f"   ❌ DNS Resolution failed")
        return False

def check_tcp(hostname, port):
    print(f"🔍 Checking TCP connection to {hostname}:{port}...")
    try:
        start = time.time()
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex((hostname, port))
        sock.close()
        duration = (time.time() - start) * 1000
        
        if result == 0:
            print(f"   ✅ Connected in {duration:.2f}ms")
            return True
        else:
            print(f"   ❌ Connection failed (Code: {result})")
            return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def check_https(url):
    print(f"🔍 Checking HTTPS request to {url}...")
    try:
        start = time.time()
        response = requests.get(url, timeout=5)
        duration = (time.time() - start) * 1000
        print(f"   ✅ Status: {response.status_code}, Time: {duration:.2f}ms")
        return True
    except requests.exceptions.Timeout:
        print(f"   ❌ Request timed out (>5s)")
        return False
    except requests.exceptions.ConnectionError as e:
        print(f"   ❌ Connection Error: {e}")
        return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def print_proxy_env():
    print("🔍 Checking proxy environment variables...")
    found = False
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "all_proxy", "no_proxy"):
        value = os.environ.get(key)
        if value:
            print(f"   {key}={value}")
            found = True
    if not found:
        print("   ✅ No proxy environment variables set")

def check_https_phases(hostname, path="/"):
    print(f"🔍 Checking HTTPS phases for {hostname}{path}...")
    timings = {}
    sock = None
    tls_sock = None
    try:
        t0 = time.time()
        addr_info = socket.getaddrinfo(hostname, 443, type=socket.SOCK_STREAM)
        timings["dns"] = time.time() - t0
        address = addr_info[0][4]

        sock = socket.socket(addr_info[0][0], socket.SOCK_STREAM)
        sock.settimeout(15)
        t1 = time.time()
        sock.connect(address)
        timings["tcp_connect"] = time.time() - t1

        context = ssl.create_default_context()
        t2 = time.time()
        tls_sock = context.wrap_socket(sock, server_hostname=hostname)
        sock = None
        timings["tls_handshake"] = time.time() - t2

        request = (
            f"GET {path} HTTP/1.1\r\n"
            f"Host: {hostname}\r\n"
            "User-Agent: ras-network-check/1.0\r\n"
            "Connection: close\r\n\r\n"
        ).encode("ascii")
        t3 = time.time()
        tls_sock.sendall(request)
        first = tls_sock.recv(1)
        timings["time_to_first_byte"] = time.time() - t3
        timings["total_to_first_byte"] = time.time() - t0

        if first:
            print(f"   ✅ DNS: {timings['dns'] * 1000:.2f}ms")
            print(f"   ✅ TCP connect: {timings['tcp_connect'] * 1000:.2f}ms")
            print(f"   ✅ TLS handshake: {timings['tls_handshake'] * 1000:.2f}ms")
            print(f"   ✅ Time to first byte: {timings['time_to_first_byte'] * 1000:.2f}ms")
            print(f"   ✅ Total to first byte: {timings['total_to_first_byte'] * 1000:.2f}ms")
            return True

        print("   ❌ Connection closed before first byte")
        return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        if timings:
            for key, value in timings.items():
                print(f"   - {key}: {value * 1000:.2f}ms")
        return False
    finally:
        if tls_sock:
            tls_sock.close()
        if sock:
            sock.close()

def main():
    print("="*50)
    print("🌐 DashScope Network Connectivity Check")
    print("="*50)
    
    endpoint = "dashscope.aliyuncs.com"
    
    # 1. Check DNS
    if not check_dns(endpoint):
        print("\n❌ Critical: DNS failure. Check your internet connection or DNS settings.")
        return

    # 2. Check TCP (Port 443)
    check_tcp(endpoint, 443)

    # 3. Check proxy env
    print_proxy_env()

    # 4. Check HTTPS phase timing without requests/proxy abstractions
    check_https_phases(endpoint)

    # 5. Check HTTPS through requests
    # Note: Accessing root might return 404 or 403, but confirms connectivity
    check_https(f"https://{endpoint}")

    print("\n" + "="*50)
    print("Diagnosis Suggestion:")
    print("1. If DNS failed: Check your router or /etc/resolv.conf")
    print("2. If TCP failed: Check firewall or proxy settings")
    print("3. If TLS handshake is high: check proxy/firewall/TLS inspection/SNI handling")
    print("4. If time-to-first-byte is high: DashScope endpoint/CDN/server path is slow from this network")
    print("5. If requests is slow but phase timing is fast: check proxy env, certificate verification, or requests settings")
    print("="*50)

if __name__ == "__main__":
    main()
