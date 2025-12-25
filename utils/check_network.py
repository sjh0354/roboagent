#!/usr/bin/env python3
import os
import sys
import socket
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

    # 3. Check HTTPS
    # Note: Accessing root might return 404 or 403, but confirms connectivity
    check_https(f"https://{endpoint}")

    print("\n" + "="*50)
    print("Diagnosis Suggestion:")
    print("1. If DNS failed: Check your router or /etc/resolv.conf")
    print("2. If TCP failed: Check firewall or proxy settings")
    print("3. If Latency is high (>1000ms): Using DashScope TTS might be unstable")
    print("="*50)

if __name__ == "__main__":
    main()
