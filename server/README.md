# FunASR Remote Server Deployment Guide

This guide explains how to deploy the ASR inference server on a workstation (e.g., IP: `192.168.24.103`) and configure the robot (e.g., IP: `192.168.24.294`) to access it over the local network.

## 1. Workstation Setup (Server Side)

**Target Machine**: Workstation (`192.168.24.103`)

### 1.1 Prerequisites
Ensure you have Python installed (Python 3.8+ recommended). It's best to use a fresh Conda environment.

```bash
conda create -n funasr-server python=3.10
conda activate funasr-server
```

### 1.2 Install Dependencies
Navigate to the `server` directory and install the required packages.

```bash
cd server
pip install -r requirements.txt
```
*Note: This will install `funasr`, `modelscope`, `fastapi`, `uvicorn`, `torch` etc. Ensure you have CUDA drivers installed if you want GPU acceleration.*

### 1.3 Start the Server
Run the server script. It is configured to listen on `0.0.0.0`, which means it accepts connections from external IPs.

```bash
python funasr_server.py
```
**Output should indicate:**
> 🚀 Starting FunASR Server on 0.0.0.0:8000

Alternatively, run with `uvicorn` directly for more control:
```bash
uvicorn funasr_server:app --host 0.0.0.0 --port 8000
```

### 1.4 Firewall Check (Crucial)
Ensure port **8000** is open on the workstation's firewall.
- **Ubuntu/Linux**: `sudo ufw allow 8000`
- **Windows**: Allow Python through Windows Defender Firewall.
- **Mac**: Allow Python to accept incoming connections.

---

## 2. Robot Setup (Client Side)

**Target Machine**: Robot (`192.168.24.294`)

### 2.1 Configure Environment Variable
The robot's code (`utils/funasr_manager.py`) looks for the `ASR_SERVER_URL` environment variable.

You can set this temporarily in the terminal before running the planner:

```bash
export ASR_SERVER_URL="http://192.168.24.103:8000"
```

Or add it to your `apikey.sh` or `.bashrc` for persistence:

```bash
echo 'export ASR_SERVER_URL="http://192.168.24.103:8000"' >> apikey.sh
source apikey.sh
```

### 2.2 Run the Planner
Now, simply run your robot planner as usual. It will automatically detect the server URL and switch to remote mode.

```bash
python planner/humanoid_planner_vlm.py
```

---

## 3. Verification & Troubleshooting

### Verify Connectivity from Robot
Before running the planner, try to ping the workstation from the robot:

```bash
ping 192.168.24.103
```

If ping works, try a simple curl command to check if the port is accessible (FastAPI provides a doc page at `/docs`):

```bash
curl -I http://192.168.24.103:8000/docs
```
*Expected: `HTTP/1.1 200 OK`*

### Common Issues
1.  **Connection Refused**: 
    - Check if server is running.
    - Check if Firewall on `192.168.24.103` is blocking port 8000.
    - Ensure Workstation and Robot are on the same Wi-Fi/Subnet.
2.  **Timeouts**:
    - Network might be slow. The client currently has a 5s timeout. If inference is slow, increase `timeout=5` in `utils/funasr_manager.py`.
3.  **Dependency Errors on Server**:
    - If `funasr` fails to load, try `pip install -U funasr modelscope`.
