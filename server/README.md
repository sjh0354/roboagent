# Server Deployment Guide

This directory currently contains two independent services:

- `funasr_server.py`: remote ASR service
- `lark_gateway.py`: native Feishu/Lark gateway for robot-to-robot messaging

## Native Lark Gateway

### Install

```bash
cd server
pip install -r requirements.txt
```

Configuration is now split by responsibility:

- `env/*.example.sh`: checked-in templates
- `env/common.env.sh`: shared API keys and model defaults
- `env/planner.g1.env.sh`: G1 planner-side transport settings
- `env/planner.ur5e.env.sh`: UR5e planner-side transport settings
- `env/planner.env.sh`: backward-compatible default that currently points to G1
- `env/lark.env.sh`: gateway and WebSocket bridge settings
- `apikey.sh`: compatibility wrapper that sources all three

Initialize local env files from templates before first use:

```bash
bash scripts/init_env.sh
```

### Configure One Bot

```bash
export LARK_APP_ID='cli_xxx'
export LARK_APP_SECRET='xxx'
export LARK_VERIFICATION_TOKEN='xxx'
export LARK_DEFAULT_ACCOUNT='default'
```

### Configure Multiple Bots

```bash
export LARK_ACCOUNT_IDS='g1,ur5e'

export LARK_G1_APP_ID='cli_xxx'
export LARK_G1_APP_SECRET='xxx'
export LARK_G1_VERIFICATION_TOKEN='xxx'

export LARK_UR5E_APP_ID='cli_yyy'
export LARK_UR5E_APP_SECRET='yyy'
export LARK_UR5E_VERIFICATION_TOKEN='yyy'
```

### Start

```bash
source ../env/common.env.sh
source ../env/lark.env.sh
python lark_gateway.py
```

For persistent-connection event subscription, start the Node bridge too:

```bash
source ../env/common.env.sh
source ../env/lark.env.sh
node lark_ws_bridge.js
```

Or start both in one detached tmux session from the repo root:

```bash
bash scripts/start_lark_tmux.sh
```

Stop the detached session with:

```bash
bash scripts/stop_lark_tmux.sh
```

For the common G1 MVP flow, you can start gateway, bridge, and humanoid planner
in one detached tmux session:

```bash
bash scripts/start_g1_demo_tmux.sh
```

Stop it with:

```bash
bash scripts/stop_g1_demo_tmux.sh
```

For the UR5e planner, use:

```bash
bash scripts/start_ur5e_demo_tmux.sh
```

Default bind:

- host: `0.0.0.0`
- port: `18889`

### Subscription Mode

Recommended for this project:

- use Feishu/Lark persistent connection mode
- keep `lark_ws_bridge.js` running

Optional fallback:

- use webhook callback mode and point the callback URL to the Python gateway

### Feishu Callback URLs

For each bot account, configure event subscription to:

- `http://<your-host>:18889/webhook/g1`
- `http://<your-host>:18889/webhook/ur5e`

Supported flow:

- persistent WebSocket event subscription through `lark_ws_bridge.js`
- URL verification
- `im.message.receive_v1`
- local polling via `/api/messages/read`
- outbound send via `/api/messages/send`

Current limitation:

- encrypted event payloads are not supported yet; leave Feishu event encryption disabled

---

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
