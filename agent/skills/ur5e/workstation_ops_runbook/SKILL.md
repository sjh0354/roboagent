# Skill: UR5e Workstation Ops Runbook

Use when:
- You need to run or validate `RAS_interactive_planner` / `RAS_interactivate_planner` on workstation environments.
- You need a repeatable procedure that avoids path confusion, env drift, and proxy-related startup failures.
- You need to decide when to hand off to @Ako for hardware-chain execution.

Naming note:
- Repository naming may differ by host (`interactive` vs `interactivate`). Always use the locally visible path on the current machine.

## 1) `ras` Environment Constraint

Rules:
- Always run with conda env `ras`.
- Prefer `conda run -n ras ...` for one-shot commands.
- If using interactive shell mode, `conda activate ras` first, then run commands.
- Never assume system Python has required packages.

Baseline checks:
```bash
conda env list | rg '(^|\\s)ras\\s'
conda run -n ras python -m pip install -r <REPO_PATH>/requirements.txt
```

## 2) Path Mapping (`/media` vs `/home`)

Known host layouts:
- `@rinrin` host primary repo path:
  - `/media/user/B29202FA9202C2B91/RAS_interactive_planner`
- `@Ako` host primary repo path:
  - `/home/ef/projects/RAS_interactivate_planner`
- UR hardware project path on `@Ako` host:
  - `/home/ef/projects/ur5e-arm-teleoperation`

Rule:
- First resolve local visible path, then substitute all commands with that path consistently.

## 3) Proxy Handling (`socks` Compatibility)

Observed blocker:
- Startup can fail with: `Unknown scheme for proxy URL('socks://...')`.

Mitigation for planner launch:
```bash
unset ALL_PROXY all_proxy HTTP_PROXY http_proxy HTTPS_PROXY https_proxy
```

Rule:
- If startup fails on proxy scheme, clear proxy env vars for this run, or switch to a runtime-supported proxy scheme before retry.

## 4) Three-Step Verification Commands

Replace `<REPO_PATH>` with your local visible repo path.

Step 1: dependency baseline
```bash
conda run -n ras python -m pip install -r <REPO_PATH>/requirements.txt
```

Step 2: entrypoint check
```bash
conda run -n ras python <REPO_PATH>/planner/arm_planner_vlm.py --help
```

Step 3: controlled simulation regression (recommended default: `--transport none`)
```bash
test -f <REPO_PATH>/env/common.env.sh || { echo "missing env file: <REPO_PATH>/env/common.env.sh"; exit 1; }
set -a
source <REPO_PATH>/env/common.env.sh
set +a
unset ALL_PROXY all_proxy HTTP_PROXY http_proxy HTTPS_PROXY https_proxy
printf 'q\\n' | conda run -n ras python -u <REPO_PATH>/planner/arm_planner_vlm.py --simulation --log --transport none
```

Step 3 (unattended/CI variant with timeout):
```bash
test -f <REPO_PATH>/env/common.env.sh || { echo "missing env file: <REPO_PATH>/env/common.env.sh"; exit 1; }
set -a
source <REPO_PATH>/env/common.env.sh
set +a
unset ALL_PROXY all_proxy HTTP_PROXY http_proxy HTTPS_PROXY https_proxy
timeout 60s conda run --no-capture-output -n ras python -u <REPO_PATH>/planner/arm_planner_vlm.py --simulation --log --transport none
```

Pass criteria:
- No `GENAI_API_KEY environment variable not set` error.
- Planner reaches `AUTONOMOUS ARM VLM PLANNER - Ready`.
- Process exits cleanly after `q`.

## 5) When to Hand Off to @Ako (Hardware Chain)

Trigger handoff to @Ako when any of these apply:
- Real depth camera (RealSense) verification is required.
- Real UR5e arm action chain is required.
- PI0 Docker execution path under `/home/ef/projects/ur5e-arm-teleoperation` is required.
- AnyGrasp real execution validation is required in `@Ako` host-visible paths.

Suggested handoff packet:
- Exact command executed.
- Exit code.
- First blocker line.
- Relevant env toggles (`ARM_ACT_BACKEND`, `ANYGRASP_*`, `UR_GRASP_EXECUTOR_CMD`).

## Security Notes

- Do not post API keys in chat plaintext.
- Share only variable names and file paths (for example `GENAI_API_KEY` in `env/common.env.sh`).


## 6) v2 Motion-Only Safety SOP (Real UR5e)

Default posture:
- Keep `UR_ENABLE_MOTION=0` outside approved execution windows.
- Keep `full` mode hard-blocked.
- Allow only `motion_only` in approved windows, with human watch + E-stop ready.

Required safety gate (must all pass):
- `ur_enable_motion=true` (`UR_ENABLE_MOTION=1`)
- `joint_limit_ok=true`
- `step_ok=true` (`max_joint_step <= 0.02`)
- `rate_ok=true` (`control_rate <= 10`)
- `gripper_blocked=true`

Window pre-check:
```bash
echo "Operator watch: ON, E-stop ready: ON"
export UR_ENABLE_MOTION=1
export ACTION_CHUNK_SIZE=5
export CONTROL_RATE=10
export MAX_JOINT_STEP=0.02
```

Executor binding:
```bash
export UR_GRASP_EXECUTOR_CMD="python <REPO_PATH>/utils/ur_policy_bridge_client.py \
  --mode motion_only \
  --policy-host ${POLICY_HOST:-127.0.0.1} \
  --policy-port ${POLICY_PORT:-8001} \
  --action-chunk-size ${ACTION_CHUNK_SIZE} \
  --control-rate ${CONTROL_RATE} \
  --max-joint-step ${MAX_JOINT_STEP}"
```

Single-run validation command:
```bash
timeout 120 conda run --no-capture-output -n ras_cpy python <REPO_PATH>/utils/anygrasp_runner.py \
  --request-json '{"action":"pick_and_place","observation_image":"<RGB_PATH>","depth_path":"<DEPTH_PATH>","joint_state":[0,0,0,0,0,0]}'
```

Mandatory report fields after each window run:
- `exit code`
- last-line JSON
- `data.execution.safety_gate` (or `data.execution.executor_result.data.safety_gate`)
- key safety logs proving:
  - `pre_grasp_high_point_only`
  - `gripper_command=blocked`
  - no `full` path

Window close (always run):
```bash
export UR_ENABLE_MOTION=0
```

Pass criteria:
- Command exits without unsafe-motion errors.
- Safety gate fields are all green.
- Motion profile is only pre-grasp high-point and gripper remains blocked.

## 7) RTDE Occupancy Rule (Residual Process Cleanup)

Observed recurrent blocker:
- `RTDE input registers are already in use`

Default assumption:
- This is usually caused by residual control processes from a previous run, or concurrent control sessions.

Hard rule:
- Before each real-robot window, run cleanup + single-chain startup.
- Never keep multiple terminal/container sessions controlling the same UR controller at the same time.

Pre-window cleanup:
```bash
pkill -f ur5e_single_arm.launch.py || true
pkill -f ur5e_robot || true
pkill -f ur_rtde || true
docker exec exp_ef_ur5e-arm-teleopration bash -lc 'pkill -f ur5e_single_arm.launch.py || true; pkill -f ur5e_robot || true; pkill -f ur_rtde || true'
```

RTDE port sanity check:
```bash
python - <<'PY'
import socket
ip='192.168.24.245'
for p in [29999,30004]:
    s=socket.socket(); s.settimeout(2)
    try:
        s.connect((ip,p)); print(p,'open')
    except Exception as e:
        print(p,'fail',e)
    finally:
        s.close()
PY
```

Single-chain startup (only one launcher):
```bash
docker exec -it exp_ef_ur5e-arm-teleopration bash -lc '
source /ros_entrypoint.sh
source /opt/ros/humble/setup.bash
source /home/ef/projects/ur5e-arm-teleoperation/install/setup.bash
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-131}
ros2 launch ur5e_teleopration ur5e_single_arm.launch.py robot_name:=arm_1 robot_ip:=192.168.24.245 use_real_robot:=true
'
```

Ready gate before AnyGrasp/bridge run:
- `30004 open`
- `/ur5e_robot` visible
- `/arm_1/arm_joint_states` can be echoed once

If still occupied:
- On teach pendant, disable Fieldbus adapters that occupy RTDE (at least EtherNet/IP; if needed PROFINET/MODBUS).
- Ensure previous robot program/session is stopped before relaunch.

Runbook helper (one-shot):
```bash
bash scripts/ur_rtde_preflight.sh
```

Optional modes:
```bash
# only validate readiness gates, skip cleanup
bash scripts/ur_rtde_preflight.sh --check-only

# only cleanup residual processes
bash scripts/ur_rtde_preflight.sh --cleanup-only
```

Gate levels:
```bash
# state gate: only require /arm_1/arm_joint_states visibility + one-shot echo
bash scripts/ur_rtde_preflight.sh --check-only --state-gate

# motion gate (default): require /ur5e_robot + command topic visibility
bash scripts/ur_rtde_preflight.sh --check-only --motion-gate
```

## 8) Micro-Increment Ladder (0.005 -> 0.01)

Scope:
- Only for `motion_only` validation.
- Do NOT enable `full`.
- Keep `gripper blocked` throughout.

Execution posture:
- Default stays `UR_ENABLE_MOTION=0`.
- Open window for one run only, then close immediately.

Ladder stages:
1) Stage A (`delta_max <= 0.005 rad`)
- Single-joint increment only.
- Must pass with:
  - `exit code=0`
  - `publish_result.returncode=0`
  - no new teach-pendant stop code
  - safety gate all green (`zero_target_ok=true`, `delta_ok=true`, `gripper_blocked=true`)

2) Stage B (`delta_max <= 0.01 rad`)
- Allowed only after Stage A pass in the same safety baseline.

## 9) RAS State-Only Bridge (No Motion Publish)

Goal:
- Reuse existing `pi0 inference` topic conventions while avoiding direct edits to the upstream `ur5e-arm-teleoperation` repository.
- Read robot state from RAS side only.

State-only command (container wrapper):
```bash
/home/ef/projects/RAS_interactivate_planner/utils/ur_state_bridge_container_wrapper.sh \
  --arm-name arm_1 \
  --joint-state-topic /arm_1/arm_joint_states \
  --timeout-sec 6
```

Expected:
- Last line JSON with `success=true`
- `data.observation.joint_state` contains 6 arm joints
- No motion/gripper commands are published by this bridge
- Still single-joint increment only.
- Same pass criteria as Stage A.

Hard stop / rollback rules:
- If any new stop code appears (for example `C157A1`), stop ladder immediately.
- Set `UR_ENABLE_MOTION=0`.
- Re-run RTDE preflight and event evidence collection before any retry.

Mandatory report fields per run:
- `exit code`
- last-line JSON (full)
- `data.execution.safety_gate`
- `publish_result` (`returncode/stdout/stderr/timeout_sec_used`)
- teach-pendant stop code status (`none` or exact code)

## 10) Owner-Switch Execution Template (Final)

When to use:
- `motion-gate` can be unstable at startup, or
- direct RTDE execution (`servoJ`) conflicts with `ur5e_robot` RTDE ownership.

Strategy:
- Optional warm-up gate (failure does not abort run).
- Mandatory owner switch during execution:
  1) stop `ur5e_robot`
  2) run one-shot `direct_servo`
  3) relaunch `ur5e_robot`
  4) re-check `motion-gate` and always run cleanup

Template (copy and run):
```bash
set -euo pipefail
CONTAINER=${UR_ROS2_CONTAINER_NAME:-exp_ef_ur5e-arm-teleopration}
ROBOT_IP=${ROBOT_IP:-192.168.24.245}
ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-131}

# 0) optional warm-up gate (non-blocking)
set +e
bash /home/ef/projects/RAS_interactivate_planner/scripts/ur_rtde_preflight.sh --check-only --motion-gate
set -e

# 1) stop current owner (ur5e_robot)
docker exec "$CONTAINER" bash -lc '
for patt in "ur5e_single_arm.launch.py" "(^|/)ur5e_robot($| )"; do
  pids=$(pgrep -f "$patt" || true)
  [ -z "$pids" ] && continue
  while read -r pid; do
    [ -z "$pid" ] && continue
    [ "$pid" = "$$" ] && continue
    [ "$pid" = "$PPID" ] && continue
    kill "$pid" >/dev/null 2>&1 || true
  done <<< "$pids"
done
sleep 1
'

# 2) one-shot direct_servo example (joint0 +0.02 rad)
docker exec "$CONTAINER" bash -lc 'python3 - <<'"'"'PY'"'"'
import json, subprocess
import rtde_receive

ip = "'"$ROBOT_IP"'"
r = rtde_receive.RTDEReceiveInterface(ip)
q = r.getActualQ()
t = q[:]
t[0] = float(t[0]) + 0.02
req = json.dumps({"action":"pick_and_place","joint_state":[float(x) for x in q],"target_joint":[float(x) for x in t]})
cmd = [
  "/home/ef/projects/RAS_interactivate_planner/utils/ur_rtde_executor_container_wrapper.sh",
  "--request-json", req,
  "--grasp-json", "{\"target_joint\":[]}",
]
res = subprocess.run(cmd, capture_output=True, text=True)
print(res.stdout.strip())
print(res.stderr.strip())
raise SystemExit(res.returncode)
PY'

# 3) bring ur5e_robot back
docker exec -d "$CONTAINER" bash -lc '
source /ros_entrypoint.sh
source /opt/ros/humble/setup.bash
[ -f /home/ef/projects/ur5e-arm-teleoperation/install/setup.bash ] && source /home/ef/projects/ur5e-arm-teleoperation/install/setup.bash
export ROS_DOMAIN_ID='"$ROS_DOMAIN_ID"'
ros2 launch ur5e_teleopration ur5e_single_arm.launch.py robot_name:=arm_1 robot_ip:='"$ROBOT_IP"' use_real_robot:=true >/tmp/ur5e_single_arm.log 2>&1
'
sleep 3

# 4) mandatory post-check + mandatory cleanup
bash /home/ef/projects/RAS_interactivate_planner/scripts/ur_rtde_preflight.sh --check-only --motion-gate
export UR_ENABLE_MOTION=0
bash /home/ef/projects/RAS_interactivate_planner/scripts/ur_rtde_preflight.sh --cleanup-only
```

Three required notes:
1) Applicable preconditions:
- `ROBOT_IP` reachable, `Remote Control=ON`, operator at E-stop.
- Use one control chain only.

2) Success criteria:
- direct executor returns `success=true`
- non-zero movement observed in telemetry (`delta_max > 0`)
- post `motion-gate` passes
- cleanup exits `0`

3) Failure rollback:
- Immediately stop further motion.
- Run:
```bash
export UR_ENABLE_MOTION=0
bash /home/ef/projects/RAS_interactivate_planner/scripts/ur_rtde_preflight.sh --cleanup-only
```
- Preserve first blocker line + final JSON for diagnosis.
