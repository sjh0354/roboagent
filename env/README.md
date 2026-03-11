# Environment Setup

This directory is split into two kinds of files:

- `*.example.sh`: checked-in templates
- `*.env.sh`: local machine config, ignored by git

Recommended setup:

```bash
cp env/common.env.example.sh env/common.env.sh
cp env/lark.env.example.sh env/lark.env.sh
cp env/planner.g1.env.example.sh env/planner.g1.env.sh
cp env/planner.ur5e.env.example.sh env/planner.ur5e.env.sh
```

`env/planner.env.sh` is a wrapper that defaults to `planner.g1.env.sh`.
