# Simulation Images

This directory contains observation images for different robot states.

## Directory Structure

```
simulation_images/
├── home/
│   ├── default.jpg              # Default home view
│   ├── ac_on_22.jpg            # AC on at 22°C
│   ├── ac_on_24.jpg            # AC on at 24°C
│   ├── ac_off.jpg              # AC off
│   ├── light_on.jpg            # Light on
│   └── light_off.jpg           # Light off
├── store/
│   ├── default.jpg              # Default store view
│   ├── robot_present.jpg       # Humanoid at store
│   └── item_ready.jpg          # Item on counter
├── in_transit/
│   ├── to_store.jpg            # Navigating to store
│   └── to_home.jpg             # Returning home
└── README.md                    # This file
```

## Adding Custom Images

1. Place images in appropriate subdirectories
2. Use descriptive names matching states
3. Supported formats: jpg, jpeg, png

## Image Naming Convention

Format: `{state_key}_{state_value}.jpg`

Examples:
- `ac_on_22.jpg` → AC on at 22°C
- `light_on.jpg` → Light on
- `robot_present.jpg` → Robot at location
