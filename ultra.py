"""
============================================================
Ultrasonic Sensor Visualization + Robot Movement Animation
============================================================

WHAT THIS PROGRAM DOES (VERY SIMPLE WORDS):
-------------------------------------------
• Reads ultrasonic sensor data from a CSV file
• There are 24 ultrasonic sensors placed around a robot
• Each sensor tells how far an obstacle is
• The robot also has a movement command per frame
• The program animates:
    1) A radar-like sensor view (left)
    2) A top-down map showing robot path (right)

This helps us understand:
• What the robot "sees"
• Where obstacles are
• How the robot moves over time

This is:
✓ Visualization
✓ Sensor understanding
✓ Perception-stage logic

 
"""

# ============================================================
# LIBRARIES USED
# ============================================================

import pandas as pd
# pandas:
# Used to read and handle CSV data easily

import numpy as np
# numpy:
# Used for math calculations (angles, movement, positions)

import matplotlib.pyplot as plt
# matplotlib:
# Used for plotting graphs and animations

import matplotlib.animation as animation
# animation:
# Used to animate frames over time


# ============================================================
# LOAD DATASET
# ============================================================

# Names of 24 ultrasonic sensors
sensor_cols = [f"US{i}" for i in range(1, 25)]

# Add a column for movement class
column_names = sensor_cols + ["Class"]

# Load CSV file
data = pd.read_csv("sensor_readings_24.csv", header=None, names=column_names)

# Ensure sensor values are numeric
data[sensor_cols] = data[sensor_cols].apply(pd.to_numeric, errors="coerce")

print(f"✅ Loaded dataset with {len(data)} frames and {len(sensor_cols)} sensors.")


# ============================================================
# SENSOR ANGLE CONFIGURATION
# ============================================================

# Each ultrasonic sensor has a fixed direction (degrees)
angles_deg = np.array([
    180, -165, -150, -135, -120, -105, -90, -75,
    -60, -45, -30, -15, 0, 15, 30, 45,
    60, 75, 90, 105, 120, 135, 150, 165
])

# Convert degrees to radians (math requirement)
angles_rad = np.deg2rad(angles_deg)


# ============================================================
# MOVEMENT COLORS (FOR VISUAL CLARITY)
# ============================================================

movement_colors = {
    "Move-Forward": "green",
    "Slight-Right-Turn": "orange",
    "Sharp-Right-Turn": "red",
    "Slight-Left-Turn": "blue",
}


# ============================================================
# OBSTACLE SETTINGS
# ============================================================

# Distance below which an obstacle is considered "too close"
OBSTACLE_THRESHOLD = 30   # units depend on dataset (e.g., cm)


# ============================================================
# SIMULATE ROBOT POSITION & HEADING
# ============================================================

# Store robot positions (x, y)
positions = [(0, 0)]

# Store robot heading angles (degrees)
headings = [0]   # 0° = facing forward

# Update robot movement frame-by-frame
for cls in data["Class"]:
    x, y = positions[-1]
    heading = headings[-1]

    if cls == "Move-Forward":
        x += np.cos(np.deg2rad(heading)) * 2
        y += np.sin(np.deg2rad(heading)) * 2

    elif cls == "Slight-Right-Turn":
        heading -= 10
        x += np.cos(np.deg2rad(heading)) * 1.5
        y += np.sin(np.deg2rad(heading)) * 1.5

    elif cls == "Sharp-Right-Turn":
        heading -= 25

    elif cls == "Slight-Left-Turn":
        heading += 10
        x += np.cos(np.deg2rad(heading)) * 1.5
        y += np.sin(np.deg2rad(heading)) * 1.5

    positions.append((x, y))
    headings.append(heading)

positions = np.array(positions)
headings = np.array(headings)


# ============================================================
# CREATE FIGURE & SUBPLOTS
# ============================================================

fig = plt.figure(figsize=(14, 7))

# Left: Radar-like sensor view
ax_radar = plt.subplot(121, polar=True)

# Right: Top-down map
ax_map = plt.subplot(122)

# Radar orientation settings
ax_radar.set_theta_zero_location('S')
ax_radar.set_theta_direction(-1)


# ============================================================
# FRAME UPDATE FUNCTION (ANIMATION)
# ============================================================

def update(frame):
    ax_radar.clear()
    ax_map.clear()

    # ---------------- RADAR VIEW ----------------
    readings = data.loc[frame, sensor_cols].values

    # Close the radar loop
    readings = np.append(readings, readings[0])
    angles = np.append(angles_rad, angles_rad[0])

    move_class = data.loc[frame, "Class"]
    base_color = movement_colors.get(move_class, "gray")

    # Draw radar shape
    ax_radar.plot(angles, readings, color=base_color, linewidth=2)
    ax_radar.fill(angles, readings, color=base_color, alpha=0.15)

    ax_radar.set_rmax(np.nanmax(data[sensor_cols].values) * 1.1)
    ax_radar.grid(True, linestyle='--', alpha=0.5)
    ax_radar.set_title(f"Radar View — {move_class}", fontsize=13)

    # Highlight close obstacles
    for i in range(len(sensor_cols)):
        if readings[i] < OBSTACLE_THRESHOLD:
            ax_radar.scatter(angles[i], readings[i], color='red', s=40)

    # ---------------- MAP VIEW ----------------
    ax_map.set_title("Top-Down Robot Path")
    ax_map.set_aspect('equal', 'box')
    ax_map.grid(True, linestyle='--', alpha=0.3)

    # Draw path
    ax_map.plot(positions[:frame+1, 0], positions[:frame+1, 1], 'k--', alpha=0.6)

    # Current robot pose
    robot_x, robot_y = positions[frame]
    heading_angle = headings[frame]

    ax_map.scatter(robot_x, robot_y, color=base_color, s=80, edgecolor='black')

    # Heading arrow
    dx = np.cos(np.deg2rad(heading_angle)) * 10
    dy = np.sin(np.deg2rad(heading_angle)) * 10
    ax_map.arrow(robot_x, robot_y, dx, dy, head_width=3, head_length=5)

    # Detection boundary
    circle = plt.Circle(
        (robot_x, robot_y),
        OBSTACLE_THRESHOLD,
        color='red',
        linestyle='--',
        fill=False,
        alpha=0.4
    )
    ax_map.add_artist(circle)

    # Convert sensor readings to global coordinates
    x_local = readings[:-1] * np.cos(angles[:-1] + np.deg2rad(heading_angle))
    y_local = readings[:-1] * np.sin(angles[:-1] + np.deg2rad(heading_angle))

    ax_map.scatter(
        x_local + robot_x,
        y_local + robot_y,
        color='lightgray',
        s=20
    )

    fig.suptitle(
        f"Frame {frame+1}/{len(data)} | Movement: {move_class}",
        fontsize=14
    )


# ============================================================
# RUN ANIMATION
# ============================================================

ani = animation.FuncAnimation(
    fig,
    update,
    frames=len(data),
    interval=250,
    repeat=False
)

plt.tight_layout()
plt.show()
