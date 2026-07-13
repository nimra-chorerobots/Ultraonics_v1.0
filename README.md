# Ultrasonic Perception Visualization for Mobile Robots

### Interactive Ultrasonic Sensor Perception and Spatial Awareness Visualization Toolkit

This repository presents an interactive ultrasonic perception visualization toolkit for mobile robots. It demonstrates how **24 ultrasonic sensors** collectively perceive nearby obstacles and transform raw distance measurements into meaningful spatial awareness around the robot.

Unlike high-resolution sensors such as LiDAR, ultrasonic sensors provide lightweight, short-range distance measurements that are widely used for obstacle avoidance, collision prevention, and robot safety. This project focuses on **interactive qualitative perception analysis**, helping users understand ultrasonic sensing, robot motion, and obstacle localization through animated visualizations.

---

# 📌 Overview

Ultrasonic sensors are among the most common perception sensors used in autonomous robots due to their low cost, simplicity, and reliability for short-range obstacle detection.

This project demonstrates how raw ultrasonic measurements are converted into spatial information through:

- Real-time visualization of 24 ultrasonic sensors
- Coordinate transformation from polar space to world coordinates
- Robot trajectory visualization
- Robot heading estimation
- Obstacle detection using configurable distance thresholds
- Animated perception over time

The repository is intended for:

- Mobile robotics
- Ultrasonic perception research
- Robotics education
- Obstacle avoidance research
- Sensor fusion development
- Physical AI perception pipelines

---

# 🚀 Features

- Real-time visualization of 24 ultrasonic sensors
- Polar ultrasonic sensor display
- Top-down world visualization
- Robot heading visualization
- Robot trajectory visualization
- Obstacle detection visualization
- Configurable detection threshold
- Animated perception playback
- Coordinate transformation
- Interactive perception analysis

---

# 📂 Input Dataset

The input dataset consists of a CSV file containing synchronized ultrasonic sensor measurements.

Example format:

```text
dataset.csv
│
├── US1
├── US2
├── ...
├── US24
└── Class
```

Each frame contains:

- 24 ultrasonic sensor readings
- Robot movement label

Example movement classes:

- `Move-Forward`
- `Move-Backward`
- `Slight-Left-Turn`
- `Slight-Right-Turn`
- `Sharp-Left-Turn`
- `Sharp-Right-Turn`

---

# 🔄 Processing Pipeline

```text
CSV Sensor Dataset
        │
        ▼
Sensor Reading Extraction
        │
        ▼
Distance Processing
        │
        ▼
Coordinate Transformation
        │
        ▼
Obstacle Detection
        │
        ▼
Polar Visualization
        │
        ▼
Top-Down World Visualization
```

---

# ⚙️ How It Works

## 1️⃣ Sensor Reading Extraction

Each frame contains distance measurements collected from 24 ultrasonic sensors positioned around the robot.

---

## 2️⃣ Polar Coordinate Representation

Sensor readings are displayed in a polar plot.

Each spike corresponds to one ultrasonic sensor:

- Angle → Physical sensor location
- Radius → Measured obstacle distance

This provides an intuitive understanding of the robot's local surroundings.

---

## 3️⃣ Coordinate Transformation

Distance measurements are transformed from polar coordinates into global world coordinates.

This enables obstacle positions to be projected around the robot within a top-down map.

---

## 4️⃣ Robot Motion Update

The robot heading and movement are continuously updated based on the recorded motion labels.

The trajectory history is preserved to illustrate robot movement over time.

---

## 5️⃣ Interactive Visualization

The visualization updates continuously throughout the dataset, providing an animated representation of the robot's perception process.

The animation supports:

- Continuous playback
- Real-time obstacle visualization
- Robot heading updates
- Motion trajectory visualization
- Spatial awareness debugging

---

# 🧠 Why Ultrasonic Sensors?

Ultrasonic sensors are:

- Low-cost
- Energy efficient
- Short-range
- Robust for nearby obstacle detection
- Widely used for collision avoidance and robot safety

Unlike LiDAR sensors, ultrasonic sensors produce sparse distance measurements rather than dense point clouds. Their purpose is obstacle awareness rather than detailed environment reconstruction.

---

# 🖥 Interactive Visualization

Unlike static perception projects, this repository generates an animated visualization during execution.

The dashboard includes two synchronized views.

## 1️⃣ Polar Sensor View (Left)

- One spike per ultrasonic sensor (US1–US24)
- Sensor angles match physical placement
- Distance represents obstacle proximity
- Near obstacles highlighted in red

---

## 2️⃣ Top-Down World View (Right)

- Robot position
- Robot heading
- Motion trajectory
- Projected obstacle positions
- Detection boundary
- World-coordinate visualization

The animation allows users to inspect perception continuously instead of relying on static screenshots.

---

# 🖼 Example Visualization

The following image illustrates a representative frame generated by the visualization pipeline.

- **Left:** Polar ultrasonic perception
- **Right:** Top-down world visualization with robot heading and detected obstacles

> **Note:** The full visualization is animated and can be explored by running the project locally.

<img width="1400" height="700" alt="Ultrasonic Perception Result" src="https://github.com/user-attachments/assets/d80bfe7a-ffc4-41da-a751-fed501bd329a" />

---

# 🏗 Architecture

```text
Ultrasonic Sensors
        │
        ▼
Distance Measurements
        │
        ▼
Coordinate Transformation
        │
        ▼
Obstacle Detection
        │
        ▼
Robot Motion Update
        │
        ▼
Interactive Visualization
```

---

# 🚀 Project Status

🟢 **Prototype**

### Current Features

- 24 Ultrasonic Sensors
- Polar Sensor Visualization
- Top-Down World Visualization
- Robot Heading
- Motion Trajectory
- Obstacle Detection
- Animated Visualization

### Planned Improvements

- ROS 2 integration
- NVIDIA Isaac Sim integration
- Camera-Ultrasonic Fusion
- LiDAR-Ultrasonic Fusion
- Dynamic Obstacle Tracking
- Navigation Safety Layer
- Multi-Sensor Fusion

---

# 📂 Repository Structure

```text
chore-ultrasonic-perception/
│
├── src/
│   └── ultrasonic_visualization.py
│
├── assets/
│   ├── input/
│   ├── output/
│   └── examples/
│
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
├── CHANGELOG.md
└── CITATION.cff
```

---

# ▶️ Installation

Clone the repository:

```bash
git clone https://github.com/nimra-chorerobots/chore-ultrasonic-perception.git

cd chore-ultrasonic-perception
```

Install dependencies:

```bash
pip install -r requirements.txt
```

or manually:

```bash
pip install numpy pandas matplotlib
```

---

# 📦 Requirements

- Python 3.9+
- NumPy
- Pandas
- Matplotlib

Example `requirements.txt`

```text
numpy
pandas
matplotlib
```

---

# ▶️ Running the Project

Run the visualization pipeline:

```bash
python src/ultrasonic_visualization.py
```

Update the dataset path inside the script before execution.

The animation will automatically display synchronized polar and top-down perception views throughout the recorded robot motion.

---

# 💡 Applications

This visualization toolkit can be used for:

- Mobile robotics
- Obstacle avoidance
- Collision prevention
- Sensor fusion research
- Physical AI
- Robotics education
- Perception debugging
- Digital twin visualization
- Autonomous robot development

---

# 🔮 Future Work

Future versions of this repository will include:

- ROS 2 integration
- NVIDIA Isaac Sim support
- Multi-sensor fusion
- Camera integration
- LiDAR integration
- Dynamic obstacle tracking
- Navigation planning
- Occupancy mapping
- Real-time robot deployment

 
