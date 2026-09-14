# NAVIGEN Member 4 — A* Path Planning & Dynamic Replanning

## Overview

Member 4 is responsible for **UGV path planning** using the A* algorithm.

The system finds a safe route for the UGV from a Start position to a Goal position while avoiding obstacles.

It also supports **dynamic obstacle detection and path replanning**.

---

## Completed Phases

### Phase 1 — Grid Map

- 20x20 simulated environment
- Static obstacles
- Start position
- Goal position
- Grid visualization

### Phase 2 — A* Path Planning

- A* search algorithm
- Manhattan distance heuristic
- Priority queue using Python `heapq`
- Finds the shortest safe path
- Avoids obstacles

### Phase 3 — UGV Simulation

- UGV follows the calculated A* path
- Animated movement from Start to Goal
- Displays the planned route

### Phase 4 — Dynamic Obstacle Replanning

- UGV starts travelling toward the Goal
- A new obstacle can be placed during navigation
- UGV stops when the obstacle is detected

- A* recalculates a new safe route
- UGV continues using the new route

---

## Interactive Judge Demo

cd C:\Users\Administrator\NAVIGEN_MEMBER4
venv\Scripts\activate
python click_obstacle_demo.py

