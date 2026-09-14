"""
NAVIGEN Member 4 — Phase 1 + Phase 2 + Phase 3

Phase 1: build a 20x20 grid with static obstacles, Start, and Goal.
Phase 2: run A* on that grid and show the planned path.
Phase 3: simulate a UGV moving along the A* path.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from astar import astar
from simulate_ugv import simulate_ugv

# Grid size (20 rows x 20 columns)
GRID_SIZE = 20

# Cell values used for the map and the colored plot
FREE = 0
OBSTACLE = 1
START_CELL = 2
GOAL_CELL = 3
EXPLORED_CELL = 4
PATH_CELL = 5

# Coordinates are (row, col)
# row = vertical index, 0 at the TOP
# col = horizontal index, 0 at the LEFT
START = (18, 1)
GOAL = (2, 18)


def create_grid():
    """Create a 20x20 grid with a designed obstacle layout.

    A valid path from START to GOAL exists. We do not generate
    random walls, so the map cannot accidentally become impossible.
    """
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)

    # Vertical wall on the left side, with a gap near the bottom.
    grid[0:15, 5] = OBSTACLE

    # Horizontal wall in the middle, with a gap on the right.
    grid[10, 6:16] = OBSTACLE

    # Vertical wall on the right side, with a gap near the top.
    grid[6:19, 16] = OBSTACLE

    # Extra blocks so the path must go around, not in a straight line.
    grid[15, 1:5] = OBSTACLE
    grid[3, 12:16] = OBSTACLE
    grid[7, 10:13] = OBSTACLE
    grid[12, 12:15] = OBSTACLE

    return grid


def validate_environment(grid, start, goal):
    """Check that Start and Goal are inside the grid and on free cells."""
    rows, cols = grid.shape

    for name, pos in (("Start", start), ("Goal", goal)):
        if not isinstance(pos, tuple) or len(pos) != 2:
            raise ValueError(f"{name} must be a (row, col) pair. Got: {pos}")

        row, col = pos
        if not isinstance(row, int) or not isinstance(col, int):
            raise ValueError(f"{name} coordinates must be integers. Got: {pos}")

        if row < 0 or row >= rows or col < 0 or col >= cols:
            raise ValueError(
                f"{name} {pos} is outside the {rows}x{cols} grid."
            )

        if grid[row, col] == OBSTACLE:
            raise ValueError(f"{name} {pos} is on an obstacle.")

    if start == goal:
        raise ValueError("Start and Goal cannot be the same cell.")


def print_grid(grid, start, goal):
    """Print a simple text map in the terminal."""
    print()
    print("NAVIGEN Phase 1 — text map")
    print("Legend: . free   # obstacle   S start   G goal")
    print()
    print("   " + "".join(f"{c:3d}" for c in range(GRID_SIZE)))
    for row in range(GRID_SIZE):
        line = f"{row:2d} "
        for col in range(GRID_SIZE):
            if (row, col) == start:
                symbol = "  S"
            elif (row, col) == goal:
                symbol = "  G"
            elif grid[row, col] == OBSTACLE:
                symbol = "  #"
            else:
                symbol = "  ."
            line += symbol
        print(line)
    print()


def print_path_grid(grid, start, goal, path, explored_nodes):
    """Print the A* result as a text map."""
    path_set = set(path)
    explored_set = set(explored_nodes)

    print()
    print("NAVIGEN Phase 2 — A* text map")
    print("Legend: . free   # obstacle   + explored   * path   S start   G goal")
    print()
    print("   " + "".join(f"{c:3d}" for c in range(GRID_SIZE)))
    for row in range(GRID_SIZE):
        line = f"{row:2d} "
        for col in range(GRID_SIZE):
            cell = (row, col)
            if cell == start:
                symbol = "  S"
            elif cell == goal:
                symbol = "  G"
            elif cell in path_set:
                symbol = "  *"
            elif grid[row, col] == OBSTACLE:
                symbol = "  #"
            elif cell in explored_set:
                symbol = "  +"
            else:
                symbol = "  ."
            line += symbol
        print(line)
    print()


def visualize_grid(grid, start, goal, save_path, show=True):
    """Show a judge-friendly colored grid and save a screenshot."""
    display = grid.copy()
    display[start] = START_CELL
    display[goal] = GOAL_CELL

    colors = [
        "#F4F7FB",  # free space
        "#2F3A4A",  # obstacle
        "#2E8B57",  # start (green)
        "#C0392B",  # goal (red)
    ]
    cmap = ListedColormap(colors)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(display, cmap=cmap, vmin=0, vmax=3, origin="upper")

    ax.set_xticks(np.arange(-0.5, GRID_SIZE, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, GRID_SIZE, 1), minor=True)
    ax.grid(which="minor", color="#9AA4B2", linewidth=0.6)
    ax.tick_params(which="minor", bottom=False, left=False)

    ax.set_xticks(range(GRID_SIZE))
    ax.set_yticks(range(GRID_SIZE))
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.set_title("NAVIGEN – A* UGV Path Planning\nPhase 1: Grid, Obstacles, Start and Goal")

    legend = [
        Patch(facecolor=colors[0], edgecolor="#9AA4B2", label="Free space (0)"),
        Patch(facecolor=colors[1], edgecolor="#9AA4B2", label="Obstacle (1)"),
        Patch(facecolor=colors[2], edgecolor="#9AA4B2", label=f"Start {start}"),
        Patch(facecolor=colors[3], edgecolor="#9AA4B2", label=f"Goal {goal}"),
    ]
    ax.legend(handles=legend, loc="upper left", bbox_to_anchor=(1.02, 1.0))

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved screenshot: {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def visualize_astar(grid, start, goal, path, explored_nodes, save_path, show=True):
    """Show Start, Goal, obstacles, explored cells, and the A* path."""
    display = grid.copy()

    # Paint explored cells first, then overwrite with the final path.
    for row, col in explored_nodes:
        if (row, col) != start and (row, col) != goal:
            display[row, col] = EXPLORED_CELL

    for row, col in path:
        if (row, col) != start and (row, col) != goal:
            display[row, col] = PATH_CELL

    if in_grid(grid, start):
        display[start] = START_CELL
    if in_grid(grid, goal):
        display[goal] = GOAL_CELL

    colors = [
        "#F4F7FB",  # 0 free space
        "#2F3A4A",  # 1 obstacle
        "#2E8B57",  # 2 start
        "#C0392B",  # 3 goal
        "#AED6F1",  # 4 explored
        "#F4D03F",  # 5 A* path
    ]
    cmap = ListedColormap(colors)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(display, cmap=cmap, vmin=0, vmax=5, origin="upper")

    # Draw the path as a connected line so the route is easy to follow.
    if len(path) >= 2:
        path_rows = [cell[0] for cell in path]
        path_cols = [cell[1] for cell in path]
        ax.plot(path_cols, path_rows, color="#D4A017", linewidth=2.2, zorder=3)

    ax.set_xticks(np.arange(-0.5, GRID_SIZE, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, GRID_SIZE, 1), minor=True)
    ax.grid(which="minor", color="#9AA4B2", linewidth=0.6)
    ax.tick_params(which="minor", bottom=False, left=False)

    ax.set_xticks(range(GRID_SIZE))
    ax.set_yticks(range(GRID_SIZE))
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")

    status = "path found" if path else "no path"
    ax.set_title(
        "NAVIGEN – A* UGV Path Planning\n"
        f"Phase 2: A* result ({status})"
    )

    legend = [
        Patch(facecolor=colors[0], edgecolor="#9AA4B2", label="Free space (0)"),
        Patch(facecolor=colors[1], edgecolor="#9AA4B2", label="Obstacle (1)"),
        Patch(facecolor=colors[4], edgecolor="#9AA4B2", label="Explored / visited"),
        Patch(facecolor=colors[5], edgecolor="#9AA4B2", label="A* path"),
        Patch(facecolor=colors[2], edgecolor="#9AA4B2", label=f"Start {start}"),
        Patch(facecolor=colors[3], edgecolor="#9AA4B2", label=f"Goal {goal}"),
    ]
    ax.legend(handles=legend, loc="upper left", bbox_to_anchor=(1.02, 1.0))

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved screenshot: {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def in_grid(grid, node):
    """Return True if node is a valid cell inside the grid."""
    if not isinstance(node, tuple) or len(node) != 2:
        return False
    rows, cols = grid.shape
    row, col = node
    return 0 <= row < rows and 0 <= col < cols


def print_planning_result(path, explored_nodes, planning_time, start, goal):
    """Print A* stats in the terminal."""
    print()
    print("NAVIGEN Phase 2 — A* planning result")
    print(f"Start            : {start}  (row, col)")
    print(f"Goal             : {goal}  (row, col)")
    if path:
        print("Path found       : YES")
        print(f"Path cells       : {len(path)}")
        print(f"Path cost g(n)   : {len(path) - 1} steps")
        print(f"Path             : {path}")
    else:
        print("Path found       : NO")
        print("Path cells       : 0")
        print("No feasible path from Start to Goal.")
    print(f"Explored cells   : {len(explored_nodes)}")
    print(f"Planning time    : {planning_time:.6f} seconds")
    print()


def run_phase1(grid, start, goal, output_dir, show=False):
    """Keep Phase 1 working: validate, print, and save the environment plot."""
    validate_environment(grid, start, goal)

    print("Phase 1 environment created successfully.")
    print(f"Grid size : {GRID_SIZE} x {GRID_SIZE}")
    print(f"Start     : {start}  (row, col)")
    print(f"Goal      : {goal}  (row, col)")
    print(f"Obstacles : {int(np.sum(grid == OBSTACLE))} cells")

    print_grid(grid, start, goal)

    output_file = output_dir / "phase1_grid.png"
    visualize_grid(grid, start, goal, output_file, show=show)


def run_phase2(grid, start, goal, output_dir, show=True):
    """Run reusable A* on the Phase 1 grid and save the path plot."""
    path, explored_nodes, planning_time = astar(grid, start, goal)
    print_planning_result(path, explored_nodes, planning_time, start, goal)
    print_path_grid(grid, start, goal, path, explored_nodes)

    output_file = output_dir / "phase2_astar.png"
    visualize_astar(
        grid,
        start,
        goal,
        path,
        explored_nodes,
        output_file,
        show=show,
    )
    return path, explored_nodes, planning_time


def run_phase3(grid, start, goal, path, explored_nodes, output_dir, show=True):
    """Simulate the UGV walking the A* path and save the final screenshot."""
    output_file = output_dir / "phase3_ugv_simulation.png"
    reached_goal = simulate_ugv(
        grid=grid,
        start=start,
        goal=goal,
        path=path,
        explored_nodes=explored_nodes,
        save_path=output_file,
        show=show,
    )
    return reached_goal


def main():
    grid = create_grid()
    output_dir = Path(__file__).resolve().parent / "outputs"

    # Phase 1 and Phase 2 still run and save their screenshots.
    # The interactive window is reserved for the Phase 3 UGV animation.
    run_phase1(grid, START, GOAL, output_dir, show=False)
    path, explored_nodes, _planning_time = run_phase2(
        grid, START, GOAL, output_dir, show=False
    )
    run_phase3(
        grid,
        START,
        GOAL,
        path,
        explored_nodes,
        output_dir,
        show=True,
    )


if __name__ == "__main__":
    main()
