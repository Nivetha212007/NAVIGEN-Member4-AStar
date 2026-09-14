"""
NAVIGEN Phase 3 — UGV movement simulation.

This module does not plan a path. It takes an A* path and moves a
simple UGV marker from Start to Goal, one grid cell at a time.

It is meant to run on a normal Windows Python + Matplotlib setup.
It does not use ROS, Gazebo, Docker, GPU, or real hardware.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap
from matplotlib.patches import Circle, Patch


# Same cell codes as main.py so the colors stay consistent.
FREE = 0
OBSTACLE = 1
START_CELL = 2
GOAL_CELL = 3
EXPLORED_CELL = 4
PATH_CELL = 5
TRAVELED_CELL = 6

COLORS = [
    "#F4F7FB",  # 0 free space
    "#2F3A4A",  # 1 obstacle
    "#2E8B57",  # 2 start
    "#C0392B",  # 3 goal
    "#AED6F1",  # 4 explored
    "#F4D03F",  # 5 remaining A* path
    "#E67E22",  # 6 cells the UGV has already driven
]


def _build_display(grid, start, goal, path, explored_nodes, ugv_index):
    """Build the colored grid for the current UGV step.

    ugv_index is the current position along the path (0 = start).
    Cells already visited by the UGV are painted as traveled.
    """
    display = grid.copy()

    for row, col in explored_nodes:
        if (row, col) != start and (row, col) != goal:
            display[row, col] = EXPLORED_CELL

    for row, col in path:
        if (row, col) != start and (row, col) != goal:
            display[row, col] = PATH_CELL

    # Mark the trail behind the UGV so motion is easy to see.
    if path:
        for row, col in path[: ugv_index + 1]:
            if (row, col) != start and (row, col) != goal:
                display[row, col] = TRAVELED_CELL

    rows, cols = grid.shape
    if 0 <= start[0] < rows and 0 <= start[1] < cols:
        display[start] = START_CELL
    if 0 <= goal[0] < rows and 0 <= goal[1] < cols:
        display[goal] = GOAL_CELL

    return display


def make_ugv_marker(cell):
    """Create the blue UGV circle used by Phase 3 and Phase 4.

    Matplotlib x is column, y is row.
    """
    row, col = cell
    return Circle(
        (col, row),
        radius=0.32,
        facecolor="#1F618D",
        edgecolor="white",
        linewidth=1.4,
        zorder=5,
    )


def is_interactive_backend(show):
    """Return True when a live Matplotlib window can be opened."""
    backend = plt.get_backend().lower()
    return bool(show) and "agg" not in backend


def _draw_static_map(ax, display, path, start, goal, title):
    """Draw the background grid, path line, legend, and labels."""
    cmap = ListedColormap(COLORS)
    image = ax.imshow(display, cmap=cmap, vmin=0, vmax=6, origin="upper")

    rows, cols = display.shape
    ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
    ax.grid(which="minor", color="#9AA4B2", linewidth=0.6)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_xticks(range(cols))
    ax.set_yticks(range(rows))
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.set_title(title)

    if len(path) >= 2:
        path_rows = [cell[0] for cell in path]
        path_cols = [cell[1] for cell in path]
        ax.plot(path_cols, path_rows, color="#D4A017", linewidth=2.0, zorder=3)

    legend = [
        Patch(facecolor=COLORS[0], edgecolor="#9AA4B2", label="Free space"),
        Patch(facecolor=COLORS[1], edgecolor="#9AA4B2", label="Obstacle"),
        Patch(facecolor=COLORS[4], edgecolor="#9AA4B2", label="Explored / visited"),
        Patch(facecolor=COLORS[5], edgecolor="#9AA4B2", label="A* planned path"),
        Patch(facecolor=COLORS[6], edgecolor="#9AA4B2", label="UGV traveled trail"),
        Patch(facecolor=COLORS[2], edgecolor="#9AA4B2", label=f"Start {start}"),
        Patch(facecolor=COLORS[3], edgecolor="#9AA4B2", label=f"Goal {goal}"),
        Patch(facecolor="#1F618D", edgecolor="white", label="UGV"),
    ]
    ax.legend(handles=legend, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    return image


def simulate_ugv(
    grid,
    start,
    goal,
    path,
    explored_nodes,
    save_path,
    interval_ms=180,
    show=True,
):
    """Move a UGV marker along the A* path, one cell at a time.

    Parameters
    ----------
    grid : 2D array
        Occupancy grid from Phase 1 (0 = free, 1 = obstacle).
    start, goal : tuple
        (row, col) cells used by A*.
    path : list of (row, col)
        Path returned by astar().
    explored_nodes : list of (row, col)
        Search cells returned by astar().
    save_path : Path
        Final screenshot, saved when the UGV reaches the last cell.
    interval_ms : int
        Delay between steps in the animation window.
    show : bool
        If True, open a Matplotlib window. If False (or Agg backend),
        the frames still run and the screenshot is still saved.

    Returns
    -------
    bool
        True if the UGV reached the goal along a non-empty path.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if not path:
        print("Phase 3: no A* path, so the UGV cannot move.")
        fig, ax = plt.subplots(figsize=(8, 8))
        display = _build_display(grid, start, goal, path, explored_nodes, 0)
        _draw_static_map(
            ax,
            display,
            path,
            start,
            goal,
            "NAVIGEN – A* UGV Path Planning\nPhase 3: no path to follow",
        )
        plt.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved screenshot: {save_path}")
        if show:
            plt.show()
        else:
            plt.close(fig)
        return False

    total_steps = len(path)
    print()
    print("NAVIGEN Phase 3 — UGV movement simulation")
    print(f"Path cells to follow : {total_steps}")
    print(f"Start                : {start}")
    print(f"Goal                 : {goal}")
    print("The UGV moves one grid cell per frame along the A* path.")
    print()

    fig, ax = plt.subplots(figsize=(8, 8))
    display = _build_display(grid, start, goal, path, explored_nodes, 0)
    image = _draw_static_map(
        ax,
        display,
        path,
        start,
        goal,
        "NAVIGEN – A* UGV Path Planning\nPhase 3: UGV at Start",
    )

    # Circle marker for the UGV. Matplotlib x = column, y = row.
    ugv_marker = make_ugv_marker(path[0])
    ax.add_patch(ugv_marker)
    plt.tight_layout()

    def update(frame):
        """Move the UGV to path[frame] and refresh the trail."""
        cell = path[frame]
        display = _build_display(grid, start, goal, path, explored_nodes, frame)
        image.set_data(display)
        ugv_marker.center = (cell[1], cell[0])

        step_number = frame + 1
        if frame == 0:
            status = "UGV at Start"
        elif frame == total_steps - 1:
            status = "UGV reached Goal"
        else:
            status = "UGV moving"

        ax.set_title(
            "NAVIGEN – A* UGV Path Planning\n"
            f"Phase 3: {status}  |  step {step_number}/{total_steps}  "
            f"|  cell {cell}"
        )
        print(f"UGV step {step_number:02d}/{total_steps:02d} -> {cell}")

        if frame == total_steps - 1:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved screenshot: {save_path}")

        return image, ugv_marker

    interactive = is_interactive_backend(show)

    if interactive:
        # Live window: the UGV walks the path, then the last frame is saved.
        # Keep a reference on the figure so the animation is not garbage-collected.
        fig._navigen_anim = FuncAnimation(
            fig,
            update,
            frames=total_steps,
            interval=interval_ms,
            blit=False,
            repeat=False,
            cache_frame_data=False,
        )
        plt.show()
        # If the window was closed early, still keep a final screenshot.
        if not save_path.exists():
            update(total_steps - 1)
            plt.close(fig)
    else:
        # Headless / Agg: step through every cell, then save and close.
        for frame in range(total_steps):
            update(frame)
        if not save_path.exists():
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved screenshot: {save_path}")
        plt.close(fig)

    last_cell = path[-1]
    reached_goal = last_cell == goal
    print()
    if reached_goal:
        print("Phase 3 complete: UGV reached the Goal.")
    else:
        print(f"Phase 3 finished, but the last cell {last_cell} is not the Goal {goal}.")
    return reached_goal
