
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle, Circle

from astar import astar


# ============================================================
# NAVIGEN - PHASE 4
# Dynamic Obstacle Detection + A* Replanning
# ============================================================

ROWS = 20
COLS = 20

START = (18, 1)
GOAL = (2, 18)


# ============================================================
# Create Grid
# ============================================================

def create_grid():

    grid = np.zeros((ROWS, COLS), dtype=int)

    # Static obstacles
    static_obstacles = [
        (14, 3, 14, 7),
        (10, 10, 13, 10),
        (7, 4, 7, 7),
        (4, 13, 5, 14)
    ]

    for r1, c1, r2, c2 in static_obstacles:

        for r in range(r1, r2 + 1):

            for c in range(c1, c2 + 1):

                grid[r, c] = 1

    return grid


# ============================================================
# Choose Dynamic Obstacle
# ============================================================

def choose_dynamic_obstacle(grid, path):

    if len(path) < 10:
        return None

    # Try different cells on the original path
    for index in range(5, len(path) - 5):

        candidate = path[index]

        test_grid = grid.copy()

        # Add temporary dynamic obstacle
        test_grid[candidate[0], candidate[1]] = 1

        # Stop UGV before obstacle
        new_start = path[index - 1]

        # Calculate new route
        new_path, explored, planning_time = astar(
            test_grid,
            new_start,
            GOAL
        )

        if new_path:

            return (
                candidate,
                new_start,
                new_path,
                explored,
                planning_time
            )

    return None


# ============================================================
# Draw Environment
# ============================================================

def draw_environment(
    ax,
    grid,
    original_path=None,
    replanned_path=None,
    dynamic_obstacle=None,
    current_position=None,
    explored_nodes=None
):

    ax.clear()

    # --------------------------------------------------------
    # Background
    # --------------------------------------------------------

    ax.set_facecolor("white")

    # --------------------------------------------------------
    # Grid
    # --------------------------------------------------------

    ax.set_xlim(0, COLS)
    ax.set_ylim(ROWS, 0)

    ax.set_xticks(np.arange(0, COLS + 1, 1))
    ax.set_yticks(np.arange(0, ROWS + 1, 1))

    ax.grid(
        True,
        linewidth=0.6,
        alpha=0.35
    )

    # --------------------------------------------------------
    # Static Obstacles
    # --------------------------------------------------------

    for r in range(ROWS):

        for c in range(COLS):

            if grid[r, c] == 1:

                # Dynamic obstacle is drawn separately
                if (
                    dynamic_obstacle is not None
                    and (r, c) == dynamic_obstacle
                ):
                    continue

                rectangle = Rectangle(
                    (c, r),
                    1,
                    1,
                    linewidth=0.5,
                    edgecolor="black",
                    facecolor="black"
                )

                ax.add_patch(rectangle)

    # --------------------------------------------------------
    # Explored Nodes
    # --------------------------------------------------------

    if explored_nodes:

        for node in explored_nodes:

            r, c = node

            rectangle = Rectangle(
                (c + 0.2, r + 0.2),
                0.6,
                0.6,
                linewidth=0,
                alpha=0.20
            )

            ax.add_patch(rectangle)

    # --------------------------------------------------------
    # Original Path
    # --------------------------------------------------------

    if original_path:

        x = [c + 0.5 for r, c in original_path]
        y = [r + 0.5 for r, c in original_path]

        ax.plot(
            x,
            y,
            linestyle="--",
            linewidth=2,
            label="Original A* Path"
        )

    # --------------------------------------------------------
    # Replanned Path
    # --------------------------------------------------------

    if replanned_path:

        x = [c + 0.5 for r, c in replanned_path]
        y = [r + 0.5 for r, c in replanned_path]

        ax.plot(
            x,
            y,
            linewidth=3,
            label="Replanned A* Path"
        )

    # --------------------------------------------------------
    # Start
    # --------------------------------------------------------

    start_circle = Circle(
        (START[1] + 0.5, START[0] + 0.5),
        0.35,
        facecolor="green",
        edgecolor="black",
        linewidth=2
    )

    ax.add_patch(start_circle)

    ax.text(
        START[1] + 0.5,
        START[0] + 0.5,
        "S",
        ha="center",
        va="center",
        fontweight="bold"
    )

    # --------------------------------------------------------
    # Goal
    # --------------------------------------------------------

    goal_circle = Circle(
        (GOAL[1] + 0.5, GOAL[0] + 0.5),
        0.35,
        facecolor="red",
        edgecolor="black",
        linewidth=2
    )

    ax.add_patch(goal_circle)

    ax.text(
        GOAL[1] + 0.5,
        GOAL[0] + 0.5,
        "G",
        ha="center",
        va="center",
        color="white",
        fontweight="bold"
    )

    # --------------------------------------------------------
    # Dynamic Obstacle
    # --------------------------------------------------------

    if dynamic_obstacle is not None:

        r, c = dynamic_obstacle

        rectangle = Rectangle(
            (c, r),
            1,
            1,
            linewidth=3,
            edgecolor="red",
            facecolor="orange"
        )

        ax.add_patch(rectangle)

        ax.text(
            c + 0.5,
            r + 0.5,
            "D",
            ha="center",
            va="center",
            fontweight="bold"
        )

    # --------------------------------------------------------
    # UGV
    # --------------------------------------------------------

    if current_position is not None:

        r, c = current_position

        ugv = Circle(
            (c + 0.5, r + 0.5),
            0.30,
            facecolor="blue",
            edgecolor="black",
            linewidth=2,
            zorder=10
        )

        ax.add_patch(ugv)

        ax.text(
            c + 0.5,
            r + 0.5,
            "U",
            ha="center",
            va="center",
            color="white",
            fontweight="bold",
            zorder=11
        )

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    ax.set_xlabel(
        "Grid Columns",
        fontsize=11,
        fontweight="bold"
    )

    ax.set_ylabel(
        "Grid Rows",
        fontsize=11,
        fontweight="bold"
    )

    ax.tick_params(
        labelsize=8
    )

    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        fontsize=9
    )

    ax.set_aspect("equal")


# ============================================================
# Main Phase 4
# ============================================================

def run_phase4():

    print()
    print("=" * 60)
    print("NAVIGEN - PHASE 4")
    print("Dynamic Obstacle Detection + A* Replanning")
    print("=" * 60)

    # --------------------------------------------------------
    # Create environment
    # --------------------------------------------------------

    grid = create_grid()

    print("\nCalculating initial A* path...")

    # --------------------------------------------------------
    # Initial A*
    # --------------------------------------------------------

    initial_path, initial_explored, initial_time = astar(
        grid,
        START,
        GOAL
    )

    if not initial_path:

        print("ERROR: No initial path found.")
        return

    print(
        f"Initial path found: {len(initial_path)} cells"
    )

    print(
        f"Initial planning time: "
        f"{initial_time:.4f} seconds"
    )

    # --------------------------------------------------------
    # Select Dynamic Obstacle
    # --------------------------------------------------------

    result = choose_dynamic_obstacle(
        grid,
        initial_path
    )

    if result is None:

        print("ERROR: Could not create dynamic obstacle.")
        return

    (
        dynamic_obstacle,
        stop_position,
        replanned_path,
        replanned_explored,
        replanned_time
    ) = result

    print(
        f"\nDynamic obstacle detected at: "
        f"{dynamic_obstacle}"
    )

    print(
        f"UGV stops at: {stop_position}"
    )

    print("\nRecalculating route using A*...")

    # --------------------------------------------------------
    # Add dynamic obstacle
    # --------------------------------------------------------

    grid[dynamic_obstacle[0], dynamic_obstacle[1]] = 1

    print(
        f"Replanned path found: "
        f"{len(replanned_path)} cells"
    )

    print(
        f"Replanning time: "
        f"{replanned_time:.4f} seconds"
    )

    # --------------------------------------------------------
    # Animation paths
    # --------------------------------------------------------

    path_before_obstacle = initial_path[
        :initial_path.index(dynamic_obstacle)
    ]

    # Make sure UGV stops before obstacle
    if stop_position not in path_before_obstacle:

        path_before_obstacle.append(stop_position)

    movement = (
        path_before_obstacle
        + replanned_path
    )

    # --------------------------------------------------------
    # Create figure
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(12, 8)
    )

    try:
        fig.canvas.manager.set_window_title(
            "NAVIGEN - Dynamic Obstacle Replanning"
        )
    except Exception:
        pass

    # --------------------------------------------------------
    # Animation
    # --------------------------------------------------------

    def update(frame):

        # ====================================================
        # BEFORE DYNAMIC OBSTACLE
        # ====================================================

        if frame < len(path_before_obstacle):

            position = path_before_obstacle[frame]

            draw_environment(
                ax=ax,
                grid=grid,
                original_path=initial_path,
                replanned_path=None,
                dynamic_obstacle=None,
                current_position=position,
                explored_nodes=initial_explored
            )

            ax.set_title(
                "NAVIGEN | UGV Navigation - Initial A* Path",
                fontsize=16,
                fontweight="bold",
                pad=15
            )

            ax.text(
                0.5,
                1.02,
                "UGV following the planned route",
                transform=ax.transAxes,
                ha="center",
                fontsize=10
            )

        # ====================================================
        # AFTER DYNAMIC OBSTACLE
        # ====================================================

        else:

            new_index = (
                frame -
                len(path_before_obstacle)
            )

            if new_index >= len(replanned_path):

                new_index = (
                    len(replanned_path) - 1
                )

            position = replanned_path[new_index]

            draw_environment(
                ax=ax,
                grid=grid,
                original_path=initial_path,
                replanned_path=replanned_path,
                dynamic_obstacle=dynamic_obstacle,
                current_position=position,
                explored_nodes=replanned_explored
            )

            if new_index == 0:

                ax.set_title(
                    "NAVIGEN | Dynamic Obstacle Detected - Replanning",
                    fontsize=16,
                    fontweight="bold",
                    pad=15
                )

                ax.text(
                    0.5,
                    1.02,
                    "A* recalculating a feasible route",
                    transform=ax.transAxes,
                    ha="center",
                    fontsize=10
                )

            elif (
                new_index ==
                len(replanned_path) - 1
            ):

                ax.set_title(
                    "NAVIGEN | Mission Completed - Goal Reached",
                    fontsize=16,
                    fontweight="bold",
                    pad=15
                )

                ax.text(
                    0.5,
                    1.02,
                    "UGV successfully reached destination",
                    transform=ax.transAxes,
                    ha="center",
                    fontsize=10
                )

            else:

                ax.set_title(
                    "NAVIGEN | Following Replanned A* Path",
                    fontsize=16,
                    fontweight="bold",
                    pad=15
                )

                ax.text(
                    0.5,
                    1.02,
                    "UGV avoiding dynamic obstacle",
                    transform=ax.transAxes,
                    ha="center",
                    fontsize=10
                )

    # --------------------------------------------------------
    # Create Animation
    # --------------------------------------------------------

    animation = FuncAnimation(
        fig,
        update,
        frames=len(movement),
        interval=180,
        repeat=False
    )

    # --------------------------------------------------------
    # Save Output
    # --------------------------------------------------------

    output_folder = "outputs"

    os.makedirs(
        output_folder,
        exist_ok=True
    )

    output_file = os.path.join(
        output_folder,
        "phase4_dynamic_replanning.png"
    )

    # Draw final frame
    update(len(movement) - 1)

    plt.tight_layout()

    plt.savefig(
        output_file,
        dpi=200,
        bbox_inches="tight"
    )

    print()
    print("=" * 60)
    print("PHASE 4 COMPLETE")
    print("=" * 60)

    print(
        f"Screenshot saved to:\n{output_file}"
    )

    print(
        "\nUGV successfully demonstrated:"
    )

    print("1. Initial A* path planning")
    print("2. Dynamic obstacle detection")
    print("3. UGV stopping before obstacle")
    print("4. A* path replanning")
    print("5. Reaching the final destination")

    plt.show()


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":

    run_phase4()