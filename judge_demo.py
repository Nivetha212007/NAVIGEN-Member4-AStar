"""
NAVIGEN Judge Demo
Interactive obstacle input + A* route replanning + UGV movement
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from astar import astar
from main import create_grid, START, GOAL
from simulate_ugv import simulate_ugv


def show_map(grid, start, goal, path, obstacle):
    """Show the new route after adding the judge's obstacle."""

    plt.figure(figsize=(9, 8))

    # Draw grid
    plt.imshow(
        grid,
        cmap="gray_r",
        origin="upper"
    )

    # Draw A* path
    if path:
        rows = [cell[0] for cell in path]
        cols = [cell[1] for cell in path]

        plt.plot(
            cols,
            rows,
            linewidth=3,
            marker="o",
            markersize=3
        )

    # Start
    plt.scatter(
        start[1],
        start[0],
        s=180,
        marker="s",
        label="START"
    )

    # Goal
    plt.scatter(
        goal[1],
        goal[0],
        s=180,
        marker="*",
        label="GOAL"
    )

    # New obstacle
    plt.scatter(
        obstacle[1],
        obstacle[0],
        s=200,
        marker="X",
        label="JUDGE OBSTACLE"
    )

    plt.title(
        "NAVIGEN - A* Dynamic Route Replanning"
    )

    plt.xlabel("Column")
    plt.ylabel("Row")

    plt.legend()
    plt.grid(True)

    plt.show()


def main():

    print()
    print("=" * 60)
    print("        NAVIGEN - JUDGE INTERACTIVE DEMO")
    print("=" * 60)
    print()

    print(f"Start : {START}")
    print(f"Goal  : {GOAL}")
    print()

    # Create original map
    grid = create_grid()

    # Ask judge for obstacle
    print("The judge can enter a new obstacle location.")
    print("Grid coordinates are from 0 to 19.")
    print()

    while True:

        try:

            row = int(
                input("Enter obstacle ROW (0-19): ")
            )

            col = int(
                input("Enter obstacle COLUMN (0-19): ")
            )

            obstacle = (row, col)

            # Check coordinate
            if not (0 <= row < 20 and 0 <= col < 20):
                print()
                print("Invalid coordinate!")
                print("Please enter values between 0 and 19.")
                print()
                continue

            # Do not allow obstacle on Start or Goal
            if obstacle == START:
                print()
                print("Cannot place obstacle on START.")
                print()
                continue

            if obstacle == GOAL:
                print()
                print("Cannot place obstacle on GOAL.")
                print()
                continue

            # Add obstacle
            grid[row, col] = 1

            print()
            print(
                f"New obstacle added at {obstacle}"
            )

            print()
            print("Recalculating route using A*...")
            print()

            # Run A*
            path, explored_nodes, planning_time = astar(
                grid,
                START,
                GOAL
            )

            # Check if route exists
            if not path:

                print(
                    "No route exists with this obstacle!"
                )

                # Remove obstacle
                grid[row, col] = 0

                print(
                    "Obstacle removed. Try another location."
                )

                print()
                continue

            # Successful route
            print("=" * 60)
            print("NEW ROUTE FOUND!")
            print("=" * 60)

            print(
                f"Obstacle       : {obstacle}"
            )

            print(
                f"New path length: {len(path)} cells"
            )

            print(
                f"Planning time  : {planning_time:.6f} seconds"
            )

            print()

            print("New A* route:")

            for i, cell in enumerate(path):
                print(
                    f"{i + 1:02d}. {cell}"
                )

            print()
            print("Opening new route visualization...")

            # Show new route
            show_map(
                grid,
                START,
                GOAL,
                path,
                obstacle
            )

            # Run UGV animation
            output_file = (
                Path("outputs")
                / "judge_dynamic_replanning.png"
            )

            print()
            print("Starting UGV...")
            print()

            simulate_ugv(
                grid,
                START,
                GOAL,
                path,
                explored_nodes,
                output_file,
                interval_ms=300,
                show=True
            )

            print()
            print("=" * 60)
            print("JUDGE DEMO COMPLETED")
            print("=" * 60)
            print()

            break

        except ValueError:

            print()
            print(
                "Please enter numbers only."
            )
            print()


if __name__ == "__main__":
    main()