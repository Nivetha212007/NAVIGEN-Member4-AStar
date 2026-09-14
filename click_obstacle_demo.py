import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

from astar import astar


# =========================================================
# START AND GOAL
# =========================================================

START = (18, 1)
GOAL = (2, 18)


# =========================================================
# CREATE GRID
# =========================================================

def create_grid():

    grid = np.zeros((20, 20), dtype=int)

    # Static obstacles
    grid[5:15, 8] = 1
    grid[3, 3:12] = 1
    grid[15, 10:18] = 1
    grid[10:17, 15] = 1

    # Make sure START and GOAL are free
    grid[START] = 0
    grid[GOAL] = 0

    return grid


# =========================================================
# SETTINGS
# =========================================================

# UGV moves one cell every 0.75 seconds
MOVE_TIME = 750

grid = create_grid()


# =========================================================
# INITIAL A* PATH
# =========================================================

current_path, explored_nodes, planning_time = astar(
    grid,
    START,
    GOAL
)

if not current_path:

    print("No initial path found!")
    exit()


# =========================================================
# UGV VARIABLES
# =========================================================

current_position = START
current_index = 0

paused_for_obstacle = False
navigation_finished = False


# =========================================================
# CREATE FIGURE
# =========================================================

fig, ax = plt.subplots(figsize=(9, 9))


# =========================================================
# DRAW MAP
# =========================================================

def draw_map(title_text=None):

    ax.clear()

    # Draw grid
    ax.imshow(
        grid,
        cmap="Greys",
        origin="upper"
    )

    # -----------------------------------------------------
    # Draw A* route
    # -----------------------------------------------------

    if current_path:

        rows = [point[0] for point in current_path]
        cols = [point[1] for point in current_path]

        ax.plot(
            cols,
            rows,
            linewidth=3,
            marker="o",
            markersize=4,
            label="A* Route"
        )

    # -----------------------------------------------------
    # START
    # -----------------------------------------------------

    ax.scatter(
        START[1],
        START[0],
        s=180,
        marker="s",
        label="START"
    )

    # -----------------------------------------------------
    # GOAL
    # -----------------------------------------------------

    ax.scatter(
        GOAL[1],
        GOAL[0],
        s=220,
        marker="*",
        label="GOAL"
    )

    # -----------------------------------------------------
    # UGV
    # -----------------------------------------------------

    ax.scatter(
        current_position[1],
        current_position[0],
        s=250,
        marker="o",
        label="UGV"
    )

    # -----------------------------------------------------
    # GRID
    # -----------------------------------------------------

    ax.set_xticks(range(grid.shape[1]))
    ax.set_yticks(range(grid.shape[0]))

    ax.grid(True)

    # -----------------------------------------------------
    # TITLE
    # -----------------------------------------------------

    if title_text:

        ax.set_title(
            title_text,
            fontsize=14
        )

    else:

        ax.set_title(
            f"NAVIGEN - UGV Position: {current_position}\n"
            "CLICK A FREE CELL TO ADD DYNAMIC OBSTACLE",
            fontsize=14
        )

    ax.legend(loc="upper left")


# Draw initial map
draw_map()


# =========================================================
# MOUSE CLICK = DYNAMIC OBSTACLE
# =========================================================

def on_click(event):

    global grid
    global current_path
    global current_position
    global current_index
    global paused_for_obstacle
    global navigation_finished

    # -----------------------------------------------------
    # Click must be inside map
    # -----------------------------------------------------

    if event.inaxes != ax:
        return

    if event.xdata is None or event.ydata is None:
        return

    # -----------------------------------------------------
    # Navigation already finished
    # -----------------------------------------------------

    if navigation_finished:
        return

    # -----------------------------------------------------
    # Convert mouse click to grid cell
    # -----------------------------------------------------

    row = int(round(event.ydata))
    col = int(round(event.xdata))

    clicked_cell = (row, col)

    # -----------------------------------------------------
    # Check boundaries
    # -----------------------------------------------------

    if not (
        0 <= row < grid.shape[0]
        and 0 <= col < grid.shape[1]
    ):

        return

    # -----------------------------------------------------
    # Cannot place obstacle on UGV
    # -----------------------------------------------------

    if clicked_cell == current_position:

        print()
        print("Cannot place obstacle on UGV!")
        return

    # -----------------------------------------------------
    # Cannot place obstacle on GOAL
    # -----------------------------------------------------

    if clicked_cell == GOAL:

        print()
        print("Cannot place obstacle on GOAL!")
        return

    # -----------------------------------------------------
    # Already obstacle
    # -----------------------------------------------------

    if grid[row, col] == 1:

        print()
        print("That cell is already an obstacle!")
        return

    # =====================================================
    # STOP UGV
    # =====================================================

    paused_for_obstacle = True

    print()
    print("==========================================")
    print("       DYNAMIC OBSTACLE DETECTED!")
    print("==========================================")

    print(
        "Obstacle location   :",
        clicked_cell
    )

    print(
        "UGV current position:",
        current_position
    )

    print("UGV STATUS          : STOPPED")

    print("==========================================")

    # =====================================================
    # ADD OBSTACLE
    # =====================================================

    grid[row, col] = 1

    print()
    print("New obstacle added.")
    print("Running A* replanning...")

    # =====================================================
    # REPLAN FROM CURRENT UGV POSITION
    # =====================================================

    new_path, new_explored, new_time = astar(
        grid,
        current_position,
        GOAL
    )

    # =====================================================
    # NO SAFE ROUTE
    # =====================================================

    if not new_path:

        print()
        print("==========================================")
        print("       NO SAFE ROUTE FOUND!")
        print("       UGV REMAINS STOPPED")
        print("==========================================")

        draw_map(
            "NO SAFE ROUTE FOUND - UGV STOPPED"
        )

        fig.canvas.draw_idle()

        return

    # =====================================================
    # USE NEW ROUTE
    # =====================================================

    current_path = new_path

    current_index = 0

    print()
    print("NEW ROUTE FOUND!")

    print(
        "New route length:",
        len(new_path)
    )

    print(
        "Planning time:",
        round(new_time, 4),
        "seconds"
    )

    print()
    print("UGV CONTINUING ON NEW ROUTE!")

    print("==========================================")
    print()

    # =====================================================
    # RESUME UGV
    # =====================================================

    paused_for_obstacle = False

    draw_map(
        f"REPLANNED - UGV Position: {current_position}"
    )

    fig.canvas.draw_idle()


# Connect mouse click
fig.canvas.mpl_connect(
    "button_press_event",
    on_click
)


# =========================================================
# UGV MOVEMENT
# =========================================================

def update(frame):

    global current_position
    global current_index
    global navigation_finished

    # -----------------------------------------------------
    # Already finished
    # -----------------------------------------------------

    if navigation_finished:
        return

    # -----------------------------------------------------
    # Wait during replanning
    # -----------------------------------------------------

    if paused_for_obstacle:
        return

    # -----------------------------------------------------
    # No path
    # -----------------------------------------------------

    if not current_path:
        return

    # =====================================================
    # MOVE ONE CELL
    # =====================================================

    if current_index < len(current_path):

        current_position = current_path[current_index]

        print(
            "UGV moving to:",
            current_position
        )

        current_index += 1

        draw_map()

        fig.canvas.draw_idle()

    # =====================================================
    # REACHED GOAL
    # =====================================================

    else:

        if current_position == GOAL:

            navigation_finished = True

            print()
            print("==========================================")
            print("          NAVIGATION FINISHED!")
            print("==========================================")

            print(
                "UGV reached destination:",
                current_position
            )

            print("==========================================")

            draw_map(
                "NAVIGATION FINISHED - UGV REACHED GOAL"
            )

            fig.canvas.draw_idle()


# =========================================================
# START ANIMATION
# =========================================================

ani = animation.FuncAnimation(
    fig,
    update,
    interval=MOVE_TIME,
    cache_frame_data=False
)


# =========================================================
# SHOW WINDOW
# =========================================================

plt.show()