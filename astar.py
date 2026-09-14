"""
NAVIGEN reusable A* path planner.

This module is meant to be imported by other NAVIGEN phases.
It does not draw plots. It only searches a 2D grid.

Grid values:
    0 = free cell
    1 = obstacle

Movement:
    4 directions only (up, down, left, right)

Heuristic:
    Manhattan distance

Cost:
    f(n) = g(n) + h(n)

Public interface:
    path, explored_nodes, planning_time = astar(grid, start, goal)
"""

import heapq
import time


# 4-direction moves: (row change, column change)
# up, down, left, right
MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]

OBSTACLE = 1


def manhattan(node, goal):
    """Return the Manhattan distance from node to goal.

    For 4-direction movement this heuristic is admissible:
    it never overestimates the remaining cost.
    """
    row, col = node
    goal_row, goal_col = goal
    return abs(row - goal_row) + abs(col - goal_col)


def in_bounds(grid, node):
    """Return True if node is inside the grid."""
    rows, cols = grid.shape
    row, col = node
    return 0 <= row < rows and 0 <= col < cols


def is_free(grid, node):
    """Return True if node is inside the grid and not an obstacle."""
    if not in_bounds(grid, node):
        return False
    return int(grid[node]) != OBSTACLE


def to_cell(position):
    """Convert a (row, col) pair into a plain Python tuple of ints."""
    if position is None:
        return None
    if not isinstance(position, (tuple, list)) or len(position) != 2:
        return None
    try:
        return (int(position[0]), int(position[1]))
    except (TypeError, ValueError):
        return None


def reconstruct_path(parents, start, goal):
    """Walk backward from goal to start using the parent dictionary.

    parents[child] = parent
    The returned path is [start, ..., goal].
    """
    path = [goal]
    current = goal
    while current != start:
        current = parents[current]
        path.append(current)
    path.reverse()
    return path


def astar(grid, start, goal):
    """Find a shortest 4-connected path from start to goal.

    Parameters
    ----------
    grid : 2D array-like
        Occupancy grid. 0 = free, 1 = obstacle.
    start : tuple
        (row, col) start cell.
    goal : tuple
        (row, col) goal cell.

    Returns
    -------
    path : list of (row, col)
        Feasible shortest path, or [] if no path exists.
    explored_nodes : list of (row, col)
        Cells expanded by A* (popped from the priority queue).
    planning_time : float
        Search time in seconds.
    """
    t0 = time.perf_counter()

    start = to_cell(start)
    goal = to_cell(goal)

    # Invalid coordinates cannot be planned.
    if start is None or goal is None:
        return [], [], time.perf_counter() - t0

    # Start or goal outside the map.
    if not in_bounds(grid, start) or not in_bounds(grid, goal):
        return [], [], time.perf_counter() - t0

    # Start or goal sitting on an obstacle.
    if not is_free(grid, start) or not is_free(grid, goal):
        return [], [], time.perf_counter() - t0

    # Trivial case: already at the goal.
    if start == goal:
        planning_time = time.perf_counter() - t0
        return [start], [start], planning_time

    # g_score[n] = cheapest known cost from start to n
    g_score = {start: 0}

    # parents[child] = parent cell used to reconstruct the path
    parents = {}

    # Min-heap of (f, tie_breaker, node)
    # tie_breaker keeps heapq from comparing cells when f values are equal
    open_heap = []
    heap_counter = 0
    heapq.heappush(open_heap, (manhattan(start, goal), heap_counter, start))

    # Cells already expanded
    closed = set()
    explored_nodes = []

    while open_heap:
        _f, _tie, current = heapq.heappop(open_heap)

        # Skip stale heap entries (a better g-score was found later).
        if current in closed:
            continue

        closed.add(current)
        explored_nodes.append(current)

        if current == goal:
            path = reconstruct_path(parents, start, goal)
            planning_time = time.perf_counter() - t0
            return path, explored_nodes, planning_time

        current_g = g_score[current]

        for d_row, d_col in MOVES:
            neighbor = (current[0] + d_row, current[1] + d_col)

            if neighbor in closed:
                continue
            if not is_free(grid, neighbor):
                continue

            # Moving one grid step costs 1.
            tentative_g = current_g + 1
            old_g = g_score.get(neighbor)

            if old_g is None or tentative_g < old_g:
                parents[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + manhattan(neighbor, goal)
                heap_counter += 1
                heapq.heappush(open_heap, (f_score, heap_counter, neighbor))

    # Open set emptied: no feasible path.
    planning_time = time.perf_counter() - t0
    return [], explored_nodes, planning_time
