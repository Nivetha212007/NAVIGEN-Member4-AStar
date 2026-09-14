# NAVIGEN Member 4 — Path Planning (Phase 1)

Phase 1 only: a 20x20 grid with obstacles, Start, and Goal.

A* path planning is **not** included yet. Confirm that this grid window looks correct before Phase 2.

## How to run on Windows 11

Open **Command Prompt** or **PowerShell**.

```text
cd C:\Users\Administrator\NAVIGEN_MEMBER4
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

## Expected output

- A text map in the terminal (`S` = Start, `G` = Goal, `#` = obstacle, `.` = free).
- A Matplotlib window titled **NAVIGEN – A* UGV Path Planning**.
- A screenshot saved at `outputs/phase1_grid.png`.

Close the plot window to finish the program.
