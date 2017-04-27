### Explanation
#   1. Naked Twins Function
#      To begin with, the scripts established in the class can already solve a sudoku problem.
#      The naked twins function "naked_twins" will help solve the sudoku problem faster though.
#
#   2. Diagonal Sudoku Problem
#      Just add the diagonal units as peer of each other.

### Variable initialization
# The following function is required for initializing other variables.
# So it has to be placed before variable initialization.
def cross(a, b):
    "Cross product of elements in A and elements in B."
    return [s + t for s in a for t in b]

assignments = []
rows = 'ABCDEFGHI'
cols = '123456789'
boxes = cross(rows, cols)
row_units = [cross(r, cols) for r in rows]
column_units = [cross(rows, c) for c in cols]
square_units = [cross(rs, cs) for rs in ('ABC','DEF','GHI') for cs in ('123','456','789')]
# Hard coding the diagonal units, not cool
# diagonal_units = [['A1', 'B2', 'C3', 'D4', 'E5', 'F6', 'G7', 'H8', 'I9'],
#                   ['A9', 'B8', 'C7', 'D6', 'E5', 'F4', 'G3', 'H2', 'I1']]
# Soft coding the diagonal units
diagonal_units = [[rows[i] + cols[i] for i in range(9)], [rows[::-1][i] + cols[i] for i in range(9)]]
# Diagnonal sudoku problem implemented by adding diagonal_units in the following line
unitlist = row_units + column_units + square_units + diagonal_units
units = dict((s, [u for u in unitlist if s in u]) for s in boxes)
peers = dict((s, set(sum(units[s],[]))-set([s])) for s in boxes)

### Updating the variable "values" according to the "box" and "value" provided
def assign_value(values, box, value):
    """
    Please use this function to update your values dictionary!
    Assigns a value to a given box. If it updates the board record it.
    """
    # Don't waste memory appending actions that don't actually change any values
    if values[box] == value:
        return values

    values[box] = value
    if len(value) == 1:
        assignments.append(values.copy())
    return values

### Function implementing the naked twins update strategy
def naked_twins(values):
    """Eliminate values using the naked twins strategy.
    Args:
        values(dict): a dictionary of the form {'box_name': '123456789', ...}

    Returns:
        the values dictionary with the naked twins eliminated from peers.
    """
    # print(values)
    for unit in unitlist:
        values_in_each_unit = [values[box] for box in unit]

        dupValues_all = []
        for v in values_in_each_unit:
            if values_in_each_unit.count(v) == 2 and len(v) == 2:
                dupValues_all.append(v)

        if dupValues_all != '':
            for dupValue in dupValues_all:
                for digit in dupValue:
                    for box in unit:
                        if values[box] != dupValue:
                            values = assign_value(values,box,values[box].replace(digit,''))
    return values

### Converting a grid in string form to a grid in dictionary form
def grid_values(grid):
    """
    Convert grid into a dict of {square: char} with '123456789' for empties.
    Args:
        grid(string) - A grid in string form.
    Returns:
        A grid in dictionary form
            Keys: The boxes, e.g., 'A1'
            Values: The value in each box, e.g., '8'. If the box has no value, then the value will be '123456789'.
    """
    values = []
    all_digits = '123456789'
    for c in grid:
        if c == '.':
            values.append(all_digits)
        elif c in all_digits:
            values.append(c)
    assert len(values) == 81
    return dict(zip(boxes, values))

### A utility function of printing the sudoku for visualization
def display(values):
    """
    Display the values as a 2-D grid.
    Args:
        values(dict): The sudoku in dictionary form
    """
    if values is False:
        print
    else:
        width = 1 + max(len(values[s]) for s in boxes)
        line = '+'.join(['-' * (width * 3)] * 3)
        for r in rows:
            print(''.join(values[r + c].center(width) + ('|' if c in '36' else '') for c in cols))
            if r in 'CF': print(line)
        print

### Main function of eliminiating impossible candidate (or digits) in each box
def eliminate(values):
    solved_values = [box for box in values.keys() if len(values[box]) == 1]
    for box in solved_values:
        digit = values[box]
        for peer in peers[box]:
            values[peer] = values[peer].replace(digit, '')
    return values

### Implementing only choice strategy, where only one candidate is allowed in that particular box
def only_choice(values):
    for unit in unitlist:
        digit_occurrence = {}  # record occurrence of each digit in the unit
        for digit in '123456789':
            for box in unit:
                if digit in values[box]:
                    if digit in digit_occurrence:
                        digit_occurrence[digit] += 1
                    else:
                        digit_occurrence[digit] = 1
        for digit in digit_occurrence:
            if digit_occurrence[digit] == 1:
                for box in unit:
                    if digit in values[box]:
                        assign_value(values, box, digit)
    return values

### The main function of implementing propagation of different kinds of constraints
def reduce_puzzle(values):
    stalled = False
    while not stalled:
        # Check how many boxes have a determined value
        solved_values_before = len([box for box in values.keys() if len(values[box]) == 1])
        # Use the Eliminate Strategy
        values = eliminate(values)

        # Use the Only Choice Strategy
        values = only_choice(values)

        # Use Naked Twins Strategy
        values = naked_twins(values)

        # Check how many boxes have a determined value, to compare
        solved_values_after = len([box for box in values.keys() if len(values[box]) == 1])
        # If no new values were added, stop the loop.
        stalled = solved_values_before == solved_values_after
        # Sanity check, return False if there is a box with zero available values:
        if len([box for box in values.keys() if len(values[box]) == 0]):
            return False
    return values

### Implementing the search strategy by creating a tree of possibilities,
#   travsering (i.e. going through) it using DFS until a solution is found.
#   Each node in the tree is splitted according to the possible values of
#   a particular box.
def search(values):
    "Using depth-first search and propagation, try all possible values."
    # First, reduce the puzzle using the previous function
    values = reduce_puzzle(values)
    if values is False:
        return False ## Failed earlier
    if all(len(values[s]) == 1 for s in boxes):
        return values ## Solved!
    # Choose one of the unfilled squares with the fewest possibilities
    n,s = min((len(values[s]), s) for s in boxes if len(values[s]) > 1)
    # Now use recurrence to solve each one of the resulting sudokus, and
    for value in values[s]:
        new_sudoku = values.copy()
        new_sudoku[s] = value
        attempt = search(new_sudoku)
        if attempt:
            return attempt

### Main function of initializing the solving of a sudoku from main
def solve(grid):
    """
    Find the solution to a Sudoku grid.
    Args:
        grid(string): a string representing a sudoku grid.
            Example: '2.............62....1....7...6..8...3...9...7...6..4...4....8....52.............3'
    Returns:
        The dictionary representation of the final sudoku grid. False if no solution exists.
    """
    values = grid_values(grid)
    return search(values)

if __name__ == '__main__':
    diag_sudoku_grid = '2.............62....1....7...6..8...3...9...7...6..4...4....8....52.............3'
    display(solve(diag_sudoku_grid))

    try:
        from visualize import visualize_assignments
        visualize_assignments(assignments)

    except SystemExit:
        pass
    except:
        print('We could not visualize your board due to a pygame issue. Not a problem! It is not a requirement.')
