# Artificial Intelligence Nanodegree
## Introductory Project: Diagonal Sudoku Solver

# Question 1 (Naked Twins)
Q: How do we use constraint propagation to solve the naked twins problem? <br />
Constraint propagation is to repeatedly propagate constraints over the search space in order to
exclude forbidding values or combinations of values for some variables of a problem. In this
regard, "Naked Twins" is a search strategy in constraint propagation. In
 the following picture, "naked twin squares" are the two boxes containing "23" highlighted in
 orange. Since the values "23" in the two boxes cannot appear in their peers, these two values,
 if appear in their peers (row, column and its corresponding square), can be eliminated. In that
 case, the search space is simplified and can fasten the solving process. <br />
<img src='images/naked_twins_board.png'> <br />

# Question 2 (Diagonal Sudoku)
Q: How do we use constraint propagation to solve the diagonal sudoku problem?   <br />
To solve the diagonal sudoku problem, we can again take advantage of constraint propagation by
implementing the uniqueness of diagonal values as a constraint, similar to the "Naked Twins"
strategy. In general, the diagonal sudoku problem is just a variant of the above problem if we add
the diagonal units as the peer of each other. And doing so does not require tremendous
modificaiton of the code. <br />

### Install

This project requires **Python 3**.

We recommend students install [Anaconda](https://www.continuum.io/downloads), a pre-packaged Python distribution that contains all of the necessary libraries and software for this project. 
Please try using the environment we provided in the Anaconda lesson of the Nanodegree.

##### Optional: Pygame

Optionally, you can also install pygame if you want to see your visualization. If you've followed our instructions for setting up our conda environment, you should be all set.

If not, please see how to download pygame [here](http://www.pygame.org/download.shtml).

### Code

* `solution.py` - You'll fill this in as part of your solution.
* `solution_test.py` - Do not modify this. You can test your solution by running `python solution_test.py`.
* `PySudoku.py` - Do not modify this. This is code for visualizing your solution.
* `visualize.py` - Do not modify this. This is code for visualizing your solution.

### Visualizing

To visualize your solution, please only assign values to the values_dict using the ```assign_values``` function provided in solution.py

### Submission
Before submitting your solution to a reviewer, you are required to submit your project to Udacity's Project Assistant, which will provide some initial feedback.  

The setup is simple.  If you have not installed the client tool already, then you may do so with the command `pip install udacity-pa`.  

To submit your code to the project assistant, run `udacity submit` from within the top-level directory of this project.  You will be prompted for a username and password.  If you login using google or facebook, visit [this link](https://project-assistant.udacity.com/auth_tokens/jwt_login for alternate login instructions.

This process will create a zipfile in your top-level directory named sudoku-<id>.zip.  This is the file that you should submit to the Udacity reviews system.

