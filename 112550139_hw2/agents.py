import numpy as np
import random
import game

def print_INFO():
    """
    Prints your homework submission details.
    Please replace the placeholders (date, name, student ID) with valid information
    before submitting.
    """
    print(
        """========================================
        DATE: 2025/04/01
        STUDENT NAME: 簡士原
        STUDENT ID: 112550139
        ========================================
        """)


#
# Basic search functions: Minimax and Alpha‑Beta
#

def minimax(grid, depth, maximizingPlayer, dep=4):
    """
    TODO (Part 1): Implement recursive Minimax search for Connect Four.

    Return:
      (boardValue, {setOfCandidateMoves})

    Where:
      - boardValue is the evaluated utility of the board state
      - {setOfCandidateMoves} is a set of columns that achieve this boardValue
    """
    if depth == 0 or grid.terminate():
        return get_heuristic(grid), {0}
    
    if maximizingPlayer:
        maxEval = -np.inf
        moves = set()
        for move in grid.valid:
            new_grid = game.drop_piece(grid, move)
            eval, __ = minimax(new_grid, depth - 1, False)
            # maxEval = max(maxEval, eval)
            if eval > maxEval:
                maxEval = eval
                moves = {move}
            elif eval == maxEval:
                moves.add(move)
        return maxEval, moves
    else:
        maxEval = np.inf
        moves = set()
        for move in grid.valid:
            new_grid = game.drop_piece(grid, move)
            eval, __ = minimax(new_grid, depth - 1, True)
            if eval < maxEval:
                maxEval = eval
                moves = {move}
            elif eval == maxEval:
                moves.add(move)
        return maxEval, moves

def alphabeta(grid, depth, maximizingPlayer, alpha, beta, dep=4):
    """
    TODO (Part 2): Implement Alpha-Beta pruning as an optimization to Minimax.

    Return:
      (boardValue, {setOfCandidateMoves})

    Where:
      - boardValue is the evaluated utility of the board state
      - {setOfCandidateMoves} is a set of columns that achieve this boardValue
      - Prune branches when alpha >= beta
    """
    if depth == 0 or grid.terminate():
        return get_heuristic(grid), {0}
    
    if maximizingPlayer:
        maxEval = -np.inf
        moves = set()
        for move in grid.valid:
            new_grid = game.drop_piece(grid, move)
            eval, __ = alphabeta(new_grid, depth - 1, False, alpha, beta)
            if eval > maxEval:
                maxEval = eval
                moves = {move}
            elif eval == maxEval:
                moves.add(move)
            alpha = max(alpha, maxEval)
            if beta <= alpha:
                break
        return maxEval, moves
    else:
        maxEval = np.inf
        moves = set()
        for move in grid.valid:
            new_grid = game.drop_piece(grid, move)
            eval, __ = alphabeta(new_grid, depth - 1, True, alpha, beta)
            if eval < maxEval:
                maxEval = eval
                moves = {move}
            elif eval == maxEval:
                moves.add(move)
            beta = min(beta, maxEval)
            if beta <= alpha:
                break
        return maxEval, moves

# Basic agents

def agent_minimax(grid):
    """
    Agent that uses the minimax() function with a default search depth (e.g., 4).
    Must return a single column (integer) where the piece is dropped.
    """
    return random.choice(list(minimax(grid, 4, True)[1]))


def agent_alphabeta(grid):
    """
    Agent that uses the alphabeta() function with a default search depth (e.g., 4).
    Must return a single column (integer) where the piece is dropped.
    """
    return random.choice(list(alphabeta(grid, 4, True, -np.inf, np.inf)[1]))


def agent_reflex(grid):
    """
    A simple reflex agent provided as a baseline:
      - Checks if there's an immediate winning move.
      - Otherwise picks a random valid column.
    """
    wins = [c for c in grid.valid if game.check_winning_move(grid, c, grid.mark)]
    if wins:
        return random.choice(wins)
    return random.choice(grid.valid)


def agent_strong(grid):
    """
    TODO (Part 3): Design your own agent (depth = 4) to consistently beat the Alpha-Beta agent (depth = 4).
    This agent will typically act as Player 2.
    """
    # Placeholder logic that calls your_function().
    return random.choice(list(your_function(grid, 4, False, -np.inf, np.inf)[1]))


#
# Heuristic functions
#

def get_heuristic(board):
    """
    Evaluates the board from Player 1's perspective using a basic heuristic.

    Returns:
      - Large positive value if Player 1 is winning
      - Large negative value if Player 2 is winning
      - Intermediate scores based on partial connect patterns
    """
    num_twos       = game.count_windows(board, 2, 1)
    num_threes     = game.count_windows(board, 3, 1)
    num_twos_opp   = game.count_windows(board, 2, 2)
    num_threes_opp = game.count_windows(board, 3, 2)

    score = (
          1e10 * board.win(1)
        + 1e6  * num_threes
        + 10   * num_twos
        - 10   * num_twos_opp
        - 1e6  * num_threes_opp
        - 1e10 * board.win(2)
    )
    return score


def get_heuristic_strong(board):
    """
    TODO (Part 3): Implement a more advanced board evaluation for agent_strong.
    Currently a placeholder that returns 0.
    """
    """
    Advanced board evaluation for agent_strong.
    Reference: 
      Kang, X.Y., Wang, Y.Q. and Hu, Y.R. (2019) Research on Different Heuristics for Minimax Algorithm Insight from Connect-4 Game. Journal of Intelligent Learning Systems and Applications, 11, 15-31.
    
    Methodology:
      - Immediately return infinity when one wins.
      - Count and weigh 3-connected and 2-connected chessmen.
      - Assign single pieces value based on position.
      - Penalize when opponent has immediate winning moves

    Return:
      - Large positive value if Player 1 is winning
      - Large negative value if Player 2 is winning
    """
    # 1)
    if board.win(1):
        return np.inf
    if board.win(2):
        return -np.inf

    # 2) 
    num_threes      = game.count_windows(board, 3, 1)
    num_threes_opp  = game.count_windows(board, 3, 2)
    num_twos        = game.count_windows(board, 2, 1)
    num_twos_opp    = game.count_windows(board, 2, 2)

    # 3) Single Piece
    heuristic_matrix = [
        [3,  4,   5,   12,  5,   4,   3],
        [4,  10,  13,  16,  13,  10,  4],
        [5,  13,  17,  20,  17,  13,  5],
        [5,  13,  17,  20,  17,  13,  5],
        [4,  10,  13,  16,  13,  10,  4],
        [3,  4,   5,   12,  5,   4,   3]
    ]

    score = 0
    for r in range(board.row):
        for c in range(board.column):
            piece = board.table[r][c]
            if piece == 1:
                score += heuristic_matrix[r][c]
            elif piece == 2:
                score -= heuristic_matrix[r][c]
    
    # 4) Penalize
    defensive_weight = 1e10
    immediate_threats = 0
    for move in board.valid:
        if game.check_winning_move(board, move, 1):
            immediate_threats += 1
    defense_penalty = defensive_weight * immediate_threats

    # Return
    score = (
        score  * 1e2
        + 9e6  * num_threes
        + 4e3  * num_twos
        - 4e3  * num_twos_opp
        - 9e6 * num_threes_opp
        + defense_penalty
    )
    
    return score


def your_function(grid, depth, maximizingPlayer, alpha, beta, dep=4):
    """
    A stronger search function that uses get_heuristic_strong() instead of get_heuristic().
    You can employ advanced features (e.g., improved move ordering, deeper lookahead).

    Return:
      (boardValue, {setOfCandidateMoves})
    """
    if depth == 0 or grid.terminate():
        return get_heuristic_strong(grid), set()
    
    valid_moves = list(grid.valid)
    move_values = []
    for move in valid_moves:
        new_grid = game.drop_piece(grid, move)
        h_val = get_heuristic_strong(new_grid)
        move_values.append((move, h_val))
    
    if maximizingPlayer:
        move_values.sort(key=lambda x: x[1], reverse=True)
    else:
        move_values.sort(key=lambda x: x[1])
    
    best_moves = set()
    if maximizingPlayer:
        maxEval = -np.inf
        for move, _ in move_values:
            new_grid = game.drop_piece(grid, move)
            eval, _ = your_function(new_grid, depth - 1, False, alpha, beta, dep)
            if eval > maxEval:
                maxEval = eval
                best_moves = {move}
            elif eval == maxEval:
                best_moves.add(move)
            alpha = max(alpha, maxEval)
            if beta <= alpha:
                break
        return maxEval, best_moves
    else:
        minEval = np.inf
        for move, _ in move_values:
            new_grid = game.drop_piece(grid, move)
            eval, _ = your_function(new_grid, depth - 1, True, alpha, beta, dep)
            if eval < minEval:
                minEval = eval
                best_moves = {move}
            elif eval == minEval:
                best_moves.add(move)
            beta = min(beta, minEval)
            if beta <= alpha:
                break
        return minEval, best_moves
