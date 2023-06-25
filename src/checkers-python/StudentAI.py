from collections import defaultdict
from copy import deepcopy
from math import sqrt, log
from random import randint
from time import time

from BoardClasses import Board

# The following part should be completed by students.
# Students can modify anything except the class name and exisiting functions and varibles.

class StudentAI():

    def __init__(self, col, row, p):
        self.col = col
        self.row = row
        self.p = p
        self.board = Board(col, row, p)
        self.board.initialize_game()
        self.color = ''
        self.total_time = 480
        self.remain_time = self.total_time
        self.time_denumerator = row * col * 0.5
        self.reduce = False
        self.move_count = 2
        self.opponent = {1: 2, 2: 1}
        self.color = 2
        self.reduce = False
        self.minmax = False
        self.minmax_threshold = 180
        self.setMCT()

    def setMCT(self):
        if self.col >= 8 and self.row >= 8:
            if self.p == 2:
                self.selection_iteration = 540
                self.mct = MCT(None, iteration=self.selection_iteration, max_step=800)
            elif self.p == 3:
                self.selection_iteration = 800
                self.mct = MCT(None, iteration=self.selection_iteration, max_step=960)

        else:
            self.selection_iteration = 480
            self.mct = MCT(None, iteration=self.selection_iteration, max_step=600)


    def get_move(self, move):
        start_time = time()
        if self.remain_time < self.total_time * 0.15:
            if not self.reduce:
                self.reduce = True
                self.mct.iteration = int(self.selection_iteration * 0.9)
        # Change to minmax if remaining time is less than 15 sec
        if self.remain_time < self.minmax_threshold:
            self.minmax = True

        if len(move) != 0:
            # update the board with opponents' latest move
            self.move_and_build(move, self.opponent[self.color])
            self.mct.iteration = self.selection_iteration
        else:
            # When you are the first player
            self.color = 1
            # first move move randomly - Improvement: can also move depends on MCT
            moves = self.board.get_all_possible_moves(self.color)
            index = randint(0, len(moves) - 1)
            inner_index = randint(0, len(moves[index]) - 1)
            move = moves[index][inner_index]
            self.move_and_build(move, self.color)
            return move
            # self.mct.root = Node(self.board, self.color)

        moves = self.board.get_all_possible_moves(self.color)

        # When there is only one possible move, execute it directly
        if len(moves) == 1 and len(moves[0]) == 1:
            self.move_and_build(moves[0][0], self.color)
            return moves[0][0]

        # execute only when remain time < 100
        if self.minmax:
            moves = self.board.get_all_possible_moves(self.color)
            best_move = moves[0][0]
            move = self.minMax(self.color, 4, -999999999, best_move, 999999999, best_move)[1]
            self.board.make_move(move, self.color)
            return move

        # Calculate time
        simulation_time = self.remain_time / self.time_denumerator
        best_move = self.mct.best_move(simulation_time)
        self.move_and_build(best_move, self.color)

        self.time_denumerator -= 0.5 - 1 / self.move_count
        self.move_count += 1

        self.remain_time -= time() - start_time

        with open('Testfile.txt', 'w') as f:
            print(self.remain_time, file=f)
        return best_move

    def move_and_build(self, move, color):
        self.board.make_move(move, color)
        self.mct.root = Node(self.board, self.color)

    def minMax(self, player, depth, best_score, best_move, opponent_score, opponent_move):
        if depth == 0:
            return self.heuristic_value(player), best_move
        # get all the moves of the current player
        moves = self.board.get_all_possible_moves(player)
        # Iterate through each move
        for i in moves:
            for ii in i:
                # change to new game state
                self.board.make_move(ii, player)
                if player == self.color:
                    opponent_score = \
                        self.minMax(self.opponent[self.color], depth - 1, best_score, best_move, opponent_score,
                                    opponent_move)[0]
                    if best_score < opponent_score:
                        best_score = opponent_score
                        best_move = ii
                # opponent's turn: find the best score based on player's move
                elif player == self.opponent[self.color]:
                    best_score = \
                        self.minMax(self.color, depth - 1, best_score, best_move, opponent_score, opponent_move)[0]
                    if opponent_score > best_score:
                        opponent_score = best_score
                        opponent_move = ii
                self.board.undo()
        return best_score, best_move, opponent_score, opponent_move

    def heuristic_value(self, color):
        # pawn, king, back row, mid rows, mid box, vulnerable, protected
        self_heu_value = [0, 0, 0, 0, 0, 0, 0]
        op_heu_value = [0, 0, 0, 0, 0, 0, 0]
        weights = [5, 7.75, 4, 2.5, 0.5, -3, 3]

        transfer_color = {'B': 1, 'W': 2}
        bottom_line = {1: 0, 2: self.board.col - 1}

        mid_row, mid_col = self.cal_mid()

        for row in range(self.board.row):
            for col in range(self.board.col):
                checker = self.board.board[row][col]
                if checker.color != '.':
                    # if our checker
                    if transfer_color[checker.color] == color:
                        # check for king
                        if not checker.is_king:
                            self_heu_value[0] += 1  # pawn += 1
                        else:
                            self_heu_value[1] += 1  # king += 1

                        if row == bottom_line[color]:
                            self_heu_value[2] += 1  # back row += 1
                            self_heu_value[6] += 1  # protected += 1

                        # check for mid range
                        else:
                            if row in mid_row:
                                # mid box
                                if col in mid_col:
                                    self_heu_value[3] += 1
                                else:
                                    self_heu_value[4] += 1

                            if bottom_line[color] == 0:
                                # check for vulnerability
                                if 0 < row < self.board.row - 1 and 0 < col < self.board.col - 1:
                                    if self.board.board[row - 1][col - 1].color != '.' and \
                                            transfer_color[self.board.board[row - 1][col - 1].color] == self.opponent[
                                        color] and \
                                            self.board.board[row + 1][col + 1].color == '.':
                                        self_heu_value[5] += 1
                                    if self.board.board[row - 1][col + 1].color != '.' and \
                                            transfer_color[self.board.board[row - 1][col + 1].color] == self.opponent[
                                        color] and \
                                            self.board.board[row + 1][col - 1].color == '.':
                                        self_heu_value[5] += 1
                                # check for protected
                                if 0 <= row < self.board.row - 1:
                                    if col == 0 or col == self.board.col - 1:
                                        self_heu_value[6] += 1
                                    else:
                                        if (self.board.board[row + 1][col - 1].color != '.' and
                                            (transfer_color[self.board.board[row + 1][col - 1].color] == self.color or \
                                             not self.board.board[row + 1][col - 1].is_king)) and \
                                                (self.board.board[row + 1][col + 1].color != '.' and \
                                                 (transfer_color[
                                                      self.board.board[row + 1][col + 1].color] == self.color or \
                                                  not self.board.board[row + 1][col + 1].is_king)):
                                            self_heu_value[6] += 1
                            else:
                                # check for vulnerability
                                if 0 < row < self.board.row - 1 and 0 < col < self.board.col - 1:
                                    if self.board.board[row + 1][col - 1].color != '.' and \
                                            transfer_color[self.board.board[row + 1][col - 1].color] == self.opponent[
                                        color] and \
                                            self.board.board[row - 1][col + 1].color == '.':
                                        self_heu_value[5] += 1
                                    if self.board.board[row + 1][col + 1].color != '.' and \
                                            transfer_color[self.board.board[row + 1][col + 1].color] == self.opponent[
                                        color] and \
                                            self.board.board[row - 1][col - 1].color == '.':
                                        self_heu_value[5] += 1
                                # check for protected
                                if 0 <= row < self.board.row - 1:
                                    if col == 0 or col == self.board.col - 1:
                                        self_heu_value[6] += 1
                                    else:
                                        if (self.board.board[row - 1][col - 1].color != '.' and \
                                            (transfer_color[self.board.board[row - 1][col - 1].color] == self.color or \
                                             not self.board.board[row - 1][col - 1].is_king)) and \
                                                (self.board.board[row - 1][col + 1].color != '.' and \
                                                 (transfer_color[
                                                      self.board.board[row - 1][col + 1].color] == self.color or \
                                                  not self.board.board[row - 1][col + 1].is_king)):
                                            self_heu_value[6] += 1

                    # if opponent checker
                    elif transfer_color[checker.color] == self.opponent[self.color]:
                        if not checker.is_king:
                            op_heu_value[0] += 1
                        else:
                            op_heu_value[1] += 1

                        if row == bottom_line[self.opponent[self.color]]:
                            op_heu_value[2] += 1
                            op_heu_value[6] += 1
                        else:
                            if row in mid_row:
                                # mid box
                                if col in mid_col:
                                    op_heu_value[3] += 1
                                else:
                                    op_heu_value[4] += 1

                            if bottom_line[color] == 0:
                                # check for vulnerability
                                if 0 < row < self.board.row - 1 and 0 < col < self.board.col - 1:
                                    if self.board.board[row - 1][col - 1].color != '.' and \
                                            transfer_color[self.board.board[row - 1][col - 1].color] == self.opponent[
                                        color] and \
                                            self.board.board[row + 1][col + 1].color == '.':
                                        op_heu_value[5] += 1
                                    if self.board.board[row - 1][col + 1].color != '.' and \
                                            transfer_color[self.board.board[row - 1][col + 1].color] == self.opponent[
                                        color] and \
                                            self.board.board[row + 1][col - 1].color == '.':
                                        op_heu_value[5] += 1
                                # check for protected
                                if 0 <= row < self.board.row - 1:
                                    if col == 0 or col == self.board.col - 1:
                                        op_heu_value[6] += 1
                                    else:
                                        if (self.board.board[row + 1][col - 1].color != '.' and \
                                            (transfer_color[self.board.board[row + 1][col - 1].color] == self.color or \
                                             not self.board.board[row + 1][col - 1].is_king)) and \
                                                (self.board.board[row + 1][col + 1].color != '.' and \
                                                 (transfer_color[
                                                      self.board.board[row + 1][col + 1].color] == self.color or \
                                                  not self.board.board[row + 1][col + 1].is_king)):
                                            op_heu_value[6] += 1
                            else:
                                # check for vulnerability
                                if 0 < row < self.board.row - 1 and 0 < col < self.board.col - 1:
                                    if self.board.board[row + 1][col - 1].color != '.' and \
                                            transfer_color[self.board.board[row + 1][col - 1].color] == self.opponent[
                                        color] and \
                                            self.board.board[row - 1][col + 1].color == '.':
                                        op_heu_value[5] += 1
                                    if self.board.board[row + 1][col + 1].color != '.' and \
                                            transfer_color[self.board.board[row + 1][col + 1].color] == self.opponent[
                                        color] and \
                                            self.board.board[row - 1][col - 1].color == '.':
                                        op_heu_value[5] += 1
                                # check for protected
                                if 0 <= row < self.board.row - 1:
                                    if col == 0 or col == self.board.col - 1:
                                        op_heu_value[6] += 1
                                    else:
                                        if (self.board.board[row - 1][col - 1].color != '.' and \
                                            (transfer_color[self.board.board[row - 1][col - 1].color] == self.color or \
                                             not self.board.board[row - 1][col - 1].is_king)) and \
                                                (self.board.board[row - 1][col + 1].color != '.' and \
                                                 (transfer_color[
                                                      self.board.board[row - 1][col + 1].color] == self.color or \
                                                  not self.board.board[row - 1][col + 1].is_king)):
                                            op_heu_value[6] += 1

        for i in range(7):
            self_heu_value[i] = weights[i] * (self_heu_value[i] - op_heu_value[i])
        return sum(self_heu_value)

    # helper function: calculate the mid area of game board
    def cal_mid(self):
        mid_row = []
        mid_col = []
        if self.board.row <= 7:
            if self.board.row % 2 == 1:
                mid_row.append(self.board.row // 2)
            else:
                mid_row += [self.board.row // 2 - 1, self.board.row // 2]
        else:
            if self.board.row % 2 == 1:
                half = self.board.row // 2
                if half % 2 == 1:
                    if self.board.p * 2 + half == self.board.row:
                        half -= 2
                else:
                    if self.board.p * 2 + half == self.board.row:
                        half -= 2
                    else:
                        half -= 1
                mid_point = self.board.row // 2
                mid_row.append(mid_point)
                half -= 1
                i = 1
                while half != 0:
                    mid_row += [mid_point + i, mid_point - i]
                    half -= 2
                    i += 1
            else:
                mid_point = self.board.row // 2
                mid_row += [mid_point - 1, mid_point]

        if self.board.col <= 6:
            if self.board.col % 2 == 1:
                mid_col.append(self.board.col // 2)
            else:
                mid_col += [self.board.col // 2 - 1, self.board.col // 2]
        else:
            if self.board.col % 2 == 1:
                half = self.board.col // 2
                if half % 2 == 0:
                    half += 1
                mid_point = self.board.col // 2
                mid_col.append(mid_point)
                half -= 1
                i = 1
                while half != 0:
                    mid_col += [mid_point + i, mid_point - i]
                    half -= 2
                    i += 1
            else:
                half = self.board.col // 2
                if half % 2 == 1:
                    half -= 1
                mid_point = self.board.col // 2
                mid_col.append(mid_point)
                half -= 1
                i = 1
                while half != 0:
                    if half != 1:
                        mid_col += [mid_point + i, mid_point - i]
                        i += 1
                        half -= 2
                    else:
                        mid_col.append(mid_point - i)
                        break
        return mid_row, mid_col

class MCT:
    def __init__(self, root, iteration, max_step):
        self.root = root
        self.opponent = {1: 2, 2: 1}
        self.iteration = iteration
        self.max_step = max_step

    """return a leaf node(just expanded) or a terminal node"""
    def select(self):
        current_node = self.root
        while not self.is_terminal(current_node):
            if not current_node.is_fully_expanded():
                return self.expand(current_node)
            else:
                current_node = current_node.best_child()
        return current_node

    """ 
    Make a random move. Called by simulation
    param: current leaf's board; current leaf's player color
    Improvement - Better rollout policy
    """
    def rollout_policy(self, board, color):
        moves = board.get_all_possible_moves(color)
        index = randint(0, len(moves) - 1)
        inner_index = randint(0, len(moves[index]) - 1)
        move = moves[index][inner_index]
        return move

    """
    Choose one possible move to expand at a time.
    return: a child node
    """
    def expand(self, node):
        move = node.popout()
        next_board = deepcopy(node.board)
        next_board.make_move(move, node.color)
        child_node = Node(next_board, self.opponent[node.color], parent=node, move=move)
        node.children.append(child_node)
        return child_node

    """
    return: game result (player1 win or player2 win)
    """

    def simulation(self, leaf, max_step):
        current_color = leaf.color
        current_board = deepcopy(leaf.board)
        game_result = 0
        for _ in range(max_step):
            game_result = current_board.is_win(self.opponent[current_color])
            if game_result != 0:
                break
            move = self.rollout_policy(current_board, current_color)
            current_board.make_move(move, current_color)
            # game_result = current_board.is_win(current_color)
            current_color = self.opponent[current_color]
        if game_result == 0:
            return 0
        elif game_result == self.opponent[leaf.color]:
            return 1
        else:
            return -1
        # return game_result

    """param result: the final result to backpropagate to each node above"""
    def backpropagate(self, result, leaf):
        # leaf.increment_N()  # increment the number of visited
        # leaf.increment_h_value()
        leaf.update_win(result)
        if leaf.parent:
            self.backpropagate(-result, leaf.parent)

    """
    Check if the state(node) is the end of the game
    Called by function MCT.select()
    """
    def is_terminal(self, node):
        return bool(node.board.is_win(self.opponent[node.color]))

    def best_move(self, simulation_time):
        terminate_time = time() + simulation_time
        i= 0
        while time() < terminate_time:
            i += 1
            for _ in range(self.iteration):
                leaf = self.select()  # select & expand
                result = self.simulation(leaf, self.max_step)
                self.backpropagate(result, leaf)
        if i == 0:
            return self.root.popout()
        return sorted(self.root.children, key=lambda x: x._number_of_visited, reverse=True)[0].move


class Node:
    def __init__(self, board, color, parent=None, move=None):
        self.board = deepcopy(board)
        self.color = color
        self.move = move
        self.parent = parent
        self.children = []
        self.win_rate = 0
        self.opponent = {1: 2, 2: 1}
        self._number_of_visited = 0
        self._possible_move = self.board.get_all_possible_moves(self.color)

    def popout(self):
        last_move = self._possible_move[len(self._possible_move) - 1].pop()
        # if one piece's all possible moves are popped, pop the nested move list of this piece
        if not self._possible_move[len(self._possible_move) - 1]:
            self._possible_move.pop()
        return last_move

    def update_win(self, result):
        self.increment_N()
        self.win_rate += (result - self.win_rate) / self._number_of_visited

    def increment_N(self):
        self._number_of_visited += 1

    """Getter function for _number_of_visited"""

    def get_total_visit(self):
        return self._number_of_visited

    """return: the child node with the highest UCB value"""

    def best_child(self, c=1.4):
        # UCB1 = P_Win/self_Visit + C* sqrt(logN(P_Visit)/self_Visit)
        children_UCB_value = {child: (child.win_rate) +
                                     c * sqrt(log(self.get_total_visit()) / child.get_total_visit())
                              for child in self.children}
        return sorted(children_UCB_value.items(), key=lambda x: x[1], reverse=True)[0][0]

    def is_fully_expanded(self):
        return len(self._possible_move) == 0

