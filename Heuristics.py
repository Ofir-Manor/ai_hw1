import numpy as np

import Robot
from MazeProblem import MazeState, MazeProblem, compute_robot_direction
from Robot import UniformCostSearchRobot
from GraphSearch import NodesCollection


def tail_manhattan_heuristic(state: MazeState):
    curr_tail = state.tail
    tail_goal = state.maze_problem.tail_goal
    return (abs(tail_goal[0] - curr_tail[0]) + abs(tail_goal[1] - curr_tail[1])) * state.maze_problem.forward_cost
    # raise NotImplemented


def center_manhattan_heuristic(state: MazeState):
    center_column = (state.head[0] - state.tail[0]) / 2 + state.tail[0]
    center_raw = (state.head[1] - state.tail[1]) / 2 + state.tail[1]
    goal_center_column = (state.maze_problem.head_goal[0] - state.maze_problem.tail_goal[0]) / 2 + state.maze_problem.tail_goal[0]
    goal_center_raw = (state.maze_problem.head_goal[1] - state.maze_problem.tail_goal[1]) / 2 + state.maze_problem.tail_goal[1]
    return (abs(center_column - goal_center_column) + abs(center_raw - goal_center_raw)) * state.maze_problem.forward_cost


class ShorterRobotHeuristic:
    def __init__(self, maze_problem: MazeProblem, k):
        assert k % 2 == 0, "odd must be even"
        assert maze_problem.length - k >= 3, f"it is not possible to shorten a {maze_problem.length}-length robot by " \
                                             f"{k} units because robot length has to at least 3"
        self.k = k
        ################################################################################################################
        shorter_robot_head_goal, shorter_robot_tail_goal = self._compute_shorter_head_and_tails(maze_problem.initial_state.tail, maze_problem.initial_state.head)
        shorter_robot_head_init, shorter_robot_tail_init = self._compute_shorter_head_and_tails(maze_problem.tail_goal, maze_problem.head_goal)
        self.new_maze_problem = MazeProblem(maze_map=maze_problem.maze_map,
                                            initial_head=np.array(shorter_robot_head_init),
                                            initial_tail=np.array(shorter_robot_tail_init),
                                            head_goal=np.array(shorter_robot_head_goal),  # doesn't matter, don't change
                                            tail_goal=np.array(shorter_robot_tail_goal))  # doesn't matter, don't change
        self.node_dists = ...().solve(maze_problem=self.new_maze_problem, compute_all_dists=True)
        ################################################################################################################

        assert isinstance(self.node_dists, NodesCollection)

    def _compute_shorter_head_and_tails(self, head, tail):
        if (head[0] - tail[0]) == 0:
            if head[1] > tail[1]:
                yield [head[0], int(head[1] - (self.k/2))]
                yield [tail[0], int(tail[1] + (self.k/2))]
            else:
                yield [head[0], int(head[1] + (self.k / 2))]
                yield [tail[0], int(tail[1] - (self.k / 2))]
        else:
            if head[0] > tail[0]:
                yield [int(head[0] - (self.k/2)), head[1]]
                yield [int(tail[0] + (self.k/2)), tail[1]]
            else:
                yield [int(head[0] + (self.k/2)), head[1]]
                yield [int(tail[0] - (self.k/2)), tail[1]]
        # raise NotImplemented

    def __call__(self, state: MazeState):
        shorter_head_location, shorter_tail_location = self._compute_shorter_head_and_tails(state.tail, state.head)
        new_state = MazeState(self.new_maze_problem, head=shorter_head_location, tail=shorter_tail_location)
        if new_state in self.node_dists:
            node = self.node_dists.get_node(new_state)
            return node.g_value
        else:
            return center_manhattan_heuristic(state)