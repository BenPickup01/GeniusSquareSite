import numpy as np
from genius_square_grid import grid_space
from get_channel_metrics import get_rows_score, get_largest_empty_space, find_monospaces

class EvaluatorEnv():

    def __init__(self, dice_set="testing", early_termination=False):
        self.dice_set = dice_set
        self.game_board = grid_space()
        self.current_step = 0
        self.max_steps = 9
        self.early_termination = early_termination
        self.visual_grid = np.zeros((6, 6), dtype=np.int32)


    def reset(self, test_dice=None):
        self.game_board.reset_board(test_dice=test_dice, dice_set=self.dice_set)

        self.current_step = 0
        self.future_states = self._get_future_steps()

        # Clear visual grid and mark blockers as -1
        self.visual_grid.fill(0)
        blocker_mask = self.game_board.board == 1
        self.visual_grid[blocker_mask] = -1

        return self.future_states, len(self.future_states)


    def step(self, chosen_index):
        self.current_step += 1

        prev_board = self.game_board.board.copy()
        chosen_state = self.future_states[chosen_index]
        self.game_board.set_state(chosen_state)
        new_board = self.game_board.board

        # Detect where a new piece was placed
        new_piece_mask = (new_board != prev_board) & (prev_board == 0)
        self.visual_grid[new_piece_mask] = self.current_step  # Use step count as piece ID

        done = False
        reward = self._get_reward()

        if self.current_step == self.max_steps:
            done = True
            return [], done, 1

        if self.early_termination:
            _, mono_spaces = get_largest_empty_space(self.game_board.board)
            if mono_spaces >= 1:
                done = True
                self.current_step -= 1
                return [], done, self._get_reward()

        self.future_states = self._get_future_steps()
        if len(self.future_states) == 0:
            done = True
            return self.future_states, done, reward

        return self.future_states, done, reward


    def _get_future_steps(self):
        return self.game_board.get_future_states(self.current_step)
    
    def _get_reward(self):
        return self.current_step / self.max_steps
        # return self.current_step

    def _get_observation(self): 


        observations = []
        # We want an observation that shows for each future state
        # - The complete grid
        # - How many Rows and columns are completed
        # - The continous area score


        for state in self.future_states:

            # These first three represent the grid and how the model can interpret it
            filled_spaces = state
            empty_spaces = np.logical_not(state.astype(bool)).astype(np.float32)
            mono_spaces = find_monospaces(state)

            # The following parameters are scores to help determine how close the model is to a complete state
            available_space = np.sum(state == 0)
            largest_empty_space, _ = get_largest_empty_space(state)

            piece_score = self.current_step / self.max_steps
            line_score = get_rows_score(state)

            if largest_empty_space == 0:
                continous_space_score = 1
            else:
                continous_space_score = largest_empty_space / available_space       

            line_score_channel = np.full((6,6), line_score)
            continous_space_score_channel = np.full((6,6), continous_space_score)
            piece_score_channel = np.full((6,6), piece_score)
            # mono_spaces_score_channel = np.full((6,6), mono_spaces_score)

        

            obs_tensor = np.stack([
                filled_spaces,
                empty_spaces,
                mono_spaces,
                line_score_channel,
                continous_space_score_channel,
                piece_score_channel
            ], axis=0)


            observations.append(obs_tensor)

        return np.stack(observations, axis=0)
    
    def set_state(self, board, step):
        self.game_board = board
        self.current_step = step

    



if __name__ == "__main__":
    test_env = EvaluatorEnv()
    print(test_env.observation_board)
    test_env.reset()
    # print(test_env.step(0))
    # print(test_env._get_observation())