import torch
from EvaluatorEnv import EvaluatorEnv
from ValueNetwork import UpdatedSingleHead

# === CONFIG ===
GRID_SIZE = 6


def run_headless_loop(env, model):
    """
    Runs the game loop without a graphical interface, printing the grid
    to the console after each step.
    """
    steps = 0
    done = False
    
    # Initialize the environment and print the starting grid
    env.reset()


    while not done:
        # Get all possible moves from the current state
        obs_batch = env._get_observation()
        if len(obs_batch) == 0:
            print("\nNo more valid moves. Game over.")
            done = True
            break

        # Convert observations to a tensor for the model
        obs_tensor = torch.tensor(obs_batch, dtype=torch.float32).to('cpu')
        
        with torch.no_grad():
            # Get the model's predicted values for each move
            values = model(obs_tensor)
            # Choose the move with the highest value
            best_index = torch.argmax(values).item()

        # Take the best step and get the new state
        _, done, _ = env.step(best_index)
        steps += 1
        

    print(env.visual_grid)


if __name__ == "__main__":
    # Load the model
    model = UpdatedSingleHead(
        board_size=6,
        spatial_channel_indices=[0, 1, 2],
        score_channel_indices=[3, 4, 5],
        conv_out_channels=8,
    )
    model.load_state_dict(torch.load("Ai Methods/genius Square /Dropout6.pt", map_location=torch.device('cpu')))
    model.eval()
    
    # Create the environment
    env = EvaluatorEnv()
    
    # Run the game loop
    run_headless_loop(env, model)