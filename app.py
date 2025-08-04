# some_file.py
import sys
# Make sure 'AiMethods' is in your system path, if necessary
sys.path.insert(1, 'AiMethods/')

from flask import Flask, render_template, request, jsonify, json
from flask_cors import CORS  # <-- New: Import CORS

from EvaluatorEnv import EvaluatorEnv
from ValueNetwork import UpdatedSingleHead

import torch

app = Flask(__name__)
CORS(app)  

# Load the model
try:
    model = UpdatedSingleHead(
        board_size=6,
        spatial_channel_indices=[0, 1, 2],
        score_channel_indices=[3, 4, 5],
        conv_out_channels=8,
    )
    model.load_state_dict(torch.load("Dropout6.pt", map_location=torch.device('cpu')))
    model.eval()
    print("AI model loaded successfully.")
except FileNotFoundError:
    print("Error: The 'Dropout6.pt' file was not found.")
    sys.exit(1) # Exit if the model can't be loaded


@app.route("/")
def index():
    """Serves the main HTML page."""
    return render_template('index.html')

# Handles solving the board game
@app.route('/solve', methods=['POST'])
def solve_board():
    data = request.json
    blockers = data.get('blockers')
    print(f"Received blockers: {blockers}")

    if not blockers:
        return jsonify({"error": "No blockers provided"}), 400

    env = EvaluatorEnv()
    env.reset(test_dice=blockers)

    try:
        while True:
            obs_batch = env._get_observation()
            if len(obs_batch) == 0:
                print("No more moves available. Solution found or no solution exists.")
                break

            obs_tensor = torch.tensor(obs_batch, dtype=torch.float32).to('cpu')
            
            with torch.no_grad():
                values = model(obs_tensor)
                best_index = torch.argmax(values).item()
            
            # --- FIX STARTS HERE ---
            step_result = env.step(best_index)
            
            # Check if step_result is None before unpacking
            if step_result is None:
                print("env.step() returned None, stopping loop.")
                break
                
            action, done, _ = step_result
            # --- FIX ENDS HERE ---
            
            if done:
                print("Solution found, stopping loop.")
                break

        solution_grid = env.visual_grid
        
        if hasattr(solution_grid, 'tolist'):
            solution_list = solution_grid.tolist()
        else:
            solution_list = solution_grid
        
        print("Final solved grid returned.")
        return jsonify({"solution": solution_list})

    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({"error": str(e), "grid_state_on_error": env.visual_grid.tolist()}), 500

if __name__ == '__main__':
    app.run(debug=True)