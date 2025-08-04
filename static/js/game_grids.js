//==================================================
//  CONSTANTS & CORE GAME STATE
//==================================================

const GRID_SIZE = 6;
const gridContainer = document.getElementById('grid-container');

// AI Grid
const AI_GRID_SIZE = 6;
const AI_gridContainer = document.getElementById('ai-grid-container');

// Data structure used to store the grid
let grid_representation = Array.from({ length: GRID_SIZE }, () => Array(GRID_SIZE).fill(0));
let AI_grid_representation = Array.from({ length: AI_GRID_SIZE }, () => Array(AI_GRID_SIZE).fill(0));

// Blocker positions
const DICE = [
  [[0, 0], [2, 0], [3, 0], [3, 1], [4, 1], [5, 2]], // Dice 1 
  [[0, 1], [1, 1], [2, 1], [0, 2], [1, 0], [1, 2]], // Dice 2
  [[2, 2], [3, 2], [4, 2], [1, 3], [2, 3], [3, 3]], // Dice 3
  [[4, 0], [5, 1], [1, 5], [0, 4]],                 // Dice 4
  [[0, 3], [1, 4], [2, 5], [2, 4], [3, 5], [5, 5]], // Dice 5
  [[4, 3], [5, 3], [4, 4], [5, 4], [3, 4], [4, 5]], // Dice 6
  [[5, 0], [0, 5]]                                  // Dice 7  
];

// Piece definitions
const PIECES = [
  [[1, 1], [1, 1]],         // 2x2 Square
  [[1, 1, 1, 1]],           // 1x4 Line
  [[1, 1, 1], [0, 1, 0]],   // T Piece
  [[1, 0], [1, 1], [0, 1]], // Z Piece
  [[1, 0], [1, 0], [1, 1]], // L Piece
  [[1, 0], [1, 1]],         // Mini L Piece
  [[1, 1, 1]],              // 1x3 Line
  [[1, 1]],                 // 2x1 Line
  [[1]]                     // 1x1 Square
];

// Piece and color state
let chosen_piece_index = 0;
let available_pieces = [...PIECES];
let current_piece = available_pieces[chosen_piece_index];
let current_piece_original_index = chosen_piece_index;
let hoveredCell = null;
let highlightedCells = [];

// Color definitions
const BACKGROUND_COLOR = { r: 240, g: 240, b: 240, a: 1 };
const BLOCKER_COLOR_KEY = -1;



const PIECE_COLOUR_MAP = new Map([
  [1, { r: 0,   g: 192, b: 0,   a: 1 }],  // 2x2 Square: Green
  [2, { r: 96,  g: 96,  b: 96,  a: 1 }],  // 1x4 Line: Grey
  [3, { r: 255, g: 255, b: 64,  a: 1 }],  // T Piece: Yellow
  [4, { r: 255, g: 64,  b: 64,  a: 1 }],  // Z Piece: Red
  [5, { r: 173, g: 216, b: 250, a: 1 }],  // L Piece: Light Blue
  [6, { r: 195, g: 72,  b: 72,  a: 1 }],  // Mini L Piece: Brown
  [7, { r: 192, g: 0,   b: 192, a: 1 }],  // 1x3 Line: Purple
  [8, { r: 255, g: 192, b: 64,  a: 1 }],  // 2x1 Line: Orange
  [9, { r: 32,  g: 32,  b: 170, a: 1 }],  // 1x1 Square: Dark Blue
]);

//==================================================
//  CORE GAME LOGIC
//==================================================

function rotatePiece(piece, direction = 'clockwise') {
  if (!piece || piece.length === 0) {
    return [];
  }
  const transposed = piece[0].map((_, colIndex) => piece.map(row => row[colIndex]));
  if (direction === 'clockwise') {
    return transposed.map(row => row.reverse());
  } else if (direction === 'anti-clockwise') {
    return transposed.reverse();
  } else {
    return piece;
  }
}

function flipPiece(piece) {
  if (!piece || piece.length === 0) {
    return [];
  }
  const flippedPiece = [];
  for (let i = 0; i < piece.length; i++) {
    flippedPiece.push([...piece[i]].reverse());
  }
  return flippedPiece;
}

function isValidPlacement(startX, startY) {
  if (!current_piece) return false;
  
  const piece = current_piece;
  const pieceHeight = piece.length;
  const pieceWidth = piece[0].length;

  for (let y = 0; y < pieceHeight; y++) {
    for (let x = 0; x < pieceWidth; x++) {
      if (piece[y][x] === 1) {
        const gridX = startX + x;
        const gridY = startY + y;
        if (gridX < 0 || gridX >= GRID_SIZE || gridY < 0 || gridY >= GRID_SIZE) {
          return false;
        }
        if (grid_representation[gridY][gridX] !== 0) {
          return false;
        }
      }
    }
  }
  return true;
}

//==================================================
//  GRID MANAGEMENT & RENDERING
//==================================================

function renderGrid() {
  for (let y = 0; y < GRID_SIZE; y++) {
    for (let x = 0; x < GRID_SIZE; x++) {
      const cell = getCell(x, y);
      if (cell) {
        let colorKey = grid_representation[y][x];
        let colorObject;
        if (colorKey === BLOCKER_COLOR_KEY || colorKey === 0) {
          colorObject = BACKGROUND_COLOR;
        } else {
          colorObject = PIECE_COLOUR_MAP.get(colorKey);
        }
        cell.style.backgroundColor = toRGBAString(colorObject);
      }
    }
  }
}

function renderAIGrid() {
  for (let y = 0; y < AI_GRID_SIZE; y++) {
    for (let x = 0; x < AI_GRID_SIZE; x++) {
      const cell = AIgetCell(x, y);
      if (cell) {
        let colorKey = AI_grid_representation[y][x];
        let colorObject;
        if (colorKey === BLOCKER_COLOR_KEY || colorKey === 0) {
          colorObject = BACKGROUND_COLOR;
        } else {
          colorObject = PIECE_COLOUR_MAP.get(colorKey);
        }
        cell.style.backgroundColor = toRGBAString(colorObject);
      }
    }
  }
}

function createGrid(container, isAI = false) {
  for (let y = 0; y < GRID_SIZE; y++) {
    for (let x = 0; x < GRID_SIZE; x++) {
      const cell = document.createElement('div');
      cell.classList.add('grid-cell');
      cell.dataset.x = x; 
      cell.dataset.y = y;
      
      if (!isAI) {
        cell.addEventListener('mouseover', handleMouseover);
        cell.addEventListener('mouseout', handleMouseout);
        cell.addEventListener('click', handlePlacement);
      }
      
      container.appendChild(cell);
    }
  }
}

function resetGrid() {
  removeBlockers();
  grid_representation = Array.from({ length: GRID_SIZE }, () => Array(GRID_SIZE).fill(0));
  AI_grid_representation = Array.from({ length: AI_GRID_SIZE }, () => Array(AI_GRID_SIZE).fill(0));
  available_pieces = [...PIECES];
  chosen_piece_index = 0;
  if (available_pieces.length > 0) {
      current_piece = available_pieces[chosen_piece_index];
      current_piece_original_index = PIECES.indexOf(current_piece);
  } else {
      current_piece = null;
      current_piece_original_index = -1;
  }
  placeRandomBlockers();
  renderGrid();
  renderAIGrid();
}

function getCell(x, y) {
  return document.querySelector(`#grid-container .grid-cell[data-x="${x}"][data-y="${y}"]`);
}

function AIgetCell(x, y) {
  return document.querySelector(`#ai-grid-container .grid-cell[data-x="${x}"][data-y="${y}"]`);
}

//==================================================
//  BLOCKER LOGIC
//==================================================

function removeBlockers() {
  const allCircles = document.querySelectorAll('.circle');
  allCircles.forEach(circle => {
    circle.remove();
  });
}

function placeBlocker(x, y) {
  const playerCell = getCell(x, y);
  const aiCell = AIgetCell(x, y);
  
  // Place blocker on the player's grid
  let playerCircle = document.createElement('div');
  playerCircle.classList.add('circle');
  playerCell.appendChild(playerCircle);
  grid_representation[y][x] = BLOCKER_COLOR_KEY;

  // Place blocker on the AI's grid
  let aiCircle = document.createElement('div');
  aiCircle.classList.add('circle');
  aiCell.appendChild(aiCircle);
  AI_grid_representation[y][x] = BLOCKER_COLOR_KEY;
}

function placeRandomBlockers() {
  let random_indexes = DICE.map(die => die[Math.floor(Math.random() * die.length)]);
  random_indexes.forEach(position => {
    placeBlocker(position[0], position[1]);
  });
}

//==================================================
//  PIECE INTERACTION (HOVER/PLACEMENT)
//==================================================

function highlightHoverShape(startX, startY) {
  if (!current_piece) return;

  clearHoverShape();
  
  const piece = current_piece;
  const pieceHeight = piece.length;
  const pieceWidth = piece[0].length;
  
  const hoverColor = getOpaqueColor(current_piece_original_index + 1, 0.8);
  for (let y = 0; y < pieceHeight; y++) {
    for (let x = 0; x < pieceWidth; x++) {
      if (piece[y][x] === 1) {
        const cell = getCell(startX + x, startY + y);
        if (cell) {
          cell.style.backgroundColor = hoverColor;
          highlightedCells.push(cell);
        }
      }
    }
  }
}

function clearHoverShape() {
  highlightedCells.forEach(cell => {
    const x = parseInt(cell.dataset.x);
    const y = parseInt(cell.dataset.y); 

    let originalColorObject;
    let colorKey = grid_representation[y][x];

    if (colorKey === BLOCKER_COLOR_KEY || colorKey === 0) {
      originalColorObject = BACKGROUND_COLOR;
    } else {
      originalColorObject = PIECE_COLOUR_MAP.get(colorKey);
    }

    cell.style.backgroundColor = toRGBAString(originalColorObject);
  });
  highlightedCells = [];
}

function handleMouseover(event) {
  const cell = event.target.closest('.grid-cell');
  if (!cell) return;
  document.body.style.cursor = 'none';
  const x = parseInt(cell.dataset.x);
  const y = parseInt(cell.dataset.y);
  hoveredCell = cell;
  highlightHoverShape(x, y);
}

function handleMouseout(event) {
  const cell = event.target.closest('.grid-cell');
  if (!cell) return;
  document.body.style.cursor = 'auto';
  hoveredCell = null;
  clearHoverShape();
}

function handlePlacement(event) {
  const cell = event.target.closest('.grid-cell');
  if (!cell) return;
  const startX = parseInt(cell.dataset.x);
  const startY = parseInt(cell.dataset.y);
  if (isValidPlacement(startX, startY)) {
    placePieceOnGrid(startX, startY);
    available_pieces.splice(chosen_piece_index, 1);
    chosen_piece_index = 0;
    if (available_pieces.length > 0) {
      current_piece = available_pieces[chosen_piece_index];
      current_piece_original_index = PIECES.indexOf(current_piece);
    } else {
      current_piece = null;
      current_piece_original_index = -1;
    }
    clearHoverShape();
  } else {
    console.log('Invalid placement!');
  }
}

function placePieceOnGrid(startX, startY) {
  const piece = current_piece;
  const pieceColorKey = current_piece_original_index + 1;
  for (let y = 0; y < piece.length; y++) {
    for (let x = 0; x < piece[0].length; x++) {
      if (piece[y][x] === 1) {
        const gridX = startX + x;
        const gridY = startY + y;
        grid_representation[gridY][gridX] = pieceColorKey;
      }
    }
  }
  console.log(grid_representation)
  renderGrid();
}

//==================================================
//  COLOUR CONVERSION 
//==================================================

function toRGBAString(colorObject) {
  return `rgba(${colorObject.r}, ${colorObject.g}, ${colorObject.b}, ${colorObject.a})`;
}

function getOpaqueColor(key, opacity = 0.8) {
  const originalColor = PIECE_COLOUR_MAP.get(key);
  if (!originalColor) {
    return 'transparent';
  }
  const newColor = { ...originalColor, a: opacity };
  return toRGBAString(newColor);
}

//==================================================
//  EVENT LISTENERS 
//==================================================

document.addEventListener('keydown', (event) => {
  if (!available_pieces.length) return;

  const numPieces = available_pieces.length;
  if (event.key === 'e') {
    chosen_piece_index = (chosen_piece_index + 1) % numPieces;
    current_piece = available_pieces[chosen_piece_index];
    current_piece_original_index = PIECES.indexOf(current_piece);
  }
  if (event.key === 'q') {
    chosen_piece_index = (chosen_piece_index - 1 + numPieces) % numPieces;
    current_piece = available_pieces[chosen_piece_index];
    current_piece_original_index = PIECES.indexOf(current_piece);
  }
  if (event.key === 'a') {
    current_piece = rotatePiece(current_piece, 'anti-clockwise');
  }
  if (event.key === 'd') {
    current_piece = rotatePiece(current_piece, 'clockwise');
  }
  if (event.key === 'w') {
    current_piece = flipPiece(current_piece);
  }

  if (hoveredCell) {
    const x = parseInt(hoveredCell.dataset.x);
    const y = parseInt(hoveredCell.dataset.y);
    highlightHoverShape(x, y);
  }
});

const resetButton = document.getElementById('reset-button');
if (resetButton) {
  resetButton.addEventListener('click', resetGrid);
}

// Add this function to your script.js file

async function solveAIGrid() {
    // 1. Get the current AI grid representation (blocker positions)
    const blockers = [];
    for (let y = 0; y < AI_GRID_SIZE; y++) {
        for (let x = 0; x < AI_GRID_SIZE; x++) {
            if (AI_grid_representation[y][x] === BLOCKER_COLOR_KEY) {
                blockers.push([y, x]);
            }
        }
    }

    if (blockers.length === 0) {
        console.log("No blockers found on the AI grid.");
        return;
    }

    console.log("Sending blockers to API:", blockers);

    try {
        // 2. Make a POST request to your backend API
        const response = await fetch('http://127.0.0.1:5000/solve', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ blockers }),
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const responseData = await response.json();
        console.log("Response received:", responseData);

        // 3. Process and display the solution on the AI board
        displaySolution(responseData);

    } catch (error) {
        console.error('Error fetching solution:', error);
        alert('Failed to get a solution from the server.');
    }
}

function displaySolution(solution) {
    if (!solution || !solution.solution) {
        console.log("No solution found or invalid response format.");
        return;
    }

    const solvedGrid = solution.solution;

    // Directly assign the 2D array from the API to the global grid representation
    AI_grid_representation = solvedGrid;
    
    // Re-render the AI grid to show the solved state
    renderAIGrid();
}

// This function is no longer needed because the API returns the final grid
// function getPieceFromSolution(piece_index, rotation_index, flip) {
//     // ... (old code) ...
// }

// Add a "Solve" button to your index.html:

const solveButton = document.getElementById('solve-button');
if (solveButton) {
    solveButton.addEventListener('click', solveAIGrid);
}

//==================================================
//  INITIALISE DOCUMENT
//==================================================

createGrid(gridContainer);
createGrid(AI_gridContainer, true);
resetGrid();