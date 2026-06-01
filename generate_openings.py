# generate openings file
# credit: chatgpt

import os
import sys

import chess
import chess.engine
from collections import deque
from tqdm import tqdm



STOCKFISH_PATH = "stockfish"
MAX_DEPTH = 4
ENGINE_TIME = 0.05
OUTPUT_FILE = f"./src/openings_best_moves_{MAX_DEPTH}_{ENGINE_TIME}.txt"



def main():
	if os.path.exists(OUTPUT_FILE):
		print(f"Error: {OUTPUT_FILE} already exists. Refusing to overwrite.")
		sys.exit(1)

	print("Building BFS position queue...")
	positions = build_position_queue()
	print(f"Generated {len(positions)} positions")

	print("Evaluating with Stockfish...")
	evaluate_positions(positions)



def build_position_queue():
	"""
	BFS phase: only generate positions, no Stockfish calls.
	"""
	root = chess.Board()
	queue = deque([(root, 0)])
	visited = set()
	positions = []

	while queue:
		board, depth = queue.popleft()
		key = board.fen()

		if key in visited:
			continue
		visited.add(key)

		if board.is_game_over() or len(list(board.legal_moves)) == 0:
			continue

		positions.append(board.copy())

		if depth >= MAX_DEPTH:
			continue

		for move in board.legal_moves:
			child = board.copy()
			child.push(move)
			queue.append((child, depth + 1))

	return positions



def evaluate_positions(positions):
	"""
	Phase 2: Stockfish evaluation with progress bar.
	"""
	engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)

	with open(OUTPUT_FILE, "w") as f:
		for board in tqdm(positions, desc="Evaluating positions"):
			try:
				result = engine.play(board, chess.engine.Limit(time=ENGINE_TIME))
				best_move = result.move
			except Exception:
				continue

			f.write(f"{board.fen()}\n")
			f.write(f"{best_move.uci()}\n")

	engine.quit()





if __name__ == "__main__":
	main()

