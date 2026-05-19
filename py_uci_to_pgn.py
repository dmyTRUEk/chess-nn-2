# uci to pgn

import sys

import chess
import chess.pgn

# moves = "f2f4 d7d5 g1f3 d8d7 c2c3 e8d8".split()
moves = sys.argv[1:]

board = chess.Board()
game = chess.pgn.Game()

node = game

for move_str in moves:
	move = chess.Move.from_uci(move_str)

	if move not in board.legal_moves:
		raise ValueError(f"Illegal move: {move_str}")

	board.push(move)
	node = node.add_variation(move)

print(game)

