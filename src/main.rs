//! chess-nn 2

#![feature(
	vec_from_fn,
)]

#![deny(
	unreachable_patterns,
	unused_results,
	// unused_variables,
	clippy::let_unit_value,
)]

#![allow(
	clippy::let_and_return,
	clippy::manual_is_multiple_of,
	clippy::map_flatten,
	clippy::print_literal,
	clippy::upper_case_acronyms,
)]

use std::cmp::Ordering;

use chess::{ALL_SQUARES, Action, Board, BoardBuilder, ChessMove, Color, Game, GameResult, MoveGen, Piece};
use itertools::Itertools;
use nalgebra::{DMatrix, DVector};
use rand::{RngExt, rng, rngs::ThreadRng, seq::SliceRandom};
use rayon::iter::{ParallelBridge, ParallelIterator};

mod extensions;
mod math;
mod math_aliases;
mod typesafe_rng;
mod utils_io;

use extensions::*;
use math::*;
use math_aliases::*;
use typesafe_rng::*;
use utils_io::*;



mod training {
	use super::*;

	pub const EPOCHS: u32 = 100;
	pub const NNS_NUMBER: u32 = 20; // it's better be multiple of number of cores/threads on your machine? or else...
	// pub const TOURNAMENTS_NUMBER: u32 = 10;
	pub const PLAY_GAME_MOVES_LIMIT: u32 = 200;

	pub const EVOLUTION_RATE_INIT: f = 0.9;
	pub const EVOLUTION_RATE_FINAL: f = 0.001;

	pub const DEFAULT_RATING: f = 1_000.;

	// pub const CHESS_NN_THINK_DEPTH_FOR_TRAINING: u8 = 1;
	// pub const CHESS_NN_THINK_DEPTH_VS_HUMAN: u8 = 3; // 4 if parallel
}

mod nn {
	use super::*;

	pub const ACTIVATION_FN: ActivationFn = ActivationFn::LeakyReLU_01;
	// pub const ACTIVATION_FN: ActivationFn = ActivationFn::LeakyReLU_001;

	pub const INNER_LAYERS_SIZES: &[u32] = &[300, 100, 30, 10, 5];
	// pub const INNER_LAYERS_SIZES: &[u32] = &[30, 10, 5];

	// pub const NUMBER_OF_DEPTH_CHANNELS: NumberOfDepthChannels = NumberOfDepthChannels::Two;
	// pub const NUMBER_OF_DEPTH_CHANNELS: NumberOfDepthChannels = NumberOfDepthChannels::Three { use_opposite_signs: false };
	pub const NUMBER_OF_DEPTH_CHANNELS: NumberOfDepthChannels = NumberOfDepthChannels::Four;
	pub const NUMBER_OF_DIFFERENT_CHESS_PIECES: u32 = chess::NUM_PIECES as u32; // 6
	pub const NUMBER_OF_SQUARES_ON_CHESS_BOARD: u32 = chess::NUM_SQUARES as u32; // 64
	pub const INPUT_SIZE_PER_COLOR_CHANNEL: u32 = NUMBER_OF_DIFFERENT_CHESS_PIECES * NUMBER_OF_SQUARES_ON_CHESS_BOARD; // 384
	pub const INPUT_SIZE: u32 = NUMBER_OF_DEPTH_CHANNELS.to_u32() * INPUT_SIZE_PER_COLOR_CHANNEL; // 768 or 1152 or 1536
	pub const OUTPUT_SIZE: u32 = 1;

	// pub const COMPUTE_UNIT: ComputeUnit = ComputeUnit::CpuOne;
	pub const COMPUTE_UNIT: ComputeUnit = ComputeUnit::CpuN(4);
	// pub const COMPUTE_UNIT: ComputeUnit = ComputeUnit::CpuAll;

	pub const W_MIN: f = -1.;
	pub const W_MAX: f =  1.;
	pub const S_MIN: f = -10.;
	pub const S_MAX: f =  10.;
}





fn main() {
	debug_assert_eq!(1, nn::OUTPUT_SIZE);

	let mut rng = rng();

	if let ComputeUnit::CpuN(n) = nn::COMPUTE_UNIT {
		rayon::ThreadPoolBuilder::new()
			.num_threads(n as usize)
			.build_global()
			.unwrap();
	}

	let algo_players = { use AlgoPlayer::*; [
		RandomMover,
		MiddleMover,
		MaterialDelta,
		MaterialDeltaSTM,
		NegMaterialDelta,
		NegMaterialDeltaSTM,
		PiecesFreedom,
		PiecesFreedomSTM,
		NegPiecesFreedom,
		NegPiecesFreedomSTM,
		PiecesFreedomDiff,
		PiecesFreedomDiffSTM,
		NegPiecesFreedomDiff,
		NegPiecesFreedomDiffSTM,
		AlgoPlayer::mix_new_random(&mut rng), // one to survive
		AlgoPlayer::mix_new_random(&mut rng), // and one to evolve
		AlgoPlayer::mix_uss_new_random(&mut rng),
		AlgoPlayer::mix_uss_new_random(&mut rng),
	]};

	let mut players = algo_players.map(Player::Algo).to_vec();

	// {
	// 	let fen = (
	// 		// "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1" // init
	// 		// "rn2kb1r/ppp1pppp/5n2/3P1B2/8/2N2N2/PPPQ1PPP/R1B1K2R b KQkq - 0 7" // white winning, +11, black to move
	// 		// "rn2kb1r/1pp1pppp/p4n2/3P1B2/8/2N2N2/PPPQ1PPP/R1B1K2R w KQkq - 0 8" // white winning, +11, white to move
	// 		// "r1b1kbnr/pppp1ppp/8/4q3/3nP3/8/PPP2PPP/RN2K2R w KQkq - 1 9" // black winning, -16.5, white to move
	// 		"r1b1kbnr/pppp1ppp/8/4q3/3nP3/8/PPPN1PPP/R3K2R b KQkq - 2 9" // black winning, -16.5, black to move
	// 		// "rnbqkbnr/pppppppp/8/8/8/7N/PPPPPPPP/RNBQKB1R b KQkq - 1 1"
	// 		// "rnbqkbnr/1ppppppp/8/p7/8/7N/PPPPPPPP/RNBQKBR1 b Qkq - 1 2"
	// 		// "rnbqkb1r/pppppppp/5n2/8/8/6PN/PPPPPP1P/RNBQKB1R b KQkq - 0 2"
	// 		// "r1bqkbnr/8/n7/ppppppNp/8/8/PPPPPPPP/RNBQKBR1 w Qkq - 10 14"
	// 	);
	// 	let board = Board::from_fen(fen.to_string()).unwrap();
	// 	dbg!(AlgoPlayer::PiecesSum.eval_board(&board));
	// 	dbg!(AlgoPlayer::PiecesFreedom.eval_board(&board));
	// 	dbg!(AlgoPlayer::PiecesSumAndFreedom { sum_weight: 0.5, freedom_weight: 0.5 }.eval_board(&board));
	// 	dbg!(AlgoPlayer::PiecesSumAndFreedomUnderSignedSqrt { sum_weight: 0.5, freedom_weight: 0.5 }.eval_board(&board));
	// 	println!();
	// }
	// return;

	let nns = Vec::from_fn(training::NNS_NUMBER as usize, |_i| NN::new_random(nn::INNER_LAYERS_SIZES, &mut rng));
	assert_eq!(training::NNS_NUMBER, nns.len() as u32);
	println!("Created {} NNs", training::NNS_NUMBER);
	players.extend(
		nns.into_iter().map(Player::NN)
	);

	let mut players: Vec<PlayerWithRatingAndStats> = players.into_iter().map(PlayerWithRatingAndStats::new).collect();
	let players_n_init = players.len();
	println!("Number of players: {}", players.len());

	// TODO(feat): exp_k aka steepness: exp(k * ...)
	// erfinal = erinit * exp(-erdrop)  =>
	// erfinal/erinit = exp(-erdrop)  =>
	// ln(erfinal/erinit) = -erdrop  =>
	// erdrop = -ln(erfinal/erinit)  =>
	// erdrop = ln(erinit/erfinal)
	let evolution_rate_drop_speed = ln(training::EVOLUTION_RATE_INIT / training::EVOLUTION_RATE_FINAL);

	for epoch in 0..training::EPOCHS {
		println!();
		println!("{dashes} EPOCH {}/{} {dashes}", epoch+1, training::EPOCHS, dashes="-".repeat(42));
		println!();

		for PlayerWithRatingAndStats { player: _, rating, stats } in players.iter_mut() {
			*rating = training::DEFAULT_RATING;
			*stats = PlayerInTournamentStats::new();
		}

		players.shuffle(&mut rng);

		play_tournament(&mut players, training::PLAY_GAME_MOVES_LIMIT);

		println!();

		players.sort_by(|p1, p2| p1.rating.partial_cmp(&p2.rating).unwrap());
		players.reverse();

		println!("rating{t}name{spaces}wins/loses, wins/loses bp, draws bp", t='\t', spaces=" ".repeat(23));
		for player in players.iter().rev() {
			let rating = player.rating;
			let name = player.player.name();
			let PlayerInTournamentStats { wins, loses, wins_by_points, loses_by_points, draws_by_points } = player.stats;
			// let stats = format!("wins/loses: {wins}/{loses}, wins/loses by points: {wins_by_points}/{loses_by_points}, draws by points: {draws_by_points}");
			let stats = format!("{wins}/{loses}, {wins_by_points}/{loses_by_points}, {draws_by_points}");
			println!("{rating:.1}:\t{name:24}   {stats}");
		}

		println!();

		{ // best vs self
			let white = &players[0].player;
			let black = &players[0].player;
			let (game_result, Some(game)) = play_game(white, black, training::PLAY_GAME_MOVES_LIMIT, true) else { unreachable!() };
			println!("best vs self ({}):   {}", game_result.to_char(), game.to_uci());
		}
		println!();
		{ // best vs second
			let white = &players[0].player;
			let black = &players[1].player;
			let (game_result, Some(game)) = play_game(white, black, training::PLAY_GAME_MOVES_LIMIT, true) else { unreachable!() };
			println!("best vs second ({}):   {}", game_result.to_char(), game.to_uci());
		}
		println!();
		{ // best vs worst
			let white = &players[0].player;
			let black = &players[players.len()-1].player;
			let (game_result, Some(game)) = play_game(white, black, training::PLAY_GAME_MOVES_LIMIT, true) else { unreachable!() };
			println!("best vs worst ({}):   {}", game_result.to_char(), game.to_uci());
		}
		println!();
		{ // best vs best nn
			let white = &players[0].player;
			let black = &players.iter().find(|p| p.player.is_nn()).unwrap().player;
			let (game_result, Some(game)) = play_game(white, black, training::PLAY_GAME_MOVES_LIMIT, true) else { unreachable!() };
			println!("best vs best NN ({}):   {}", game_result.to_char(), game.to_uci());
		}
		println!();
		{ // best nn vs worst
			let white = &players.iter().find(|p| p.player.is_nn()).unwrap().player;
			let black = &players[players.len()-1].player;
			let (game_result, Some(game)) = play_game(white, black, training::PLAY_GAME_MOVES_LIMIT, true) else { unreachable!() };
			println!("best NN vs worst ({}):   {}", game_result.to_char(), game.to_uci());
		}
		println!();
		{ // best nn vs worst nn
			let white = &players.iter().find(|p| p.player.is_nn()).unwrap().player;
			let black = &players.iter().rev().find(|p| p.player.is_nn()).unwrap().player;
			let (game_result, Some(game)) = play_game(white, black, training::PLAY_GAME_MOVES_LIMIT, true) else { unreachable!() };
			println!("best NN vs worst NN ({}):   {}", game_result.to_char(), game.to_uci());
		}
		println!();
		{ // best nn vs second best nn
			let [white, black] = players.iter()
				.filter(|p| p.player.is_nn())
				.k_largest_by(2, |p1, p2| p1.rating.partial_cmp(&p2.rating).unwrap())
				.map(|p| &p.player)
				.collect::<Vec<_>>()[..] else { unreachable!() };
			let (game_result, Some(game)) = play_game(white, black, training::PLAY_GAME_MOVES_LIMIT, true) else { unreachable!() };
			println!("best NN vs second best NN ({}):   {}", game_result.to_char(), game.to_uci());
		}

		println!();

		// TODO(optim): parallel evolution
		let evolution_rate = training::EVOLUTION_RATE_INIT * exp(-evolution_rate_drop_speed * (epoch as f) / (training::EPOCHS as f - 1.));
		macro_rules! get_random_evo_rate { () => {{
			let k: f = rng.random_range(1. .. 10.);
			let evo_rate = evolution_rate * if rng.random_bool(0.5) { k } else { k.recip() };
			let evo_rate = evo_rate.clamp(0., 1.);
			evo_rate
		}}; }
		println!("evolving with evo_rate = {evolution_rate:.4} ...");
		let players_n = players.len();
		{ // evolve nns:
			const KEEP_TOP_NNS_FRAC: f = 2.3;
			let nns_n = players.iter().filter(|p| p.player.is_nn()).count();
			let keep_top_n_nns = ((nns_n as f) / KEEP_TOP_NNS_FRAC).round() as usize;
			let mut nns_i = 0;
			for i in 0..players_n {
				if nns_i < keep_top_n_nns { continue }
				if !players[i].player.is_nn() { continue }
				nns_i += 1;
				// natural selection:
				let player_to_clone = &players[i - keep_top_n_nns];
				players[i] = player_to_clone.clone(); // its always nn bc of `if is_nn => continue`
				// evolution:
				players[i].player.evolve(get_random_evo_rate!(), &mut rng);
			}
			// and (at least) one new random nn:
			for i in (keep_top_n_nns..players_n).rev() {
				if players[i].player.is_nn() {
					// println!("reseting `player[{i}]`");
					players[i] = PlayerWithRatingAndStats::new(Player::NN(NN::new_random(nn::INNER_LAYERS_SIZES, &mut rng)));
					break
				}
			}
		}
		{ // evolve algo mix:
			for i in 0..players_n {
				if !players[i].player.is_algo_mix() { continue }
				let index_of_first_algo_mix = players.iter()
					.position(|p| p.player.is_algo_mix())
					.unwrap();
				if i == index_of_first_algo_mix { continue }
				if players[i].rating < training::DEFAULT_RATING {
					if rng.random_bool(0.1) {
						players[i] = PlayerWithRatingAndStats::new(Player::Algo(AlgoPlayer::mix_new_random(&mut rng)));
						continue // dont evolve
					} else {
						players[i] = players[index_of_first_algo_mix].clone();
					};
				}
				players[i].player.evolve(get_random_evo_rate!(), &mut rng);
			}
		}
		{ // evolve algo mix uss:
			for i in 0..players_n {
				if !players[i].player.is_algo_mix_uss() { continue }
				let index_of_first_algo_mix_uss = players.iter()
					.position(|p| p.player.is_algo_mix_uss())
					.unwrap();
				if i == index_of_first_algo_mix_uss { continue }
				if players[i].rating < training::DEFAULT_RATING {
					if rng.random_bool(0.1) {
						players[i] = PlayerWithRatingAndStats::new(Player::Algo(AlgoPlayer::mix_uss_new_random(&mut rng)));
						continue // dont evolve
					} else {
						players[i] = players[index_of_first_algo_mix_uss].clone();
					};
				}
				players[i].player.evolve(get_random_evo_rate!(), &mut rng);
			}
		}

		assert_eq!(players_n_init, players.len());

		println!();
		println!();
	}
	println!();

	todo!("cli/repl to interact (play,inspect,save,etc) with nns/algos");
}





fn play_tournament(players: &mut [PlayerWithRatingAndStats], move_limit: u32) {
	let players_n = players.len();
	match nn::COMPUTE_UNIT {
		ComputeUnit::CpuOne => {
			print("games results: ");
			for white_i in 0..players_n {
				for black_i in 0..players_n {
					if white_i == black_i { continue }
					let [white, black] = players.get_disjoint_mut([white_i, black_i]).unwrap();
					let (game_result, _) = play_game(&white.player, &black.player, move_limit, false);
					update_stats(&mut white.stats, &mut black.stats, game_result);
					update_ratings(&mut white.rating, &mut black.rating, game_result);
					print(game_result.to_char());
				}
				print(" ");
			}
			println!();
		}
		ComputeUnit::CpuN(_) | ComputeUnit::CpuAll => {
			let players_ref: &[PlayerWithRatingAndStats] = &*players; // this fixes bc `&mut T` is not Copy (need for move), but `&T` is Copy (src: chatgpt)
			print("games results: ");
			let mut games: Vec<(usize, usize, GameResult_)> = (0..players_n).cartesian_product(0..players_n)
				.par_bridge()
				.map(|(white_i, black_i)| {
					if white_i == black_i { return None }
					let (game_result, _) = play_game(&players_ref[white_i].player, &players_ref[black_i].player, move_limit, false);
					print(game_result.to_char());
					Some((white_i, black_i, game_result))
				})
				.flatten() // remove `None`s // TODO?: dont flatten, and when printing print `_` or something
				.collect();
			games.sort_unstable_by_key(|g| (g.0, g.1));
			println!();
			println!();
			print!("organized: ");
			for (i, (white_i, black_i, game_result)) in games.into_iter().enumerate() {
				// println!("{white_i}, {black_i}, {game_result:?}");
				if white_i == black_i { unreachable!() }
				let [white, black] = players.get_disjoint_mut([white_i, black_i]).unwrap();
				update_stats(&mut white.stats, &mut black.stats, game_result);
				update_ratings(&mut white.rating, &mut black.rating, game_result);
				print!("{}", game_result.to_char());
				// println!("CHAR: {}", game_result.to_char());
				if i % (players_n - 1) == players_n - 2 { print(" "); }
			}
			println!();
		}
		ComputeUnit::Gpu => {
			todo!("use same as CpuOne?")
		}
	}
}

fn update_ratings(white: &mut f, black: &mut f, game_result: GameResult_) {
	use GameResult_::*;
	match game_result {
		WhiteWins => {
			let elo_rating_delta = calc_elo_rating_delta(*white, *black);
			*white += elo_rating_delta;
			*black -= elo_rating_delta;
		}
		BlackWins => {
			let elo_rating_delta = calc_elo_rating_delta(*black, *white);
			*black += elo_rating_delta;
			*white -= elo_rating_delta;
		}
		WhiteWinsByPoints => {
			let elo_rating_delta = calc_elo_rating_delta(*white, *black) / 100.;
			*white += elo_rating_delta;
			*black -= elo_rating_delta;
		}
		BlackWinsByPoints => {
			let elo_rating_delta = calc_elo_rating_delta(*black, *white) / 100.;
			*black += elo_rating_delta;
			*white -= elo_rating_delta;
		}
		DrawByPoints => {
			// let elo_rating_delta_1 = calc_elo_rating_delta(*white, *black);
			// let elo_rating_delta_2 = calc_elo_rating_delta(*black, *white);
			// let elo_rating_delta = (elo_rating_delta_1 + elo_rating_delta_2) / 2.;
			// let elo_rating_delta = elo_rating_delta / 1000.;
			// TODO?
			*black *= 0.999;
			*white *= 0.999;
		}
	}
}

fn update_stats(white: &mut PlayerInTournamentStats, black: &mut PlayerInTournamentStats, game_result: GameResult_) {
	match game_result {
		GameResult_::WhiteWins => {
			white.wins += 1;
			black.loses += 1;
		}
		GameResult_::BlackWins => {
			black.wins += 1;
			white.loses += 1;
		}
		GameResult_::WhiteWinsByPoints => {
			white.wins_by_points += 1;
			black.loses_by_points += 1;
		}
		GameResult_::BlackWinsByPoints => {
			black.wins_by_points += 1;
			white.loses_by_points += 1;
		}
		GameResult_::DrawByPoints => {
			white.draws_by_points += 1;
			black.draws_by_points += 1;
		}
	}
}

fn calc_elo_rating_delta(winner: f, loser: f) -> f {
	100. / ( 1. + 10_f32.powf( (winner-loser) / 400. ) )
}

#[derive(Debug, Clone)]
struct PlayerInTournamentStats { wins: u32, loses: u32, wins_by_points: u32, loses_by_points: u32, draws_by_points: u32 }
impl PlayerInTournamentStats {
	fn new() -> Self { Self { wins: 0, loses: 0, wins_by_points: 0, loses_by_points: 0, draws_by_points: 0 } }
}

fn play_game(white: &Player, black: &Player, move_limit: u32, get_game: bool) -> (GameResult_, Option<Game>) {
	let mut rng = rng();
	let mut game = Game::new(); // TODO!(optim): dont use Game, use Board directly
	let mut move_number: u32 = 0;
	while game.result() == None && move_number < move_limit { // TODO
		move_number += 1;
		let board = game.current_position();
		let side_to_move: Color = board.side_to_move();
		let player_to_make_move = match side_to_move {
			Color::White => white,
			Color::Black => black,
		};
		let selected_move = player_to_make_move.select_move(&board, &mut rng);
		let is_move_successful = game.make_move(selected_move);
		assert!(is_move_successful);
		if game.can_declare_draw() {
			let _ = game.declare_draw();
		}
	}
	let game_res = game.result();
	type GR = GameResult;
	let winner = match game_res.unwrap_or(GR::Stalemate) {
		GR::WhiteCheckmates | GR::BlackResigns => Some(GameResult_::WhiteWins),
		GR::WhiteResigns | GR::BlackCheckmates => Some(GameResult_::BlackWins),
		GR::Stalemate | GR::DrawAccepted | GR::DrawDeclared => None,
	};
	let gr = if let Some(winner) = winner {
		winner
	} else {
		let board = game.current_position();
		let points = board.count_material_delta();
		match points.partial_cmp(&0.).unwrap() {
			Ordering::Less => GameResult_::BlackWinsByPoints,
			Ordering::Equal => GameResult_::DrawByPoints,
			Ordering::Greater => GameResult_::WhiteWinsByPoints,
		}
	};
	(gr, get_game.then(|| game))
}

pub trait BoardCountMaterialDelta { fn count_material_delta(self) -> f; }
impl BoardCountMaterialDelta for Board {
	fn count_material_delta(self) -> f {
		let mut material_delta = 0.;
		let board_builder: BoardBuilder = self.into();
		for square in ALL_SQUARES.into_iter() {
			let maybe_piece_and_color: Option<(Piece, Color)> = board_builder[square];
			if let Some((piece, color)) = maybe_piece_and_color {
				let piece_value = piece_and_color_to_value(piece, color); // TODO(optim): dont count kings
				material_delta += piece_value;
			}
		}
		material_delta
	}
}

pub fn piece_and_color_to_value(piece: Piece, color: Color) -> f {
	let value = piece_to_value(piece);
	value * color.to_sign()
}

pub fn piece_to_value(piece: Piece) -> f {
	match piece {
		Piece::Pawn => 1.,
		Piece::Knight => 2.5,
		Piece::Bishop => 3.,
		Piece::Rook => 5.,
		Piece::Queen => 8.,
		Piece::King => 10.,
	}
}

pub trait BoardSideToMoveToSign { fn side_to_move_to_sign(&self) -> f; }
impl BoardSideToMoveToSign for Board {
	fn side_to_move_to_sign(&self) -> f {
		self.side_to_move().to_sign()
	}
}

pub trait ColorToSign { fn to_sign(self) -> f; }
impl ColorToSign for Color {
	fn to_sign(self) -> f {
		match self {
			Color::White => 1.,
			Color::Black => -1.,
		}
	}
}



#[derive(Debug, Clone, Copy)]
#[repr(u8)]
enum GameResult_ {
	WhiteWins,
	BlackWins,
	// Draw,
	WhiteWinsByPoints,
	BlackWinsByPoints,
	DrawByPoints,
}
impl GameResult_ {
	pub fn to_white_game_result(self) -> PlayerGameResult {
		match self {
			GameResult_::WhiteWins => PlayerGameResult::Win,
			GameResult_::BlackWins => PlayerGameResult::Lose,
			// GameResult_::Draw => PlayerGameResult::Draw,
			GameResult_::WhiteWinsByPoints => PlayerGameResult::WinByPoints,
			GameResult_::BlackWinsByPoints => PlayerGameResult::LoseByPoints,
			GameResult_::DrawByPoints => PlayerGameResult::DrawByPoints,
		}
	}
	pub fn to_black_game_result(self) -> PlayerGameResult {
		match self {
			GameResult_::BlackWins => PlayerGameResult::Win,
			GameResult_::WhiteWins => PlayerGameResult::Lose,
			// GameResult_::Draw => PlayerGameResult::Draw,
			GameResult_::BlackWinsByPoints => PlayerGameResult::WinByPoints,
			GameResult_::WhiteWinsByPoints => PlayerGameResult::LoseByPoints,
			GameResult_::DrawByPoints => PlayerGameResult::DrawByPoints,
		}
	}
	pub fn to_char(self) -> char {
		use GameResult_::*;
		match self {
			WhiteWins => 'W',
			BlackWins => 'B',
			WhiteWinsByPoints => 'w',
			BlackWinsByPoints => 'b',
			DrawByPoints => '.',
		}
	}
}

#[derive(Debug, Clone, Copy)]
#[repr(u8)]
enum PlayerGameResult {
	Win,
	Lose,
	// Draw,
	WinByPoints,
	LoseByPoints,
	DrawByPoints,
}





#[repr(u8)]
enum ActivationFn {
	LeakyReLU_01,
	LeakyReLU_001,
}
impl ActivationFn {
	pub fn eval(self, x: f) -> f {
		use ActivationFn::*;
		match self {
			LeakyReLU_01 => if x.is_sign_positive() { x } else { 0.1 * x },
			LeakyReLU_001 => if x.is_sign_positive() { x } else { 0.01 * x },
		}
	}
}



// TODO(refactor): make more variants
#[repr(u8)]
enum NumberOfDepthChannels {
	Two,
	Three { use_opposite_signs: bool },
	Four,
}
impl NumberOfDepthChannels {
	const fn to_u32(self) -> u32 {
		use NumberOfDepthChannels::*;
		match self {
			Two => 2,
			Three { .. } => 3,
			Four => 4,
		}
	}
}



#[derive(Clone)]
#[repr(u8)]
enum Player {
	NN(NN),
	Algo(AlgoPlayer),
	Human { name: String },
}
impl Player {
	pub fn select_move(&self, board: &Board, rng: &mut ThreadRng) -> ChessMove {
		use Player::*;
		match self {
			NN(nn) => nn.select_move(board),
			Algo(algo) => algo.select_move(board, rng),
			Human { name: _ } => todo!(),
		}
	}
	pub fn name(&self) -> String {
		use Player::*;
		match self {
			NN(nn) => format!("NN {:x}", nn.calc_hash()),
			Algo(algo) => algo.get_name(),
			Human { name } => name.clone(),
		}
	}
	pub fn is_evolvalbe(&self) -> bool {
		self.is_nn()
		| self.is_algo_mix()
		| self.is_algo_mix_uss()
	}
	pub fn is_nn(&self) -> bool {
		matches!(self, Player::NN(_))
	}
	pub fn is_algo_mix(&self) -> bool {
		matches!(self, Player::Algo(AlgoPlayer::Mix(_)))
	}
	pub fn is_algo_mix_uss(&self) -> bool {
		matches!(self, Player::Algo(AlgoPlayer::MixUnderSignedSqrt(_)))
	}
	pub fn evolve(&mut self, evolution_rate: f, rng: &mut ThreadRng) {
		use Player::*;
		match self {
			NN(nn) => {
				nn.evolve(evolution_rate, rng);
			}
			Algo(AlgoPlayer::Mix(mix)) => {
				mix.evolve(evolution_rate, rng);
			}
			Algo(_) => {}
			Human { name: _ } => {}
		}
	}
}

#[derive(Clone)]
struct PlayerWithRatingAndStats { player: Player, rating: f, stats: PlayerInTournamentStats }
impl PlayerWithRatingAndStats {
	pub fn new(player: Player) -> PlayerWithRatingAndStats {
		Self { player, rating: training::DEFAULT_RATING, stats: PlayerInTournamentStats::new() }
	}
}
// #[derive(Debug, Clone, Copy)]
// struct Rating(f);
// impl Rating { fn new() -> Self { Self(training::DEFAULT_RATING) } }


#[derive(Clone)]
struct NN { layers: Vec<NNLayer> }
impl NN {
	pub fn new_random(inner_layers_sizes: &[u32], rng: &mut ThreadRng) -> Self {
		let all_layers_sizes = [&[nn::INPUT_SIZE], inner_layers_sizes, &[nn::OUTPUT_SIZE]].concat();
		Self {
			layers: all_layers_sizes
				.array_windows()
				.cloned()
				.map(|[size_in, size_out]| NNLayer::new_random(size_in, size_out, rng))
				.collect()
		}
	}
	pub fn select_move(&self, board: &Board) -> ChessMove {
		let moves_and_scores = MoveGen::new_legal(board)
			.flat_map(|move_| {
				let board_after_move = board.make_move_new(move_);
				let score = self.eval_board(&board_after_move);
				score.is_finite().then_some((move_, score))
			});
		let (best_move, _best_move_score) = match board.side_to_move() {
			Color::White => {
				moves_and_scores
					.max_by(|(_m1,s1), (_m2,s2)| s1.partial_cmp(s2).unwrap())
					.unwrap()
			}
			Color::Black => {
				moves_and_scores
					.min_by(|(_m1,s1), (_m2,s2)| s1.partial_cmp(s2).unwrap())
					.unwrap()
			}
		};
		best_move
	}
	fn eval_board(&self, board: &Board) -> f {
		let nn_input = board_to_vector_for_nn(board);
		let nn_output = self.eval_input(nn_input);
		nn_output
	}
	fn eval_input(&self, input: Vec<f>) -> f {
		let mut v: DVector<f> = input.into();
		for layer in self.layers.iter() {
			v = layer.eval(v);
		}
		debug_assert_eq!(nn::OUTPUT_SIZE, v.len() as u32);
		v[0]
	}
	pub fn evolve(&mut self, evolution_rate: f, rng: &mut ThreadRng) {
		for layer in self.layers.iter_mut() {
			layer.evolve(evolution_rate, rng);
		}
	}
	pub fn calc_hash(&self) -> u64 {
		let layers: &[NNLayer] = &self.layers;
		let mut hash: u64 = 0x_1e88d6f0_b31da73f;
		for layer in layers {
			hash ^= layer.calc_hash();
		}
		hash
	}
}

#[derive(Clone)]
struct NNLayer { weights: DMatrix<f>, biases: DVector<f> }
impl NNLayer {
	pub fn new_random(size_in: u32, size_out: u32, rng: &mut ThreadRng) -> Self {
		// TODO(optim): dont use `random_range` repeatedly, instead create uniform distribution and multi sample it
		Self {
			weights: DMatrix::from_fn(
				size_out as usize,
				size_in as usize,
				|_i, _j| rng.random_range(nn::W_MIN .. nn::W_MAX)
			),
			biases: DVector::from_fn(
				size_out as usize,
				|_i, _| rng.random_range(nn::S_MIN .. nn::S_MAX) // TODO?: `S_MAX` dependant on `size_out`
			),
		}
	}
	pub fn calc_hash(&self) -> u64 {
		let mut hash: u64 = 0x_c695d51f_e59c7bed;
		for bias in self.biases.iter() {
			let bits = bias.to_bits() as u64;
			hash ^= if hash.count_ones() % 2 == 0 { bits } else { bits << 32 };
		}
		for weight in self.weights.iter() {
			let bits = weight.to_bits() as u64;
			hash ^= if hash.count_ones() % 2 == 0 { bits } else { bits << 32 };
		}
		hash
	}
	pub fn eval(&self, input: DVector<f>) -> DVector<f> {
		let sums = &self.weights * input + &self.biases;
		sums.map(|sum| nn::ACTIVATION_FN.eval(sum))
	}
	pub fn evolve(&mut self, evolution_rate: f, rng: &mut ThreadRng) {
		// TODO(optim): generate indices to evolve
		for bias in self.biases.iter_mut() {
			if rng.random_bool(evolution_rate as f64) {
				evolve_bias(bias, rng);
			}
		}
		for weight in self.weights.iter_mut() {
			if rng.random_bool(evolution_rate as f64) {
				evolve_weight(weight, rng);
			}
		}
	}
}

fn evolve_bias(bias: &mut f, rng: &mut ThreadRng) {
	evolve_value(bias, rng);
}

fn evolve_weight(weight: &mut f, rng: &mut ThreadRng) {
	evolve_value(weight, rng);
}

fn evolve_value(v: &mut f, rng: &mut ThreadRng) {
	match_random_weighted! {rng,
		// */
		0.01 => { *v *= -1.; },
		1. => { *v *= 2.; },
		1. => { *v /= 2.; },
		2. => { *v *= 1.4; },
		2. => { *v /= 1.4; },
		2. => { *v *= 1.1; },
		2. => { *v /= 1.1; },
		1. => { *v *= 1.01; },
		1. => { *v /= 1.01; },
		// +-
		0.2 => { *v += 0.1; },
		0.2 => { *v -= 0.1; },
		0.5 => { *v += 0.01; },
		0.5 => { *v -= 0.01; },
		0.2 => { *v += 0.001; },
		0.2 => { *v -= 0.001; },
		0.1 => { *v += 0.0001; },
		0.1 => { *v -= 0.0001; },
		0.01 => { *v += 0.3; },
		0.01 => { *v -= 0.3; },
		// +- sqrt
		0.001 => { *v += sqrt(abs(*v)); },
		0.001 => { *v -= sqrt(abs(*v)); },
		0.001 => { *v += sqrt(1. / abs(*v)); },
		0.001 => { *v -= sqrt(1. / abs(*v)); },
		// +- ln
		// 0.0001 => { *v += sqrt(ln(*v)); },
		// 0.0001 => { *v -= sqrt(ln(*v)); },
		// 0.0001 => { *v += sqrt(1. / ln(*v)); },
		// 0.0001 => { *v -= sqrt(1. / ln(*v)); },
	}
}





// TODO: test
fn board_to_vector_for_nn(board: &Board) -> Vec<f> {
	let mut result: Vec<f> = vec![0.; nn::INPUT_SIZE as usize];
	let board_builder: BoardBuilder = board.into();
	for (index_in_64, square) in ALL_SQUARES.into_iter().enumerate() {
		let option_piece_and_color: Option<(Piece, Color)> = board_builder[square];
		if let Some((piece, color)) = option_piece_and_color {
			// bow = white or black
			let index_of_64_wob = match (piece, color) {
				(Piece::Pawn  , Color::White) => 0,
				(Piece::Knight, Color::White) => 1,
				(Piece::Bishop, Color::White) => 2,
				(Piece::Rook  , Color::White) => 3,
				(Piece::Queen , Color::White) => 4,
				(Piece::King  , Color::White) => 5,
				(Piece::Pawn  , Color::Black) => 6,
				(Piece::Knight, Color::Black) => 7,
				(Piece::Bishop, Color::Black) => 8,
				(Piece::Rook  , Color::Black) => 9,
				(Piece::Queen , Color::Black) => 10,
				(Piece::King  , Color::Black) => 11,
			};
			// set white or black channel:
			result[64*index_of_64_wob + index_in_64] = 1.;
			// TODO(refactor)?: extract this `1.` into const

			fn get_index_of_64_wab(piece: Piece) -> usize {
				match piece {
					Piece::Pawn   => 12,
					Piece::Knight => 13,
					Piece::Bishop => 14,
					Piece::Rook   => 15,
					Piece::Queen  => 16,
					Piece::King   => 17,
				}
			}

			match nn::NUMBER_OF_DEPTH_CHANNELS {
				NumberOfDepthChannels::Two => {}
				NumberOfDepthChannels::Three { use_opposite_signs } => {
					// wab = white and black
					let index_of_64_wab = get_index_of_64_wab(piece);
					let value = if !use_opposite_signs { 1. } else { color.to_sign() };
					// set white and black channel:
					result[64*index_of_64_wab + index_in_64] = value;
				}
				NumberOfDepthChannels::Four => {
					// wab = white and black
					let index_of_64_wab = get_index_of_64_wab(piece);
					// set white and black channel:
					result[64*index_of_64_wab + index_in_64] = 1.;

					// wanb = white and negative black
					let index_of_64_wanb = match piece {
						Piece::Pawn   => 18,
						Piece::Knight => 19,
						Piece::Bishop => 20,
						Piece::Rook   => 21,
						Piece::Queen  => 22,
						Piece::King   => 23,
					};
					let value = color.to_sign();
					// set white and negative black channel:
					result[64*index_of_64_wanb + index_in_64] = value;
				}
			}
		}
	}
	result
}





#[derive(Debug, Clone, Copy)]
#[repr(u8)]
enum AlgoPlayer {
	RandomMover,
	MiddleMover,
	MaterialDelta,
	MaterialDeltaSTM,
	NegMaterialDelta,
	NegMaterialDeltaSTM,
	PiecesFreedom,
	PiecesFreedomSTM,
	NegPiecesFreedom,
	NegPiecesFreedomSTM,
	PiecesFreedomDiff,
	PiecesFreedomDiffSTM,
	NegPiecesFreedomDiff,
	NegPiecesFreedomDiffSTM,
	Mix(AlgoPlayerMix),
	MixUnderSignedSqrt(AlgoPlayerMix),
}
impl AlgoPlayer {
	pub fn mix_new_random(rng: &mut ThreadRng) -> Self {
		Self::Mix(AlgoPlayerMix::new_random(rng))
	}
	pub fn mix_uss_new_random(rng: &mut ThreadRng) -> Self {
		Self::MixUnderSignedSqrt(AlgoPlayerMix::new_random(rng))
	}
	pub fn select_move(self, board: &Board, rng: &mut ThreadRng) -> ChessMove {
		use AlgoPlayer::*;
		match self {
			RandomMover => {
				let moves = MoveGen::new_legal(board);
				let moves = moves.into_iter().collect::<Vec<_>>();
				let random_move_index = rng.random_range(0..moves.len());
				let random_move = moves[random_move_index];
				random_move
			}
			MiddleMover => {
				let moves = MoveGen::new_legal(board);
				let moves = moves.into_iter().collect::<Vec<_>>();
				let middle_move = moves[moves.len()/2];
				middle_move
			}
			MaterialDelta
			| MaterialDeltaSTM
			| NegMaterialDelta
			| NegMaterialDeltaSTM
			| PiecesFreedom
			| PiecesFreedomSTM
			| NegPiecesFreedom
			| NegPiecesFreedomSTM
			| PiecesFreedomDiff
			| PiecesFreedomDiffSTM
			| NegPiecesFreedomDiff
			| NegPiecesFreedomDiffSTM
			| Mix(_)
			| MixUnderSignedSqrt(_)
			=> {
				let moves_and_scores = MoveGen::new_legal(board)
					.flat_map(|move_| {
						let board_after_move = board.make_move_new(move_);
						let score = self.eval_board(&board_after_move, rng);
						score.is_finite().then_some((move_, score))
					});
				let (best_move, _best_move_score) = match board.side_to_move() {
					Color::White => {
						moves_and_scores
							.max_by(|(_m1,s1), (_m2,s2)| s1.partial_cmp(s2).unwrap())
							.unwrap_or_else(|| {
								dbg!(self);
								panic!()
							})
					}
					Color::Black => {
						moves_and_scores
							.min_by(|(_m1,s1), (_m2,s2)| s1.partial_cmp(s2).unwrap())
							.unwrap_or_else(|| {
								dbg!(self);
								let moves_and_scores = MoveGen::new_legal(board)
									.map(|move_| {
										let board_after_move = board.make_move_new(move_);
										let score = self.eval_board(&board_after_move, rng);
										(move_, score)
									});
								dbg!(moves_and_scores.collect::<Vec<_>>());
								panic!()
							})
					}
				};
				best_move
			}
		}
	}
	pub fn eval_board(self, board: &Board, rng: &mut ThreadRng) -> f {
		use AlgoPlayer::*;
		match self {
			RandomMover => rng.random_range(-50. .. 50.),
			MiddleMover => unreachable!(),
			Mix(mix) => {
				AlgoPlayerMix::ALGOS
					.map(|algo| algo.eval_board(board, rng))
					.iter().zip_eq(mix.to_array())
					.map(|(score, weight)| score * weight)
					.sum()
			}
			MixUnderSignedSqrt(mix) => {
				let under_sqrt = AlgoPlayerMix::ALGOS
					.map(|algo| algo.eval_board(board, rng))
					.iter().zip_eq(mix.to_array())
					.map(|(&score, weight)| signed_sqrt(score) * weight)
					.sum();
				signed_square(under_sqrt)
			}
			_ => self._eval_board(board)
		}
	}
	fn _eval_board(self, board: &Board) -> f {
		use AlgoPlayer::*;
		match self {
			RandomMover
			| MiddleMover
			=> unreachable!(),

			MaterialDelta => board.count_material_delta(),
			MaterialDeltaSTM => MaterialDelta._eval_board(board) * board.side_to_move_to_sign(),
			NegMaterialDelta => -MaterialDelta._eval_board(board),
			NegMaterialDeltaSTM => -MaterialDeltaSTM._eval_board(board),

			PiecesFreedom => MoveGen::new_legal(board).count() as f,
			PiecesFreedomSTM => PiecesFreedom._eval_board(board) * board.side_to_move_to_sign(),
			NegPiecesFreedom => -PiecesFreedom._eval_board(board),
			NegPiecesFreedomSTM => -PiecesFreedomSTM._eval_board(board),

			PiecesFreedomDiff => {
				let board_toggled_stm = board.null_move(); // "toggle" side to move
				PiecesFreedom._eval_board(board) - board_toggled_stm.map(|b| PiecesFreedom._eval_board(&b)).unwrap_or(0.)
			}
			PiecesFreedomDiffSTM => PiecesFreedomDiff._eval_board(board) * board.side_to_move_to_sign(),
			NegPiecesFreedomDiff => -PiecesFreedomDiff._eval_board(board),
			NegPiecesFreedomDiffSTM => -PiecesFreedomDiffSTM._eval_board(board),

			Mix(_) => unreachable!(),
			MixUnderSignedSqrt(_) => unreachable!(),
		}
	}
	pub fn get_name(&self) -> String {
		use AlgoPlayer::*;
		match self {
			RandomMover => "RandomMover".to_string(),
			MiddleMover => "MiddleMover".to_string(),
			MaterialDelta => "MaterialDelta".to_string(),
			MaterialDeltaSTM => "MaterialDelta StM".to_string(),
			NegMaterialDelta => "-MaterialDelta".to_string(),
			NegMaterialDeltaSTM => "-MaterialDelta StM".to_string(),
			PiecesFreedom => "PiecesFreedom".to_string(),
			PiecesFreedomSTM => "PiecesFreedom StM".to_string(),
			NegPiecesFreedom => "-PiecesFreedom".to_string(),
			NegPiecesFreedomSTM => "-PiecesFreedom StM".to_string(),
			PiecesFreedomDiff => "PiecesFreedomDiff".to_string(),
			PiecesFreedomDiffSTM => "PiecesFreedomDiff StM".to_string(),
			NegPiecesFreedomDiff => "-PiecesFreedomDiff".to_string(),
			NegPiecesFreedomDiffSTM => "-PiecesFreedomDiff StM".to_string(),
			Mix(mix) => format!("Mix {:x} ({})", mix.calc_hash(), mix.to_string()),
			MixUnderSignedSqrt(mix) => format!("MixUSS {:x} ({})", mix.calc_hash(), mix.to_string()),
		}
	}
}

#[derive(Debug, Clone, Copy)]
pub struct AlgoPlayerMix {
	random_mover: f,
	// middle_mover: f,
	material_delta: f,
	material_delta_stm: f,
	neg_material_delta: f,
	neg_material_delta_stm: f,
	pieces_freedom: f,
	pieces_freedom_stm: f,
	neg_pieces_freedom: f,
	neg_pieces_freedom_stm: f,
	pieces_freedom_diff: f,
	pieces_freedom_diff_stm: f,
	neg_pieces_freedom_diff: f,
	neg_pieces_freedom_diff_stm: f,
	// TODO?: remove Copy & use via Box when size bigger than 128 bytes
}
impl AlgoPlayerMix {
	const N: usize = 13;
	const ALGOS: [AlgoPlayer; Self::N] = { use AlgoPlayer::*; [
		RandomMover,
		MaterialDelta,
		MaterialDeltaSTM,
		NegMaterialDelta,
		NegMaterialDeltaSTM,
		PiecesFreedom,
		PiecesFreedomSTM,
		NegPiecesFreedom,
		NegPiecesFreedomSTM,
		PiecesFreedomDiff,
		PiecesFreedomDiffSTM,
		NegPiecesFreedomDiff,
		NegPiecesFreedomDiffSTM,
	]};
	pub fn new_random(rng: &mut ThreadRng) -> Self {
		let ws: [f; Self::N] = std::array::from_fn(|_i| rng.random_range(0. .. 1.));
		let ws_sum: f = ws.iter().sum();
		let ws = ws.map(|v| v / ws_sum);
		Self::from_array(ws)
	}
	pub fn from_array(weights: [f; Self::N]) -> Self {
		assert!(weights.iter().all(|w| w.is_finite()), "{weights:?}");
		let [random_mover, material_delta, material_delta_stm, neg_material_delta, neg_material_delta_stm, pieces_freedom, pieces_freedom_stm, neg_pieces_freedom, neg_pieces_freedom_stm, pieces_freedom_diff, pieces_freedom_diff_stm, neg_pieces_freedom_diff, neg_pieces_freedom_diff_stm ] = weights;
		Self { random_mover, material_delta, material_delta_stm, neg_material_delta, neg_material_delta_stm, pieces_freedom, pieces_freedom_stm, neg_pieces_freedom, neg_pieces_freedom_stm, pieces_freedom_diff, pieces_freedom_diff_stm, neg_pieces_freedom_diff, neg_pieces_freedom_diff_stm }
	}
	pub fn to_array(&self) -> [f; Self::N] {
		let Self { random_mover, material_delta, material_delta_stm, neg_material_delta, neg_material_delta_stm, pieces_freedom, pieces_freedom_stm, neg_pieces_freedom, neg_pieces_freedom_stm, pieces_freedom_diff, pieces_freedom_diff_stm, neg_pieces_freedom_diff, neg_pieces_freedom_diff_stm } = *self;
		[random_mover, material_delta, material_delta_stm, neg_material_delta, neg_material_delta_stm, pieces_freedom, pieces_freedom_stm, neg_pieces_freedom, neg_pieces_freedom_stm, pieces_freedom_diff, pieces_freedom_diff_stm, neg_pieces_freedom_diff, neg_pieces_freedom_diff_stm]
	}
	pub fn evolve(&mut self, evolution_rate: f, rng: &mut ThreadRng) {
		let mut ws = self.to_array();
		for w in ws.iter_mut() {
			if rng.random_bool(evolution_rate as f64) {
				// toggle zero / non-zero
				*w = if *w == 0. { rng.random_range(0. .. 1.) } else { 0. };
			}
			if rng.random_bool(evolution_rate as f64) {
				evolve_value(w, rng);
			}
			*w = w.clamp(0., 1.);
		}
		let ws_sum: f = ws.iter().sum();
		if ws_sum != 0. {
			ws = ws.map(|v| v / ws_sum);
		}
		*self = Self::from_array(ws);
	}
	pub fn calc_hash(&self) -> u64 {
		let mut hash: u64 = 0x_7dc29f45_3decba81;
		for w in self.to_array() {
			let bits = w.to_bits() as u64;
			hash ^= if hash.count_ones() % 2 == 0 { bits } else { bits << 32 };
		}
		hash
	}
}
impl ToString for AlgoPlayerMix {
	fn to_string(&self) -> String {
		let Self { random_mover, material_delta, material_delta_stm, neg_material_delta, neg_material_delta_stm, pieces_freedom, pieces_freedom_stm, neg_pieces_freedom, neg_pieces_freedom_stm, pieces_freedom_diff, pieces_freedom_diff_stm, neg_pieces_freedom_diff, neg_pieces_freedom_diff_stm } = *self;
		let sep = "=";
		let mut parts = vec![];
		if random_mover != 0. { parts.push(format!("random_mover{sep}{random_mover:.2}")); }
		if material_delta != 0. { parts.push(format!("material_delta{sep}{material_delta:.2}")); }
		if material_delta_stm != 0. { parts.push(format!("material_delta_stm{sep}{material_delta_stm:.2}")); }
		if neg_material_delta != 0. { parts.push(format!("neg_material_delta{sep}{neg_material_delta:.2}")); }
		if neg_material_delta_stm != 0. { parts.push(format!("neg_material_delta_stm{sep}{neg_material_delta_stm:.2}")); }
		if pieces_freedom != 0. { parts.push(format!("pieces_freedom{sep}{pieces_freedom:.2}")); }
		if pieces_freedom_stm != 0. { parts.push(format!("pieces_freedom_stm{sep}{pieces_freedom_stm:.2}")); }
		if neg_pieces_freedom != 0. { parts.push(format!("neg_pieces_freedom{sep}{neg_pieces_freedom:.2}")); }
		if neg_pieces_freedom_stm != 0. { parts.push(format!("neg_pieces_freedom_stm{sep}{neg_pieces_freedom_stm:.2}")); }
		if pieces_freedom_diff != 0. { parts.push(format!("pieces_freedom_diff{sep}{pieces_freedom_diff:.2}")); }
		if pieces_freedom_diff_stm != 0. { parts.push(format!("pieces_freedom_diff_stm{sep}{pieces_freedom_diff_stm:.2}")); }
		if neg_pieces_freedom_diff != 0. { parts.push(format!("neg_pieces_freedom_diff{sep}{neg_pieces_freedom_diff:.2}")); }
		if neg_pieces_freedom_diff_stm != 0. { parts.push(format!("neg_pieces_freedom_diff_stm{sep}{neg_pieces_freedom_diff_stm:.2}")); }
		parts.join(", ")
	}
}





#[repr(u8)]
enum ComputeUnit {
	CpuOne,
	CpuN(u32),
	CpuAll,
	Gpu,
}





pub trait ToUci { fn to_uci(&self) -> String; }
impl ToUci for Game {
	fn to_uci(&self) -> String {
		let moves_strs: Vec<String> = self.actions().into_iter().flat_map(|action| {
			match action {
				Action::MakeMove(move_) => Some(move_.to_string()),
				// Action::OfferDraw(Color) => todo!(),
				// Action::AcceptDraw => todo!(),
				// Action::DeclareDraw => todo!(),
				// Action::Resign(Color) => todo!(),
				_ => None
			}
		}).collect();
		moves_strs.join(" ")
	}
}





#[allow(non_camel_case_types)]
pub type f = f32;





#[cfg(test)]
mod tests {
	use super::*;

	mod calc_elo_rating_delta {
		use super::*;
		#[test] fn strong_wins() { assert_eq!(9.090909, calc_elo_rating_delta(1200., 800.)) }
		#[test] fn weak_wins  () { assert_eq!(90.90909, calc_elo_rating_delta(800., 1200.)) }
		#[test] fn white_1200__black_800__white_wins() { assert_eq!(9.090909, calc_elo_rating_delta(1200., 800.)) }
		#[test] fn white_1200__black_800__black_wins() { assert_eq!(90.90909, calc_elo_rating_delta(800., 1200.)) }
		#[test] fn white_800__black_1200__white_wins() { assert_eq!(90.90909, calc_elo_rating_delta(800., 1200.)) }
		#[test] fn white_800__black_1200__black_wins() { assert_eq!(9.090909, calc_elo_rating_delta(1200., 800.)) }
	}
}

