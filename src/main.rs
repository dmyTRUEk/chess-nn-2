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
	clippy::upper_case_acronyms,
)]

use std::cmp::Ordering;

use chess::{ALL_SQUARES, Action, Board, BoardBuilder, ChessMove, Color, Game, GameResult, MoveGen, Piece};
use itertools::Itertools;
use nalgebra::{DMatrix, DVector};
use rand::{RngExt, rng, rngs::ThreadRng};
use rayon::iter::{ParallelBridge, ParallelIterator};

mod extensions;
mod math_aliases;
mod typesafe_rng;
mod utils_io;

use extensions::*;
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

	// pub const NUMBER_OF_DEPTH_CHANNELS: NumberOfDepthChannels = NumberOfDepthChannels::Two;
	// pub const NUMBER_OF_DEPTH_CHANNELS: NumberOfDepthChannels = NumberOfDepthChannels::Three { use_opposite_signs: false };
	pub const NUMBER_OF_DEPTH_CHANNELS: NumberOfDepthChannels = NumberOfDepthChannels::Four;
	pub const NUMBER_OF_DIFFERENT_CHESS_PIECES: u32 = chess::NUM_PIECES as u32; // 6
	pub const NUMBER_OF_SQUARES_ON_CHESS_BOARD: u32 = chess::NUM_SQUARES as u32; // 64
	pub const INPUT_SIZE_PER_COLOR_CHANNEL: u32 = NUMBER_OF_DIFFERENT_CHESS_PIECES * NUMBER_OF_SQUARES_ON_CHESS_BOARD; // 384
	pub const INPUT_SIZE: u32 = NUMBER_OF_DEPTH_CHANNELS.to_u32() * INPUT_SIZE_PER_COLOR_CHANNEL; // 768 or 1152 or 1536
	pub const OUTPUT_SIZE: u32 = 1;

	// pub const COMPUTE_UNIT: ComputeUnit = ComputeUnit::CpuOne;
	pub const COMPUTE_UNIT: ComputeUnit = ComputeUnit::CpuAll;

	pub const W_MIN: f = -1.;
	pub const W_MAX: f =  1.;
	pub const S_MIN: f = -10.;
	pub const S_MAX: f =  10.;
}





fn main() {
	debug_assert_eq!(1, nn::OUTPUT_SIZE);

	let mut rng = rng();

	let mut players = vec![
		Player::Algo(AlgoPlayer::RandomMover),
		// Player::Algo(AlgoPlayer::PiecesSum),
		// Player::Algo(AlgoPlayer::PiecesFreedom),
		// Player::Algo(AlgoPlayer::PiecesSumAndFreedom { sum_weight: todo!(), freedom_weight: todo!() }),
	];

	let nns = Vec::from_fn(training::NNS_NUMBER as usize, |_i| NN::new_with_rng(nn::INNER_LAYERS_SIZES, &mut rng));
	assert_eq!(training::NNS_NUMBER, nns.len() as u32);
	println!("Created {} NNs", training::NNS_NUMBER);
	players.extend(
		nns.into_iter().map(Player::NN)
	);

	let mut players: Vec<PlayerWithRatingAndStats> = players.into_iter().map(PlayerWithRatingAndStats::new).collect();
	let players_n_init = players.len();
	println!("Number of players: {}", players.len());

	// erfinal = erinit * exp(-erdrop)  =>
	// erfinal/erinit = exp(-erdrop)  =>
	// ln(erfinal/erinit) = -erdrop  =>
	// erdrop = -ln(erfinal/erinit)  =>
	// erdrop = ln(erinit/erfinal)
	let evolution_rate_drop_speed = ln(training::EVOLUTION_RATE_INIT / training::EVOLUTION_RATE_FINAL);

	for epoch in 0..training::EPOCHS {
		println!();
		print!("epoch {}/{}: ", epoch+1, training::EPOCHS); flush();

		for PlayerWithRatingAndStats { player: _, rating, stats } in players.iter_mut() {
			*rating = training::DEFAULT_RATING;
			*stats = PlayerInTournamentStats::new();
		}

		play_tournament(&mut players, training::PLAY_GAME_MOVES_LIMIT);

		print!("ratings:"); players.iter().for_each(|p| print!(" {:.1}", p.rating)); println!();
		print!("ratings (sorted):"); players.iter().sorted_by(|p1, p2| p2.rating.partial_cmp(&p1.rating).unwrap()).for_each(|p| print!(" {:.1}", p.rating)); println!();

		println!();

		// TODO(refactor)
		{ // best player
			let title = "best";
			let (player_i, player) = players.iter().enumerate().max_by(|(_i1,p1), (_i2,p2)| p1.rating.partial_cmp(&p2.rating).unwrap()).unwrap();
			let name = match &player.player {
				Player::NN(nn) => format!("NN #{player_i} ({})", nn.name),
				Player::Algo(algo) => algo.get_name(),
				Player::Human { name } => format!("human ({name})"),
			};
			let rating = player.rating;
			let PlayerInTournamentStats { wins, loses, wins_by_points, loses_by_points, draws_by_points } = player.stats;
			println!("{title} player ({rating:.1}): {name}, wins/loses: {wins}/{loses}, wins/loses by points: {wins_by_points}/{loses_by_points}, draws by points: {draws_by_points}");
		}
		{ // second player
			let title = "second";
			let (player_i, player) = players.iter().enumerate().k_largest_by(2, |(_i1,p1), (_i2,p2)| p1.rating.partial_cmp(&p2.rating).unwrap()).last().unwrap();
			let name = match &player.player {
				Player::NN(nn) => format!("NN #{player_i} ({})", nn.name),
				Player::Algo(algo) => algo.get_name(),
				Player::Human { name } => format!("human ({name})"),
			};
			let rating = player.rating;
			let PlayerInTournamentStats { wins, loses, wins_by_points, loses_by_points, draws_by_points } = player.stats;
			println!("{title} player ({rating:.1}): {name}, wins/loses: {wins}/{loses}, wins/loses by points: {wins_by_points}/{loses_by_points}, draws by points: {draws_by_points}");
		}
		{ // worst player
			let title = "worst";
			let (player_i, player) = players.iter().enumerate().min_by(|(_i1,p1), (_i2,p2)| p1.rating.partial_cmp(&p2.rating).unwrap()).unwrap();
			let name = match &player.player {
				Player::NN(nn) => format!("NN #{player_i} ({})", nn.name),
				Player::Algo(algo) => algo.get_name(),
				Player::Human { name } => format!("human ({name})"),
			};
			let rating = player.rating;
			let PlayerInTournamentStats { wins, loses, wins_by_points, loses_by_points, draws_by_points } = player.stats;
			println!("{title} player ({rating:.1}): {name}, wins/loses: {wins}/{loses}, wins/loses by points: {wins_by_points}/{loses_by_points}, draws by points: {draws_by_points}");
		}
		println!();

		players.sort_by(|p1, p2| p2.rating.partial_cmp(&p1.rating).unwrap());

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

		let evolution_rate = training::EVOLUTION_RATE_INIT * exp(-evolution_rate_drop_speed * (epoch as f) / (training::EPOCHS as f - 1.));
		println!("evolving with evo_rate = {evolution_rate:.4} ...");
		let players_n = players.len();
		let save_top_n = players_n / 3;
		for i in save_top_n..players_n-1 {
			// natural selection:
			let player_to_clone = &players[i - save_top_n];
			players[i] = if player_to_clone.player.is_nn() {
				player_to_clone.clone()
			} else {
				PlayerWithRatingAndStats::new(Player::NN(NN::new_with_rng(nn::INNER_LAYERS_SIZES, &mut rng)))
			};
			// evolution:
			let k: f = rng.random_range(1. .. 10.);
			let evo_rate = evolution_rate * if rng.random_bool(0.5) { k } else { k.recip() };
			let evo_rate = evo_rate.clamp(0., 1.);
			evolve_player(&mut players[i].player, evo_rate, &mut rng);
		}
		players[players_n-1] = PlayerWithRatingAndStats::new(Player::NN(NN::new_with_rng(nn::INNER_LAYERS_SIZES, &mut rng)));

		assert_eq!(players_n_init, players.len());

		println!();
		println!();
	}
}



fn evolve_player(player: &mut Player, evolution_rate: f, rng: &mut ThreadRng) {
	use Player::*;
	match player {
		NN(nn) => { nn.evolve(evolution_rate, rng); }
		Algo(_) => {}
		Human { name: _ } => {}
	}
}



fn play_tournament(players: &mut [PlayerWithRatingAndStats], move_limit: u32) {
	let players_n = players.len();
	match nn::COMPUTE_UNIT {
		ComputeUnit::CpuOne => {
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
		ComputeUnit::Cpu(_) => {
			unimplemented!()
		}
		ComputeUnit::CpuAll => {
			let players_ref: &[PlayerWithRatingAndStats] = &*players; // this fixes bc `&mut T` is not Copy (need for move), but `&T` is Copy (src: chatgpt)
			let mut games: Vec<(usize, usize, GameResult_)> = (0..players_n).cartesian_product(0..players_n)
				.par_bridge()
				.map(|(white_i, black_i)| {
					if white_i == black_i { return None }
					let (game_result, _) = play_game(&players_ref[white_i].player, &players_ref[black_i].player, move_limit, false);
					print(game_result.to_char());
					Some((white_i, black_i, game_result))
				})
				.flatten() // remove `None`s
				.collect();
			games.sort_unstable_by_key(|g| (g.0, g.1));
			println!();
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
			// todo!();
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

pub trait CountMaterialDelta { fn count_material_delta(self) -> f; }
impl CountMaterialDelta for Board {
	fn count_material_delta(self) -> f {
		let mut material_delta = 0.;
		let board_builder: BoardBuilder = self.into();
		for square in ALL_SQUARES.into_iter() {
			let maybe_piece_and_color: Option<(Piece, Color)> = board_builder[square];
			if let Some((piece, color)) = maybe_piece_and_color {
				let piece_value = match piece {
					Piece::Pawn => 1.,
					Piece::Knight => 2.5,
					Piece::Bishop => 3.,
					Piece::Rook => 5.,
					Piece::Queen => 8.,
					Piece::King => 10., // TODO(optim): dont count
				};
				match color {
					Color::White => { material_delta += piece_value; }
					Color::Black => { material_delta -= piece_value; }
				}
			}
		}
		material_delta
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
			NN(nn) => nn.name.clone(),
			Algo(algo) => algo.get_name(),
			Human { name } => name.clone(),
		}
	}
	pub fn is_nn(&self) -> bool {
		use Player::*;
		match self {
			NN(_nn) => true,
			Algo(_)
			| Human { name: _ }
			=> false
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
struct NN { layers: Vec<NNLayer>, name: String }
impl NN {
	pub fn new_with_rng(inner_layers_sizes: &[u32], rng: &mut ThreadRng) -> Self {
		let all_layers_sizes = [&[nn::INPUT_SIZE], inner_layers_sizes, &[nn::OUTPUT_SIZE]].concat();
		let layers: Vec<_> = all_layers_sizes.array_windows().cloned().map(|[size_in, size_out]| NNLayer::new_with_rng(size_in, size_out, rng)).collect();
		let name = format!("{:x}", Self::calc_hash(&layers));
		Self { layers, name }
	}
	fn calc_hash(layers: &[NNLayer]) -> u64 {
		let mut hash: u64 = 0x_1e88d6f0_b31da73f;
		for layer in layers {
			hash ^= layer.calc_hash();
		}
		hash
	}
	pub fn select_move(&self, board: &Board) -> ChessMove {
		let mut best_move_and_score: Option<(ChessMove, f)> = None;
		for move_ in MoveGen::new_legal(board) {
			let board_after_move = board.make_move_new(move_);
			let this_move_score = self.eval_board(&board_after_move);
			if let Some((_best_move, best_move_score)) = best_move_and_score {
				if this_move_score > best_move_score {
					best_move_and_score = Some((move_, this_move_score));
				}
			} else {
				best_move_and_score = Some((move_, this_move_score));
			}
		}
		best_move_and_score.unwrap().0
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
}

#[derive(Clone)]
struct NNLayer { weights: DMatrix<f>, biases: DVector<f> }
impl NNLayer {
	pub fn new_with_rng(size_in: u32, size_out: u32, rng: &mut ThreadRng) -> Self {
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
			let bias_bits = bias.to_bits() as u64;
			hash ^= if hash.count_ones() % 2 == 0 { bias_bits } else { bias_bits << 32 };
		}
		for weight in self.weights.iter() {
			let weight_bits = weight.to_bits() as u64;
			hash ^= if hash.count_ones() % 2 == 0 { weight_bits } else { weight_bits << 32 };
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
		0.0001 => { *v += sqrt(ln(*v)); },
		0.0001 => { *v -= sqrt(ln(*v)); },
		0.0001 => { *v += sqrt(1. / ln(*v)); },
		0.0001 => { *v -= sqrt(1. / ln(*v)); },
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

			fn value_from_color(color: Color) -> f {
				match color {
					Color::White => 1.,
					Color::Black => -1.,
				}
			}

			match nn::NUMBER_OF_DEPTH_CHANNELS {
				NumberOfDepthChannels::Two => {}
				NumberOfDepthChannels::Three { use_opposite_signs } => {
					// wab = white and black
					let index_of_64_wab = get_index_of_64_wab(piece);
					let value = if !use_opposite_signs { 1. } else { value_from_color(color) };
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
					let value = value_from_color(color);
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
	PiecesSum,
	PiecesFreedom,
	PiecesSumAndFreedom { sum_weight: f, freedom_weight: f },
}
impl AlgoPlayer {
	pub fn select_move(self, board: &Board, rng: &mut ThreadRng) -> ChessMove {
		use AlgoPlayer::*;
		match self {
			RandomMover => select_random_move(board, rng),
			PiecesSum => todo!(),
			PiecesFreedom => todo!(),
			PiecesSumAndFreedom { sum_weight, freedom_weight } => todo!(),
		}
	}
	fn eval_board(self, board: &Board) -> f {
		use AlgoPlayer::*;
		match self {
			RandomMover => unreachable!(),
			PiecesSum => todo!(),
			PiecesFreedom => todo!(),
			PiecesSumAndFreedom { sum_weight, freedom_weight } => todo!(),
		}
	}
	pub fn get_name(self) -> String {
		use AlgoPlayer::*;
		match self {
			RandomMover => "RandomMover".to_string(),
			PiecesSum => "PiecesSum".to_string(),
			PiecesFreedom => "PiecesFreedom".to_string(),
			PiecesSumAndFreedom { sum_weight, freedom_weight } => todo!(),
		}
	}
}

fn select_random_move(board: &Board, rng: &mut ThreadRng) -> ChessMove {
	let moves = MoveGen::new_legal(board);
	let moves = moves.into_iter().collect::<Vec<_>>();
	let random_move_index = rng.random_range(0..moves.len());
	let random_move = moves[random_move_index];
	random_move
}





#[repr(u8)]
enum ComputeUnit {
	CpuOne,
	Cpu(u32),
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

