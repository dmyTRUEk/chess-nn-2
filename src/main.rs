//! chess-nn 2

#![feature(
	iter_array_chunks,
	vec_from_fn,
)]

#![deny(
	unreachable_patterns,
	unused_must_use,
	unused_results,
	unused_variables,
	clippy::let_unit_value,
	clippy::match_overlapping_arm,
	clippy::unusual_byte_groupings,
)]

#![allow(
	clippy::just_underscores_and_digits,
	clippy::let_and_return,
	clippy::manual_is_multiple_of,
	clippy::map_flatten,
	clippy::print_literal,
	clippy::upper_case_acronyms,
)]

use std::{cmp::Ordering, path::{Path, PathBuf}, str::FromStr, time::Instant};

use chess::{ALL_SQUARES, Action, Board, BoardBuilder, ChessMove, Color, Game, GameResult, MoveGen, Piece};
use chrono::Local;
use itertools::Itertools;
use nalgebra::{DMatrix, DVector};
use num_enum::{IntoPrimitive, TryFromPrimitive};
use rand::{RngExt, SeedableRng, rng, rngs::StdRng, seq::SliceRandom};
use rayon::iter::{ParallelBridge, ParallelIterator};

mod activation_fns;
mod extensions;
mod math;
mod math_aliases;
mod typesafe_rng;
mod utils_io;

use activation_fns::*;
use extensions::*;
use math::*;
use math_aliases::*;
use typesafe_rng::*;
use utils_io::*;



mod training_default {
	use super::*;

	pub const NNS_NUMBER: u32 = 30; // it's better be multiple of number of cores/threads on your machine? or else...
	pub const KEEP_TOP_NNS_FRAC: f = 2.;
	pub const EPOCHS: u32 = 100;
	pub const SAVE_EVERY_N_EPOCHS: u32 = 1;

	pub const PLAY_GAME_MOVES_LIMIT: u32 = 200;

	pub const EVOLUTION_RATE_INIT: f = 0.9;
	pub const EVOLUTION_RATE_FINAL: f = 0.001;

	pub const WIN_BY_POINTS_K: f = 1. / 10.;

	pub const DRAW_BY_POINTS_K: f = 0.999;
	// pub const DRAW_BY_POINTS_K: f = 1.;

	// pub const ALGO_WEIGHTS_CLAMP: f = 1.;
	pub const ALGO_WEIGHTS_CLAMP: f = 999.;

	// pub const CHESS_NN_THINK_DEPTH_FOR_TRAINING: u8 = 1;
	// pub const CHESS_NN_THINK_DEPTH_VS_HUMAN: u8 = 3; // 4 if parallel
}

pub const DEFAULT_RATING: f = 1_000.;

mod nn_default {
	use super::*;

	pub const INNER_LAYERS_SIZES: &[u32] = &[1000, 500, 300];
	// pub const INNER_LAYERS_SIZES: &[u32] = &[1000, 300, 100, 30, 10, 5];
	// pub const INNER_LAYERS_SIZES: &[u32] = &[300, 100, 30, 10, 5];
	// pub const INNER_LAYERS_SIZES: &[u32] = &[100, 30, 10, 5];
	// pub const INNER_LAYERS_SIZES: &[u32] = &[30, 10, 5]; // for tests

	pub const EXTRA_NOISE_INPUT: bool = false;
	// pub const NUMBER_OF_DEPTH_CHANNELS: NumberOfDepthChannels = NumberOfDepthChannels::Two;
	// pub const NUMBER_OF_DEPTH_CHANNELS: NumberOfDepthChannels = NumberOfDepthChannels::Three { use_opposite_signs: false };
	pub const NUMBER_OF_DEPTH_CHANNELS: NumberOfDepthChannels = NumberOfDepthChannels::Four;
	pub const NUMBER_OF_DIFFERENT_CHESS_PIECES: u32 = chess::NUM_PIECES as u32; // 6
	pub const NUMBER_OF_SQUARES_ON_CHESS_BOARD: u32 = chess::NUM_SQUARES as u32; // 64
	pub const INPUT_SIZE_PER_DEPTH_CHANNEL: u32 = NUMBER_OF_DIFFERENT_CHESS_PIECES * NUMBER_OF_SQUARES_ON_CHESS_BOARD; // 384
	pub const INPUT_SIZE: u32 = NUMBER_OF_DEPTH_CHANNELS.to_u32() * INPUT_SIZE_PER_DEPTH_CHANNEL; // 768 or 1152 or 1536
	pub const OUTPUT_SIZE: u32 = 1;

	// TODO(refactor): move outside?
	pub const COMPUTE_UNIT_STR: &str = "cpuall";

	pub const W_MIN: f = -1.;
	pub const W_MAX: f =  1.;
	// pub const S_MIN: f = -10.;
	// pub const S_MAX: f =  10.;
}

pub const NN_FILE_FORMAT_EXT: &str = "nn";
pub const NN_FILE_FORMAT_MAGIC: u64 = 0x_066262e3_145aaa67;





#[derive(Debug)]
enum Task {
	Train,
	Inspect,
	Play,
}

#[derive(Debug)]
struct ConfigTrain {
	compute_unit: ComputeUnit,
	create_nn_from: CreateNNFrom,
	nns_number: u32,
	keep_top_nns_frac: f,
	epochs: u32,
	save_every_n_epochs: Option<u32>,
	play_game_moves_limit: u32,
	evolution_rate_init: f,
	evolution_rate_final: f,
	win_by_points_k: f,
	draw_by_points_k: f,
	algo_weights_clamp: f,
}
#[derive(Debug)]
enum CreateNNFrom {
	LoadFile(PathBuf),
	InnerLayersSizes(Vec<u32>),
}





fn main() {
	let timestamp_begin = Instant::now();

	debug_assert_eq!(1, nn_default::OUTPUT_SIZE);

	let mut rng = rng();

	// TODO: print all params

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

	let task: Task = loop {
		break match prompt_with_name_and_default("Task (Train, Inspect, Play)", "train".to_string()).as_str() {
			"t" | "train" => Task::Train,
			"i" | "inspect" => Task::Inspect,
			"p" | "play" => Task::Play,
			_ => continue
		}
	};

	fn load_nns_files() -> Vec<PathBuf> {
		let mut nns_files = vec![];
		for entry in std::fs::read_dir(".").unwrap() {
			let path = entry.unwrap().path();
			if path.is_file() && path.extension().is_some_and(|ext| ext == NN_FILE_FORMAT_EXT) && path.file_stem().is_some() {
				nns_files.push(path);
			}
		}
		nns_files
	}

	match task {
		Task::Train => { /* continue */ }
		Task::Inspect => {
			// let load_nn_from_file: bool = loop {
			// 	break match prompt_with_name_and_default("Load NN from file (Yes/No)", "yes".to_string()).as_str() {
			// 		"y" | "yes" => true,
			// 		"n" | "no" => false,
			// 		_ => continue
			// 	}
			// };
			// TODO(refactor): f2f6ee
			let mut nns_files = load_nns_files();
			if nns_files.is_empty() { unimplemented!() }
			nns_files.sort();
			for (i, nn_file) in nns_files.iter().enumerate() {
				println!("{i}. {}", nn_file.file_stem().unwrap().display());
			}
			let index_of_nn_to_load = prompt_with_name_and_default("Index of NN to load", nns_files.len()-1);
			let filename = &nns_files[index_of_nn_to_load];
			let nn = NN::load_from_file(filename);
			println!();
			println!("Loaded NN's hash: {}", nn.calc_hash_to_string());
			println!();
			println!("all layer sizes: {:?}", nn.get_all_layers_sizes());
			println!();
			println!("activation fns: {:#?}", nn.get_activation_fns());
			// TODO: more?
			return
		}
		Task::Play => {
			let human_name: String = prompt_with_name_and_default("Your name", "human".to_string());
			let player_human = Player::Human { name: human_name };

			enum PlayAgainst { NN, Algo }
			let play_against: PlayAgainst = loop {
				break match prompt_with_name_and_default("Play against (NN, Algo)", "nn".to_string()).as_str() {
					"n" | "nn" => PlayAgainst::NN,
					"a" | "algo" => PlayAgainst::Algo,
					_ => continue
				}
			};
			let player_bot: Player = match play_against {
				PlayAgainst::NN => {
					// TODO(refactor): f2f6ee
					let mut nns_files = load_nns_files();
					if nns_files.is_empty() { unimplemented!() }
					nns_files.sort();
					for (i, nn_file) in nns_files.iter().enumerate() {
						println!("{i}. {}", nn_file.file_stem().unwrap().display());
					}
					let index_of_nn_to_load = prompt_with_name_and_default("Index of NN to load", nns_files.len()-1);
					let filename = &nns_files[index_of_nn_to_load];
					let nn = NN::load_from_file(filename);
					println!();
					println!("Loaded NN's hash: {}", nn.calc_hash_to_string());
					println!();
					Player::NN(nn)
				}
				PlayAgainst::Algo => {
					#[derive(Debug, TryFromPrimitive)]
					#[repr(u8)]
					enum Algo {
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
						Mix,
						MixUnderSignedSqrt,
					}
					let mut algos_n = 0;
					while Algo::try_from_primitive(algos_n).is_ok() {
						algos_n += 1;
					}
					assert_eq!(16, algos_n);
					for i in 0..algos_n {
						let algo = Algo::try_from_primitive(i).unwrap();
						println!("{i}. {algo:?}");
					}
					println!();
					let algo: Algo = loop {
						break match prompt_str("Algo Number: ") {
							n if let Ok(n) = n.parse() && n < algos_n => Algo::try_from_primitive(n).unwrap(),
							_ => continue
						}
					};
					println!();
					let algo_player: AlgoPlayer = match algo {
						Algo::RandomMover => AlgoPlayer::RandomMover,
						Algo::MiddleMover => AlgoPlayer::MiddleMover,
						Algo::MaterialDelta => AlgoPlayer::MaterialDelta,
						Algo::MaterialDeltaSTM => AlgoPlayer::MaterialDeltaSTM,
						Algo::NegMaterialDelta => AlgoPlayer::NegMaterialDelta,
						Algo::NegMaterialDeltaSTM => AlgoPlayer::NegMaterialDeltaSTM,
						Algo::PiecesFreedom => AlgoPlayer::PiecesFreedom,
						Algo::PiecesFreedomSTM => AlgoPlayer::PiecesFreedomSTM,
						Algo::NegPiecesFreedom => AlgoPlayer::NegPiecesFreedom,
						Algo::NegPiecesFreedomSTM => AlgoPlayer::NegPiecesFreedomSTM,
						Algo::PiecesFreedomDiff => AlgoPlayer::PiecesFreedomDiff,
						Algo::PiecesFreedomDiffSTM => AlgoPlayer::PiecesFreedomDiffSTM,
						Algo::NegPiecesFreedomDiff => AlgoPlayer::NegPiecesFreedomDiff,
						Algo::NegPiecesFreedomDiffSTM => AlgoPlayer::NegPiecesFreedomDiffSTM,
						Algo::Mix | Algo::MixUnderSignedSqrt => {
							println!("Info: StM := score * side_to_move (1 or -1)");
							println!();
							let random_mover = prompt_with_name_and_default("Random Mover", 0.);
							let material_delta = prompt_with_name_and_default("Material Delta", 0.);
							let material_delta_stm = prompt_with_name_and_default("Material Delta StM", 0.);
							let neg_material_delta = prompt_with_name_and_default("Neg Material Delta", 0.);
							let neg_material_delta_stm = prompt_with_name_and_default("Neg Material Delta StM", 0.);
							let pieces_freedom = prompt_with_name_and_default("Pieces Freedom", 0.);
							let pieces_freedom_stm = prompt_with_name_and_default("Pieces Freedom StM", 0.);
							let neg_pieces_freedom = prompt_with_name_and_default("Neg Pieces Freedom", 0.);
							let neg_pieces_freedom_stm = prompt_with_name_and_default("Neg Pieces Freedom StM", 0.);
							let pieces_freedom_diff = prompt_with_name_and_default("Pieces Freedom Diff", 0.);
							let pieces_freedom_diff_stm = prompt_with_name_and_default("Pieces Freedom Diff StM", 0.);
							let neg_pieces_freedom_diff = prompt_with_name_and_default("Neg Pieces Freedom Diff", 0.);
							let neg_pieces_freedom_diff_stm = prompt_with_name_and_default("Neg Pieces Freedom Diff StM", 0.);
							println!();
							let algo_player_mix = AlgoPlayerMix { random_mover, material_delta, material_delta_stm, neg_material_delta, neg_material_delta_stm, pieces_freedom, pieces_freedom_stm, neg_pieces_freedom, neg_pieces_freedom_stm, pieces_freedom_diff, pieces_freedom_diff_stm, neg_pieces_freedom_diff, neg_pieces_freedom_diff_stm };
							match algo {
								Algo::Mix => AlgoPlayer::Mix(algo_player_mix),
								Algo::MixUnderSignedSqrt => AlgoPlayer::MixUnderSignedSqrt(algo_player_mix),
								_ => unreachable!()
							}
						}
					};
					Player::Algo(algo_player)
				}
			};

			// let think_depth: u32 = prompt_with_name_and_default("Think Depth", 1);

			let side_to_play: Color = loop {
				break match prompt_with_name_and_default("Side to play (White/Black)", "white".to_string()).as_str() {
					"w" | "white" => Color::White,
					"b" | "black" => Color::Black,
					_ => continue
				}
			};
			let (player_white, player_black) = match side_to_play {
				Color::White => (player_human, player_bot),
				Color::Black => (player_bot, player_human),
			};

			let moves_limit = prompt_with_name_and_default("Play game moves limit", training_default::PLAY_GAME_MOVES_LIMIT);

			let (game_result, Some(game)) = play_game(&player_white, &player_black, PlayGameParams {
				moves_limit,
				print_players_moves_scores: true,
				get_game: true,
			}) else { unreachable!() };
			println!();
			println!("{}", board_to_human_viewable(&game.current_position(), BoardToHumanViewableConfig::all()));
			println!();
			println!("Game Result: {game_result:?}");
			println!();
			println!("moves: {}", game.to_uci());
			println!();
			return
		}
	}

	let config: ConfigTrain = {
		let compute_unit: ComputeUnit = loop {
			break match prompt_with_name_and_default("Compute unit (cpu<n>, cpuall, gpu)", nn_default::COMPUTE_UNIT_STR.to_string()).as_str() {
				// TODO(refactor): extract into `ComputeUnit::from_str`
				"cpu1" | "cpuone" => ComputeUnit::CpuOne,
				"cpuall" => ComputeUnit::CpuAll,
				"gpu" => ComputeUnit::Gpu,
				cpun if cpun.starts_with("cpu") => if let Ok(n) = cpun[3..].parse() { ComputeUnit::CpuN(n) } else { continue },
				_ => continue
			}
		};
		let load_nn_from_file: bool = loop {
			break match prompt_with_name_and_default("Load NN from file (Yes/No)", "yes".to_string()).as_str() {
				"y" | "yes" => true,
				"n" | "no" => false,
				_ => continue
			}
		};
		fn prompt_inner_layers_sizes() -> Vec<u32> {
			let inner_layers_sizes_default = nn_default::INNER_LAYERS_SIZES.iter().join(" ");
			loop {
				let input = prompt_with_name_and_default("NN's inner layers sizes", inner_layers_sizes_default.clone());
				let inner_layers_sizes = input.split(" ").map(|s| s.parse()).collect();
				if let Ok(inner_layers_sizes) = inner_layers_sizes { return inner_layers_sizes }
			}
		}
		let create_nn_from: CreateNNFrom = if load_nn_from_file {
			// TODO(feat): load multiple NNs (multi nn archs support)
			// TODO(refactor): f2f6ee
			let mut nns_files = load_nns_files();
			if !nns_files.is_empty() {
				nns_files.sort();
				for (i, nn_file) in nns_files.iter().enumerate() {
					println!("{i}. {}", nn_file.file_stem().unwrap().display());
				}
				let index_of_nn_to_load = prompt_with_name_and_default("Index of NN to load", nns_files.len()-1);
				CreateNNFrom::LoadFile(nns_files[index_of_nn_to_load].clone())
			} else {
				println!("No `.nn` files found, falling back to creating from NN's inner layers sizes...");
				CreateNNFrom::InnerLayersSizes(prompt_inner_layers_sizes())
			}
		} else {
			CreateNNFrom::InnerLayersSizes(prompt_inner_layers_sizes())
		};
		let nns_number = prompt_with_name_and_default("NNs number", training_default::NNS_NUMBER);
		let keep_top_nns_frac = prompt_with_name_and_default("Keep top N NNs fraction", training_default::KEEP_TOP_NNS_FRAC);
		let epochs = prompt_with_name_and_default("Epochs", training_default::EPOCHS);
		let save_every_n_epochs = prompt_with_name_and_default("Save every N epochs (0 for never)", training_default::SAVE_EVERY_N_EPOCHS);
		let save_every_n_epochs = (save_every_n_epochs != 0).then_some(save_every_n_epochs);
		let play_game_moves_limit = prompt_with_name_and_default("Play game moves limit", training_default::PLAY_GAME_MOVES_LIMIT);
		let evolution_rate_init = prompt_with_name_and_default("Evolution rate init", training_default::EVOLUTION_RATE_INIT);
		let evolution_rate_final = prompt_with_name_and_default("Evolution rate final", training_default::EVOLUTION_RATE_FINAL);
		let win_by_points_k = prompt_with_name_and_default("Win by points K", training_default::WIN_BY_POINTS_K);
		let draw_by_points_k = prompt_with_name_and_default("Draw by points K", training_default::DRAW_BY_POINTS_K);
		// let algo_weights_clamp = prompt_with_name_and_default("Algo weights clamp", training_default::ALGO_WEIGHTS_CLAMP);
		let algo_weights_clamp = training_default::ALGO_WEIGHTS_CLAMP;
		ConfigTrain {
			compute_unit,
			create_nn_from,
			nns_number,
			keep_top_nns_frac,
			epochs,
			save_every_n_epochs,
			play_game_moves_limit,
			evolution_rate_init,
			evolution_rate_final,
			win_by_points_k,
			draw_by_points_k,
			algo_weights_clamp,
		}
	};

	println!();
	println!("Starting with config: {config:#?}");
	println!();

	if let ComputeUnit::CpuN(n) = config.compute_unit {
		rayon::ThreadPoolBuilder::new()
			.num_threads(n as usize)
			.build_global()
			.unwrap();
	}

	// TODO(feat): remove `inner_layers_sizes` and use parents params for evo/gen (needed for multi nn archs support)
	let (inner_layers_sizes, nns) = match config.create_nn_from {
		CreateNNFrom::InnerLayersSizes(inner_layers_sizes) => {
			(inner_layers_sizes.clone(), Vec::from_fn(config.nns_number as usize, |_i| {
				NN::new_random(&inner_layers_sizes, &mut rng)
			}))
		}
		CreateNNFrom::LoadFile(filename) => {
			let nn = NN::load_from_file(&filename);
			println!("Loaded NN's hash: {}", nn.calc_hash_to_string());
			// TODO: why slow?
			(nn.get_inner_layers_sizes(), Vec::from_fn(config.nns_number as usize, |i| {
				if i == 0 { nn.clone() } else { nn.clone().evolved(config.evolution_rate_init, &mut rng) }
			}))
		}
	};
	assert_eq!(config.nns_number, nns.len() as u32);
	println!("Created {} NNs", config.nns_number);
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
	let evolution_rate_drop_speed = ln(config.evolution_rate_init / config.evolution_rate_final);

	for epoch in 0..config.epochs {
		println!();
		println!("{dashes} EPOCH {}/{} {dashes}", epoch+1, config.epochs, dashes="-".repeat(42));
		println!();

		for PlayerWithRatingAndStats { player: _, rating, stats } in players.iter_mut() {
			*rating = DEFAULT_RATING;
			*stats = PlayerInTournamentStats::new();
		}

		players.shuffle(&mut rng);

		play_tournament(&mut players, &mut rng, TournamentParams {
			moves_limit: config.play_game_moves_limit,
			update_ratings_params: UpdateRatingsParams { win_by_points_k: config.win_by_points_k, draw_by_points_k: config.draw_by_points_k },
			compute_unit: config.compute_unit,
		});

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

		{
			let play_game_params = PlayGameParams {
				moves_limit: config.play_game_moves_limit,
				print_players_moves_scores: false,
				get_game: true,
			};
			{ // best NN vs best
				let white = &players.iter().find(|p| p.player.is_nn()).unwrap().player;
				let black = &players[0].player;
				let (game_result, Some(game)) = play_game(white, black, play_game_params) else { unreachable!() };
				println!("best NN vs best ({}):   {}", game_result.to_char(), game.to_uci());
			}
			println!();
			{ // best NN vs self
				let white = &players.iter().find(|p| p.player.is_nn()).unwrap().player;
				let black = white;
				let (game_result, Some(game)) = play_game(white, black, play_game_params) else { unreachable!() };
				println!("best NN vs self ({}):   {}", game_result.to_char(), game.to_uci());
			}
			println!();
			{ // best NN vs second best NN
				let [white, black] = players.iter()
					.filter(|p| p.player.is_nn())
					.k_largest_by(2, |p1, p2| p1.rating.partial_cmp(&p2.rating).unwrap())
					.map(|p| &p.player)
					.collect::<Vec<_>>()[..] else { unreachable!() };
				let (game_result, Some(game)) = play_game(white, black, play_game_params) else { unreachable!() };
				println!("best NN vs second best NN ({}):   {}", game_result.to_char(), game.to_uci());
			}
			println!();
			{ // best NN vs worst NN
				let white = &players.iter().find(|p| p.player.is_nn()).unwrap().player;
				let black = &players.iter().rev().find(|p| p.player.is_nn()).unwrap().player;
				let (game_result, Some(game)) = play_game(white, black, play_game_params) else { unreachable!() };
				println!("best NN vs worst NN ({}):   {}", game_result.to_char(), game.to_uci());
			}
			println!();
			{ // best NN vs worst
				let white = &players.iter().find(|p| p.player.is_nn()).unwrap().player;
				let black = &players[players.len()-1].player;
				let (game_result, Some(game)) = play_game(white, black, play_game_params) else { unreachable!() };
				println!("best NN vs worst ({}):   {}", game_result.to_char(), game.to_uci());
			}
		}

		println!();

		// TODO(optim): parallel evolution
		let evolution_rate = config.evolution_rate_init * exp(-evolution_rate_drop_speed * (epoch as f) / (config.epochs as f - 1.));
		macro_rules! get_random_evo_rate { () => {{
			let k: f = rng.random_range(1. .. 10.);
			let evo_rate = evolution_rate * if rng.random_bool(0.5) { k } else { k.recip() };
			let evo_rate = evo_rate.clamp(0., 1.);
			evo_rate
		}}; }
		println!("evolving with evo_rate = {evolution_rate:.4} ...");
		let players_n = players.len();
		{ // evolve and natsel: nns
			let index_of_best_nn = players.iter().position(|p| p.player.is_nn()).unwrap();
			let nns_n = players.iter().filter(|p| p.player.is_nn()).count();
			let keep_top_n_nns = ((nns_n as f) / config.keep_top_nns_frac).round() as usize;
			let mut nns_i = 0;
			for i in 0..players_n {
				if !players[i].player.is_nn() { continue }
				nns_i += 1;
				if nns_i < keep_top_n_nns { continue }
				let player_to_clone = &players[i - keep_top_n_nns];
				if player_to_clone.player.is_nn() {
					players[i] = player_to_clone.clone();
				} else {
					if rng.random_bool(0.1) {
						players[i] = PlayerWithRatingAndStats::new(Player::NN(NN::new_random(&inner_layers_sizes, &mut rng)));
						continue // dont evolve
					} else {
						players[i] = players[index_of_best_nn].clone();
						// TODO(feat): mix two nns
					}
				}
				// evolution:
				players[i].player.evolve(get_random_evo_rate!(), &mut rng, config.algo_weights_clamp);
			}
			// and (at least) one new random nn:
			for i in (keep_top_n_nns..players_n).rev() {
				if players[i].player.is_nn() {
					// println!("reseting `player[{i}]`");
					players[i] = PlayerWithRatingAndStats::new(Player::NN(NN::new_random(&inner_layers_sizes, &mut rng)));
					break
				}
			}
		}
		{ // evolve and natsel: algo mix
			let index_of_best_algo_mix = players.iter()
				.position(|p| p.player.is_algo_mix())
				.unwrap();
			for i in 0..players_n {
				if !players[i].player.is_algo_mix() { continue }
				if i == index_of_best_algo_mix { continue }
				if players[i].rating < DEFAULT_RATING {
					if rng.random_bool(0.1) {
						players[i] = PlayerWithRatingAndStats::new(Player::Algo(AlgoPlayer::mix_new_random(&mut rng)));
						continue // dont evolve
					} else {
						players[i] = players[index_of_best_algo_mix].clone();
					};
				}
				players[i].player.evolve(get_random_evo_rate!(), &mut rng, config.algo_weights_clamp);
			}
		}
		{ // evolve and natsel: algo mix uss
			let index_of_best_algo_mix_uss = players.iter()
				.position(|p| p.player.is_algo_mix_uss())
				.unwrap();
			for i in 0..players_n {
				if !players[i].player.is_algo_mix_uss() { continue }
				if i == index_of_best_algo_mix_uss { continue }
				if players[i].rating < DEFAULT_RATING {
					if rng.random_bool(0.1) {
						players[i] = PlayerWithRatingAndStats::new(Player::Algo(AlgoPlayer::mix_uss_new_random(&mut rng)));
						continue // dont evolve
					} else {
						players[i] = players[index_of_best_algo_mix_uss].clone();
					};
				}
				players[i].player.evolve(get_random_evo_rate!(), &mut rng, config.algo_weights_clamp);
			}
		}

		assert_eq!(players_n_init, players.len());

		if let Some(save_every_n_epochs) = config.save_every_n_epochs && (epoch+1).is_multiple_of(save_every_n_epochs) {
			// TODO(feat): save not only best NN, best top N NNs
			let best_nn = players.iter().find(|p| p.player.is_nn()).unwrap();
			let Player::NN(best_nn) = &best_nn.player else { unreachable!() };
			let now = Local::now().format("%Y-%m-%d_%H-%M-%S");
			let hash = best_nn.calc_hash_to_string();
			let inner_layers_sizes = best_nn.get_inner_layers_sizes().into_iter().map(|s| s.to_string()).join("_");
			let filename = format!("{now}__{hash}__{inner_layers_sizes}.{NN_FILE_FORMAT_EXT}");
			best_nn.save_to_file(&filename);
			// TODO(feat): print "saved" or something
		}

		println!();
		println!();
	}
	println!();

	// TODO: print all params?

	let timestamp_end = Instant::now();
	let time_spent = timestamp_end.duration_since(timestamp_begin);
	println!("time spent: {:.1}s", time_spent.as_secs_f64());

	todo!("cli/repl to interact (play,inspect,save,etc) with nns/algos");
}



#[derive(Debug, Clone, Copy)]
struct TournamentParams {
	moves_limit: u32,
	update_ratings_params: UpdateRatingsParams,
	compute_unit: ComputeUnit,
}

fn play_tournament(
	players: &mut [PlayerWithRatingAndStats],
	rng: &mut impl RngExt,
	params: TournamentParams,
) {
	let players_n = players.len();
	let play_game_params = PlayGameParams {
		moves_limit: params.moves_limit,
		print_players_moves_scores: false,
		get_game: false,
	};
	match params.compute_unit {
		ComputeUnit::CpuOne => {
			let mut games: Vec<(usize, usize, GameResult_)> = vec![];
			print("games results: ");
			for white_i in 0..players_n {
				for black_i in 0..players_n {
					if white_i == black_i { continue }
					let [white, black] = players.get_disjoint_mut([white_i, black_i]).unwrap();
					let (game_result, _) = play_game(&white.player, &black.player, play_game_params);
					print(game_result.to_char());
					games.push((white_i, black_i, game_result));
				}
				print(" ");
			}
			println!();
			// TODO(refactor): extract?
			games.shuffle(rng);
			for (white_i, black_i, game_result) in games.into_iter() {
				if white_i == black_i { unreachable!() }
				let [white, black] = players.get_disjoint_mut([white_i, black_i]).unwrap();
				update_stats(&mut white.stats, &mut black.stats, game_result);
				update_ratings(&mut white.rating, &mut black.rating, game_result, params.update_ratings_params);
			}
		}
		ComputeUnit::CpuN(_) | ComputeUnit::CpuAll => {
			let players_ref: &[PlayerWithRatingAndStats] = &*players; // this fixes bc `&mut T` is not Copy (need for move), but `&T` is Copy (src: chatgpt)
			print("games results: ");
			let mut games: Vec<(usize, usize, GameResult_)> = (0..players_n).cartesian_product(0..players_n)
				.par_bridge()
				.map(|(white_i, black_i)| {
					if white_i == black_i { return None }
					let (game_result, _) = play_game(&players_ref[white_i].player, &players_ref[black_i].player, play_game_params);
					print(game_result.to_char());
					Some((white_i, black_i, game_result))
				})
				.flatten() // remove `None`s // TODO?: dont flatten, and when printing print `_` or something
				.collect();
			games.sort_unstable_by_key(|g| (g.0, g.1)); // TODO: recheck
			println!();
			println!();
			print!("organized: ");
			for (i, (white_i, black_i, game_result)) in games.iter().enumerate() {
				if white_i == black_i { unreachable!() }
				// println!("{white_i}, {black_i}, {game_result:?}");
				print!("{}", game_result.to_char());
				// println!("CHAR: {}", game_result.to_char());
				if i % (players_n - 1) == players_n - 2 { print(" "); }
			}
			println!();
			// TODO(refactor): extract?
			games.shuffle(rng);
			for (white_i, black_i, game_result) in games.into_iter() {
				if white_i == black_i { unreachable!() }
				let [white, black] = players.get_disjoint_mut([white_i, black_i]).unwrap();
				update_stats(&mut white.stats, &mut black.stats, game_result);
				update_ratings(&mut white.rating, &mut black.rating, game_result, params.update_ratings_params);
			}
		}
		ComputeUnit::Gpu => {
			todo!("use same as CpuOne?")
		}
	}
}



#[derive(Debug, Clone, Copy)]
struct UpdateRatingsParams {
	win_by_points_k: f,
	draw_by_points_k: f,
}

fn update_ratings(
	white: &mut f,
	black: &mut f,
	game_result: GameResult_,
	params: UpdateRatingsParams,
) {
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
			let elo_rating_delta = calc_elo_rating_delta(*white, *black) * params.win_by_points_k;
			*white += elo_rating_delta;
			*black -= elo_rating_delta;
		}
		BlackWinsByPoints => {
			let elo_rating_delta = calc_elo_rating_delta(*black, *white) * params.win_by_points_k;
			*black += elo_rating_delta;
			*white -= elo_rating_delta;
		}
		DrawByPoints => {
			// let elo_rating_delta_1 = calc_elo_rating_delta(*white, *black);
			// let elo_rating_delta_2 = calc_elo_rating_delta(*black, *white);
			// let elo_rating_delta = (elo_rating_delta_1 + elo_rating_delta_2) / 2.;
			// let elo_rating_delta = elo_rating_delta / 1000.;
			// TODO?
			*black *= params.draw_by_points_k;
			*white *= params.draw_by_points_k;
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



#[derive(Debug, Clone, Copy)]
struct PlayGameParams {
	moves_limit: u32,
	print_players_moves_scores: bool,
	get_game: bool,
}

fn play_game(white: &Player, black: &Player, params: PlayGameParams) -> (GameResult_, Option<Game>) {
	let mut rng = rng();
	let mut game = Game::new(); // TODO(optim): dont use Game, use Board directly
	let mut move_number: u32 = 0;
	while game.result().is_none() && move_number < params.moves_limit {
		move_number += 1;
		let board = game.current_position();
		let side_to_move: Color = board.side_to_move();
		let player_to_make_move = match side_to_move {
			Color::White => white,
			Color::Black => black,
		};
		let (selected_move, moves_and_scores) = player_to_make_move.select_move(&board, &mut rng, SelectMoveParams { get_moves_scores: params.print_players_moves_scores });
		if params.print_players_moves_scores {
			println!();
			if let Some(moves_and_scores) = moves_and_scores {
				for (move_, score) in moves_and_scores {
					println!("{move_}: {score}");
				}
				println!();
			} else {
				println!("Moves and Scores not provided");
			}
			println!("Selected move: {selected_move}");
		}
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
	(gr, params.get_game.then_some(game))
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





#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, IntoPrimitive, TryFromPrimitive)]
#[repr(u8)]
pub enum ActivationFn {
	// src: https://en.wikipedia.org/wiki/Activation_function#Table_of_activation_functions
	ReLU,
	LeakyReLU_01,
	LeakyReLU_001,
	Sign,
	Step,
	Sigmoid,
	Tanh,
	SoftSign,
	SoftPlus,
	ExpLU,
	SiLU,
	ELiSH,
	Gaussian,
	Clamp01,
	ReSqrt, // 0 if x < 0
	SignedSqrt,
	LeakyReSqrt01, // if x<0 => *0.1
	LeakyReSqrt001,
	SignedSqrtP1,
	ReSquare,
	SignedSquare,
	LeakyReSquare01, // if x<0 => *0.1
	LeakyReSquare001,
	Sinc,
	ReSinc,
	Softmax,
	Maxout,
	// NOTE: dont forget to add new fn to `new_random`
}
impl ActivationFn {
	// const NUMBER_OF_VARIANTS: u8 = {
	// 	let mut res = None;
	// 	let mut i: u8 = 0;
	// 	while i < u8::MAX {
	// 		if ActivationFn::try_from(i).is_err() {
	// 			res = Some(i);
	// 			break
	// 		}
	// 		i += 1;
	// 	}
	// 	todo("test");
	// 	res.unwrap()
	// };
	pub /* const */ fn get_number_of_variants() -> u8 {
		for i in 0.. {
			if ActivationFn::try_from(i).is_err() {
				return i
			}
		}
		unreachable!()
	}
	pub fn get_all_variants() -> Vec<ActivationFn> { // [ActivationFn; NUMBER_OF_VARIANTS]
		Vec::from_fn(Self::get_number_of_variants() as usize, |i| ActivationFn::try_from(i as u8).unwrap())
	}
	pub fn new_random(rng: &mut impl RngExt) -> Self {
		use ActivationFn::*;
		// assert_eq!(ActivationFn::NUMBER_OF_VARIANTS, VN::NUMBER_OF_VARIANTS);
		use V19::*;
		debug_assert_eq!(ActivationFn::get_number_of_variants(), V27::NUMBER_OF_VARIANTS);
		match rng.random_variant() {
			_1 => ReLU,
			_2 => LeakyReLU_01,
			_3 => LeakyReLU_001,
			_4 => Sign,
			_5 => Step,
			_6 => Sigmoid,
			_7 => Tanh,
			_8 => SoftSign,
			// _ => SoftPlus,
			_9 => ExpLU,
			_10 => SiLU,
			// _ => ELiSH,
			_11 => Gaussian,
			_12 => Clamp01,
			_13 => ReSqrt,
			_14 => SignedSqrt,
			_15 => LeakyReSqrt01,
			_16 => LeakyReSqrt001,
			_17 => SignedSqrtP1,
			// _ => ReSquare,
			// _ => SignedSquare,
			// _ => LeakyReSquare01,
			// _ => LeakyReSquare001,
			_18 => Sinc,
			_19 => ReSinc,
			// _ => Softmax,
			// _ => Maxout,
		}
	}
	pub fn eval(self, xs: DVector<f>) -> DVector<f> {
		use ActivationFn::*;
		match self {
			ReLU => xs.map(relu),
			LeakyReLU_01 => xs.map(leaky_relu_01),
			LeakyReLU_001 => xs.map(leaky_relu_001),
			Sign => xs.map(sign),
			Step => xs.map(step),
			Sigmoid => xs.map(sigmoid),
			Tanh => xs.map(tanh),
			SoftSign => xs.map(soft_sign),
			SoftPlus => xs.map(soft_plus),
			ExpLU => xs.map(explu),
			SiLU => xs.map(silu),
			ELiSH => xs.map(elish),
			Gaussian => xs.map(gaussian),
			Clamp01 => xs.map(clamp01),
			ReSqrt => xs.map(resqrt),
			SignedSqrt => xs.map(signed_sqrt),
			LeakyReSqrt01 => xs.map(leaky_resqrt_01),
			LeakyReSqrt001 => xs.map(leaky_resqrt_001),
			SignedSqrtP1 => xs.map(signed_sqrt_p1),
			ReSquare => xs.map(resquare),
			SignedSquare => xs.map(signed_square),
			LeakyReSquare01 => xs.map(leaky_resquare_01),
			LeakyReSquare001 => xs.map(leaky_resquare_001),
			Sinc => xs.map(sinc),
			ReSinc => xs.map(resinc),
			Softmax => {
				if xs.iter().any(|&x| abs(x) > 80. /* ln(f32::MAX) = 88.72284 */) { return Self::Maxout.eval(xs) }
				let exps = xs.map(exp);
				let exp_sum = exps.sum();
				exps / exp_sum
			}
			Maxout => {
				let index_of_max = xs.as_slice().index_of_max().unwrap();
				let mut one_hot = DVector::zeros(xs.len());
				one_hot[index_of_max] = 1.;
				one_hot
			}
		}
	}
	pub fn to_hash(self) -> u64 {
		use ActivationFn::*;
		match self {
			ReLU => 0x_39b728e5_6ca7f4a8,
			LeakyReLU_01 => 0x_807042bc_9e3e8170,
			LeakyReLU_001 => 0x_c9d5e1e8_3f02d2a7,
			Sign => 0x_a677c501_3b6fcda2,
			Step => 0x_f85d4bb4_12b20a92,
			Sigmoid => 0x_bf8bd5a3_05d3be95,
			Tanh => 0x_deba1dd1_d30ee4b3,
			SoftSign => 0x_e9067194_5163f143,
			SoftPlus => 0x_520545ee_98ca6be2,
			ExpLU => 0x_1334fd2f_27429035,
			SiLU => 0x_c4921e89_35e84654,
			ELiSH => 0x_03f8f2c8_37165a17,
			Gaussian => 0x_61424e39_bf3c44a7,
			Clamp01 => 0x_b35ef484_34e47d87,
			ReSqrt => 0x_1b969ca2_d42487d6,
			SignedSqrt => 0x_07a6a02f_2f82a958,
			LeakyReSqrt01 => 0x_a09f1663_655ca19e,
			LeakyReSqrt001 => 0x_02e4e717_3ac11f84,
			SignedSqrtP1 => 0x_2e8351d3_1cd6db2e,
			ReSquare => 0x_4193da3a_50e20467,
			SignedSquare => 0x_d588873b_771910de,
			LeakyReSquare01 => 0x_f84207c5_34b62a65,
			LeakyReSquare001 => 0x_3299d4f0_b20d7e00,
			Sinc => 0x_4a2831dc_c744401b,
			ReSinc => 0x_b5a8b8c5_52e65861,
			Softmax => 0x_f0fc8c2d_b6a422db,
			Maxout => 0x_36bc3c93_6f06f7f8,
		}
	}
	pub fn from_hash(hash: u64) -> Self {
		use ActivationFn::*;
		match hash {
			0x_39b728e5_6ca7f4a8 => ReLU,
			0x_807042bc_9e3e8170 => LeakyReLU_01,
			0x_c9d5e1e8_3f02d2a7 => LeakyReLU_001,
			0x_a677c501_3b6fcda2 => Sign,
			0x_f85d4bb4_12b20a92 => Step,
			0x_bf8bd5a3_05d3be95 => Sigmoid,
			0x_deba1dd1_d30ee4b3 => Tanh,
			0x_e9067194_5163f143 => SoftSign,
			0x_520545ee_98ca6be2 => SoftPlus,
			0x_1334fd2f_27429035 => ExpLU,
			0x_c4921e89_35e84654 => SiLU,
			0x_03f8f2c8_37165a17 => ELiSH,
			0x_61424e39_bf3c44a7 => Gaussian,
			0x_b35ef484_34e47d87 => Clamp01,
			0x_1b969ca2_d42487d6 => ReSqrt,
			0x_07a6a02f_2f82a958 => SignedSqrt,
			0x_a09f1663_655ca19e => LeakyReSqrt01,
			0x_02e4e717_3ac11f84 => LeakyReSqrt001,
			0x_2e8351d3_1cd6db2e => SignedSqrtP1,
			0x_4193da3a_50e20467 => ReSquare,
			0x_d588873b_771910de => SignedSquare,
			0x_f84207c5_34b62a65 => LeakyReSquare01,
			0x_3299d4f0_b20d7e00 => LeakyReSquare001,
			0x_4a2831dc_c744401b => Sinc,
			0x_b5a8b8c5_52e65861 => ReSinc,
			0x_f0fc8c2d_b6a422db => Softmax,
			0x_36bc3c93_6f06f7f8 => Maxout,
			_ => panic!("unknown activation function, hash: {hash}")
		}
	}
}





// TODO(refactor): make more variants
#[derive(Debug, Clone, Copy)]
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
			Three { use_opposite_signs: _ } => 3,
			Four => 4,
		}
	}
}





#[derive(Debug, Clone, Copy)]
struct SelectMoveParams {
	get_moves_scores: bool, // TODO: use?
}

trait SelectMove {
	fn select_move(&self, board: &Board, rng: &mut impl RngExt, params: SelectMoveParams) -> (ChessMove, Option<Vec<(ChessMove, f)>>);
}



#[derive(Clone)]
#[repr(u8)]
enum Player {
	NN(NN),
	// NNFromSeed { seed: u64 }, // TODO
	Algo(AlgoPlayer),
	Human { name: String },
}
impl Player {
	pub fn name(&self) -> String {
		use Player::*;
		match self {
			NN(nn) => format!("NN {}", nn.calc_hash_to_string()),
			Algo(algo) => algo.get_name(),
			Human { name } => name.clone(),
		}
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
	pub fn evolve(&mut self, evolution_rate: f, rng: &mut impl RngExt, algo_weights_clamp: f) {
		use Player::*;
		match self {
			NN(nn) => {
				nn.evolve(evolution_rate, rng);
			}
			Algo(AlgoPlayer::Mix(mix)) => {
				mix.evolve(evolution_rate, rng, algo_weights_clamp);
			}
			Algo(AlgoPlayer::MixUnderSignedSqrt(mix)) => {
				mix.evolve(evolution_rate, rng, algo_weights_clamp);
			}
			Algo(_) => {}
			Human { name: _ } => {}
		}
	}
}
impl SelectMove for Player {
	fn select_move(&self, board: &Board, rng: &mut impl RngExt, params: SelectMoveParams) -> (ChessMove, Option<Vec<(ChessMove, f)>>) {
		use Player::*;
		match self {
			NN(nn) => nn.select_move(board, rng, params),
			Algo(algo) => algo.select_move(board, rng, params),
			Human { name } => {
				println!();
				println!("{}", board_to_human_viewable(board, BoardToHumanViewableConfig::all()));
				println!();
				let all_legal_moves: Vec<ChessMove> = MoveGen::new_legal(board).collect();
				let move_: ChessMove = loop {
					let move_ = prompt_str(&format!("{name}, your move: "));
					// if move_.len() > 5 { continue } // TODO?
					let move_: ChessMove = match (ChessMove::from_str(&move_), ChessMove::from_san(board, &move_)) {
						(Ok(move_), _) => move_,
						(_, Ok(move_)) => move_,
						(Err(_), Err(_)) => { println!("Invalid move"); continue }
					};
					if all_legal_moves.contains(&move_) { break move_ } else { println!("Illegal move"); }
				};
				(move_, None)
			}
		}
	}
}

#[derive(Clone)]
struct PlayerWithRatingAndStats {
	player: Player,
	rating: f,
	stats: PlayerInTournamentStats,
	// TODO(feat): add global number of wins in tournaments
}
impl PlayerWithRatingAndStats {
	pub fn new(player: Player) -> PlayerWithRatingAndStats {
		Self { player, rating: DEFAULT_RATING, stats: PlayerInTournamentStats::new() }
	}
}
// #[derive(Debug, Clone, Copy)]
// struct Rating(f);
// impl Rating { fn new() -> Self { Self(training::DEFAULT_RATING) } }





pub struct NNSpec {
	inner_layers_sizes: Vec<u32>,
	activation_fns: Vec<ActivationFn>,
}
impl NNSpec {
	pub fn new(inner_layers_sizes: Vec<u32>, activation_fns: Vec<ActivationFn>) -> Self {
		Self { inner_layers_sizes, activation_fns }
	}
}



#[derive(Clone, PartialEq)]
struct NN { layers: Vec<NNLayer> }
impl NN {
	pub fn new_random(inner_layers_sizes: &[u32], rng: &mut impl RngExt) -> Self {
		let all_layers_sizes = [&[nn_default::INPUT_SIZE], inner_layers_sizes, &[nn_default::OUTPUT_SIZE]].concat();
		Self {
			layers: all_layers_sizes
				.array_windows()
				.cloned()
				.map(|[size_in, size_out]| NNLayer::new_random(size_in, size_out, rng))
				.collect()
		}
	}
	// pub fn new_random_with_seed(inner_layers_sizes: &[u32], seed: u64) -> Self {
	// 	let mut rng = StdRng::seed_from_u64(seed);
	// 	Self::new_random(inner_layers_sizes, &mut rng)
	// }
	// pub fn new_random_from_spec_with_seed(spec: NNSpec, seed: u64) -> Self {
	// 	let mut rng = StdRng::seed_from_u64(seed);
	// 	Self::new_random_from_spec(spec, &mut rng)
	// }
	// pub fn new_random_from_spec(spec: NNSpec, rng: &mut impl RngExt) -> Self {
	// 	let all_layers_sizes = [&[nn_default::INPUT_SIZE], spec.inner_layers_sizes.as_slice(), &[nn_default::OUTPUT_SIZE]].concat();
	// 	Self {
	// 		layers: all_layers_sizes
	// 			.array_windows()
	// 			.cloned()
	// 			.zip_eq(spec.activation_fns)
	// 			.map(|([size_in, size_out], af)| NNLayer::new_random_from_spec(size_in, size_out, af, rng))
	// 			.collect()
	// 	}
	// }
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
		debug_assert_eq!(nn_default::OUTPUT_SIZE, v.len() as u32);
		v[0]
	}
	pub fn evolved(mut self, evolution_rate: f, rng: &mut impl RngExt) -> Self {
		self.evolve(evolution_rate, rng);
		self
	}
	pub fn evolve(&mut self, evolution_rate: f, rng: &mut impl RngExt) {
		for layer in self.layers.iter_mut() {
			layer.evolve(evolution_rate, rng);
		}
	}
	// TODO(refactor)?: rename?
	pub fn calc_hash_to_string(&self) -> String {
		let hash: u64 = self.calc_hash();
		hash_to_string(hash)
	}
	fn calc_hash(&self) -> u64 {
		// TODO(refactor): use MyHash
		let mut hash: u64 = 0x_1e88d6f0_b31da73f;
		for layer in self.layers.iter() {
			hash ^= layer.calc_hash();
		}
		hash
	}
	pub fn get_all_layers_sizes(&self) -> Vec<u32> {
		let mut all_layers_sizes = self.get_inner_layers_sizes();
		all_layers_sizes.insert(0, self.layers[0].weights.ncols() as u32);
		all_layers_sizes.push(self.layers.last().unwrap().biases.len() as u32);
		// dbg!(&all_layers_sizes);
		assert_eq!(nn_default::INPUT_SIZE, all_layers_sizes[0]);
		assert_eq!(nn_default::OUTPUT_SIZE, *all_layers_sizes.last().unwrap());
		all_layers_sizes
	}
	pub fn get_inner_layers_sizes(&self) -> Vec<u32> {
		let mut inner_layers_sizes: Vec<u32> = self.layers.iter().map(|l| l.weights.ncols() as u32).collect();
		let _input_size = inner_layers_sizes.remove(0);
		// dbg!(&inner_layers_sizes);
		inner_layers_sizes
	}
	pub fn get_activation_fns(&self) -> Vec<ActivationFn> {
		self.layers.iter().map(|l| l.activation_fn).collect()
	}
	pub fn save_to_file(&self, filename: &str) {
		std::fs::write(filename, self.to_bytes()).unwrap()
	}
	pub fn load_from_file(filename: &Path) -> Self {
		Self::from_bytes(&std::fs::read(filename).unwrap())
	}
	pub fn to_bytes(&self) -> Vec<u8> {
		// my nn file format:
		// [ magic (8 bytes) ]
		// [ hash of nn (u64) ]
		// all layer sizes: [ NL, array len (u32) ] [ u32 * (NL+1) ]
		// activations fns hashes: [ u64 * (NL) ]   // -1 bc first layer/size is not a layer, its an input
		// layer 0:
		//   [ NB1, number of biases (u32) ] [ f32 * NB1 ]   // yes, number of them is redundant
		//   [ NW1, number of weights (u32) ] [ f32 * NW1 ]
		// ...
		// layer NL-1/-2?:
		//   ...
		let layers_n: u32 = self.layers.len() as _;
		let all_layers_sizes = self.get_all_layers_sizes();
		assert_eq!(layers_n+1, all_layers_sizes.len() as u32);
		let activation_fns = self.get_activation_fns();
		assert_eq!(layers_n, activation_fns.len() as u32);
		let mut file_parts: Vec<Vec<u8>> = vec![
			NN_FILE_FORMAT_MAGIC.to_le_bytes().to_vec(),
			self.calc_hash().to_le_bytes().to_vec(),
			layers_n.to_le_bytes().to_vec(),
			all_layers_sizes.iter().flat_map(|ls| ls.to_le_bytes()).collect(),
			activation_fns.into_iter().flat_map(|af| af.to_hash().to_le_bytes()).collect()
		];
		for layer in self.layers.iter() {
			file_parts.push((layer.biases.len() as u32).to_le_bytes().to_vec());
			file_parts.push(layer.biases.iter().flat_map(|b| b.to_le_bytes()).collect());
			file_parts.push((layer.weights.len() as u32).to_le_bytes().to_vec());
			file_parts.push(layer.weights.iter().flat_map(|w| w.to_le_bytes()).collect());
		}
		file_parts.concat()
	}
	pub fn from_bytes(bytes: &[u8]) -> Self {
		// my nn file format:
		// [ magic (8 bytes) ]
		// [ hash of nn (u64) ]
		// all layer sizes: [ NL, array len (u32) ] [ u32 * (NL+1) ]
		// activations fns hashes: [ u64 * (NL) ]   // -1 bc first layer/size is not a layer, its an input
		// layer 0:
		//   [ NB1, number of biases (u32) ] [ f32 * NB1 ]   // yes, number of them is redundant
		//   [ NW1, number of weights (u32) ] [ f32 * NW1 ]
		// ...
		// layer NL-1/-2?:
		//   ...
		let magic = &bytes[..8];
		assert_eq!(NN_FILE_FORMAT_MAGIC.to_le_bytes(), magic);
		let bytes = &bytes[8..];
		let hash = u64::from_le_bytes(bytes[..8].try_into().unwrap());
		let bytes = &bytes[8..];
		let layers_n = u32::from_le_bytes(bytes[..4].try_into().unwrap());
		let bytes = &bytes[4..];
		let all_layers_sizes: Vec<u32> = bytes[..4*(layers_n+1) as usize]
			.iter().cloned()
			.array_chunks()
			.map(u32::from_le_bytes)
			.collect();
		let bytes = &bytes[4*(layers_n+1) as usize..];
		let activation_fns: Vec<ActivationFn> = bytes[..(8*layers_n) as usize]
			.iter().cloned()
			.array_chunks()
			.map(u64::from_le_bytes)
			.map(ActivationFn::from_hash)
			.collect();
		let mut bytes = &bytes[(8*layers_n) as usize..];
		let mut layers = vec![];
		for (activation_fn, [size_in, size_out]) in activation_fns.iter().cloned().zip_eq(all_layers_sizes.array_windows()) {
			let biases_n = u32::from_le_bytes(bytes[..4].try_into().unwrap());
			bytes = &bytes[4..];
			let biases: Vec<f> = bytes[..(4*biases_n) as usize]
				.iter().cloned()
				.array_chunks()
				.map(f32::from_le_bytes)
				.collect();
			bytes = &bytes[(4*biases_n) as usize..];
			let biases = DVector::from(biases);
			let weights_n = u32::from_le_bytes(bytes[..4].try_into().unwrap());
			bytes = &bytes[4..];
			let weights = bytes[..(4*weights_n) as usize]
				.iter().cloned()
				.array_chunks()
				.map(f32::from_le_bytes);
			bytes = &bytes[(4*weights_n) as usize..];
			let weights = DMatrix::from_iterator(*size_out as usize, *size_in as usize, weights);
			layers.push(NNLayer { weights, biases, activation_fn });
		}
		assert!(bytes.is_empty());
		let nn = NN { layers };
		assert_eq!(hash_to_string(hash), nn.calc_hash_to_string(), "loaded and calculated hashes must match");
		nn
	}
}
impl SelectMove for NN {
	fn select_move(&self, board: &Board, _rng: &mut impl RngExt, _params: SelectMoveParams) -> (ChessMove, Option<Vec<(ChessMove, f)>>) {
		let moves_and_scores: Vec<(ChessMove, f)> = MoveGen::new_legal(board)
			.map(|move_| {
				let board_after_move = board.make_move_new(move_);
				let score = self.eval_board(&board_after_move);
				(move_, score)
			})
			.collect();
		let (best_move, _best_move_score) = match board.side_to_move() {
			Color::White => {
				moves_and_scores
					.iter()
					.max_by(|(_m1,s1), (_m2,s2)| s1.partial_cmp(s2).unwrap())
					.unwrap_or_else(|| {
						println!();
						dbg!(self.get_activation_fns());
						let moves_and_scores = MoveGen::new_legal(board)
							.map(|move_| {
								let board_after_move = board.make_move_new(move_);
								let score = self.eval_board(&board_after_move);
								(move_, score)
							});
						for (move_, score) in moves_and_scores {
							println!("{move_}: {score}");
						}
						panic!()
					})
			}
			Color::Black => {
				moves_and_scores
					.iter()
					.min_by(|(_m1,s1), (_m2,s2)| s1.partial_cmp(s2).unwrap())
					.unwrap_or_else(|| {
						println!();
						dbg!(self.get_activation_fns());
						let moves_and_scores = MoveGen::new_legal(board)
							.map(|move_| {
								let board_after_move = board.make_move_new(move_);
								let score = self.eval_board(&board_after_move);
								(move_, score)
							});
						for (move_, score) in moves_and_scores {
							println!("{move_}: {score}");
						}
						panic!()
					})
			}
		};
		(*best_move, Some(moves_and_scores))
	}
}

#[derive(Clone, PartialEq)]
struct NNLayer {
	weights: DMatrix<f>,
	biases: DVector<f>,
	activation_fn: ActivationFn,
}
impl NNLayer {
	pub fn new_random(size_in: u32, size_out: u32, rng: &mut impl RngExt) -> Self {
		// TODO(optim): dont use `random_range` repeatedly, instead create uniform distribution and multi sample it
		Self {
			weights: DMatrix::from_fn(
				size_out as usize,
				size_in as usize,
				|_i, _j| rng.random_range(nn_default::W_MIN .. nn_default::W_MAX)
			),
			biases: DVector::from_fn(
				size_out as usize,
				|_i, _| {
					let range = match_random_weighted! {rng,
						1. => size_in as f,
						1. => size_out as f,
						1. => ((size_in as f) + (size_out as f)) / 2.,
						1. => sqrt((size_in as f) * (size_out as f)),
					};
					rng.random_range(-range .. range)
				}
			),
			activation_fn: ActivationFn::new_random(rng),
		}
	}
	// pub fn new_random_from_spec(size_in: u32, size_out: u32, activation_fn: ActivationFn, rng: &mut impl RngExt) -> Self {
	// 	// TODO(optim): dont use `random_range` repeatedly, instead create uniform distribution and multi sample it
	// 	Self {
	// 		weights: DMatrix::from_fn(
	// 			size_out as usize,
	// 			size_in as usize,
	// 			|_i, _j| rng.random_range(nn_default::W_MIN .. nn_default::W_MAX)
	// 		),
	// 		biases: DVector::from_fn(
	// 			size_out as usize,
	// 			|_i, _| todo!()
	// 		),
	// 		activation_fn,
	// 	}
	// }
	fn calc_hash(&self) -> u64 {
		let mut hash = MyHash::from_seed(0x_c695d51f_e59c7bed);
		hash.hash(self.biases.as_slice());
		hash.hash(self.weights.as_slice());
		hash.hash(self.activation_fn.to_hash());
		hash.finish()
	}
	pub fn eval(&self, input: DVector<f>) -> DVector<f> {
		let sums = &self.weights * input + &self.biases;
		self.activation_fn.eval(sums)
	}
	pub fn evolve(&mut self, evolution_rate: f, rng: &mut impl RngExt) {
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
		if rng.random_bool((evolution_rate/3.) as f64) {
			self.activation_fn = ActivationFn::new_random(rng);
		}
	}
}

fn evolve_bias(bias: &mut f, rng: &mut impl RngExt) {
	evolve_value(bias, rng);
}

fn evolve_weight(weight: &mut f, rng: &mut impl RngExt) {
	evolve_value(weight, rng);
}

fn evolve_value(v: &mut f, rng: &mut impl RngExt) {
	match_random_weighted! {rng,
		// */
		0.001 => { *v *= -1.; },
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
		// 0.001 => { *v += sqrt(1. / abs(*v)); },
		// 0.001 => { *v -= sqrt(1. / abs(*v)); },
		// +- ln
		// 0.0001 => { *v += sqrt(ln(*v)); },
		// 0.0001 => { *v -= sqrt(ln(*v)); },
		// 0.0001 => { *v += sqrt(1. / ln(*v)); },
		// 0.0001 => { *v -= sqrt(1. / ln(*v)); },
	}
}





// TODO: test
fn board_to_vector_for_nn(board: &Board) -> Vec<f> {
	if nn_default::EXTRA_NOISE_INPUT { todo!() }
	let mut result: Vec<f> = vec![0.; nn_default::INPUT_SIZE as usize];
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

			match nn_default::NUMBER_OF_DEPTH_CHANNELS {
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
	// Centrist, // tries to keep pieces closer to center // TODO
	// Positionist, // tries to keep pieces at their "good" positions // TODO
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
	// BruteForceMinMax { depth: u8 },
	// AlphaBeta { depth?: u8 },
	Mix(AlgoPlayerMix),
	MixUnderSignedSqrt(AlgoPlayerMix),
}
impl AlgoPlayer {
	pub fn mix_new_random(rng: &mut impl RngExt) -> Self {
		Self::Mix(AlgoPlayerMix::new_random(rng))
	}
	pub fn mix_uss_new_random(rng: &mut impl RngExt) -> Self {
		Self::MixUnderSignedSqrt(AlgoPlayerMix::new_random(rng))
	}
	pub fn eval_board(self, board: &Board, rng: &mut impl RngExt) -> f {
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
			Mix(mix) => format!("Mix {} ({})", mix.calc_hash_to_string(), mix.to_string()),
			MixUnderSignedSqrt(mix) => format!("MixUSS {} ({})", mix.calc_hash_to_string(), mix.to_string()),
		}
	}
}
impl SelectMove for AlgoPlayer {
	fn select_move(&self, board: &Board, rng: &mut impl RngExt, _params: SelectMoveParams) -> (ChessMove, Option<Vec<(ChessMove, f)>>) {
		use AlgoPlayer::*;
		match self {
			RandomMover => {
				let moves = MoveGen::new_legal(board);
				let moves = moves.into_iter().collect::<Vec<_>>();
				let random_move_index = rng.random_range(0..moves.len());
				let random_move = moves[random_move_index];
				(random_move, None)
			}
			MiddleMover => {
				let moves = MoveGen::new_legal(board);
				let moves = moves.into_iter().collect::<Vec<_>>();
				let middle_move = moves[moves.len()/2];
				(middle_move, None)
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
				let moves_and_scores: Vec<(ChessMove, f)> = MoveGen::new_legal(board)
					.map(|move_| {
						let board_after_move = board.make_move_new(move_);
						let score = self.eval_board(&board_after_move, rng);
						(move_, score)
					})
					.collect();
				let (best_move, _best_move_score) = match board.side_to_move() {
					Color::White => {
						moves_and_scores
							.iter()
							.max_by(|(_m1,s1), (_m2,s2)| s1.partial_cmp(s2).unwrap())
							.unwrap_or_else(|| {
								println!();
								dbg!(self);
								let moves_and_scores = MoveGen::new_legal(board)
									.map(|move_| {
										let board_after_move = board.make_move_new(move_);
										let score = self.eval_board(&board_after_move, rng);
										(move_, score)
									});
								for (move_, score) in moves_and_scores {
									println!("{move_}: {score}");
								}
								panic!()
							})
					}
					Color::Black => {
						moves_and_scores
							.iter()
							.min_by(|(_m1,s1), (_m2,s2)| s1.partial_cmp(s2).unwrap())
							.unwrap_or_else(|| {
								println!();
								dbg!(self);
								let moves_and_scores = MoveGen::new_legal(board)
									.map(|move_| {
										let board_after_move = board.make_move_new(move_);
										let score = self.eval_board(&board_after_move, rng);
										(move_, score)
									});
								for (move_, score) in moves_and_scores {
									println!("{move_}: {score}");
								}
								panic!()
							})
					}
				};
				(*best_move, Some(moves_and_scores))
			}
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
	pub fn new_random(rng: &mut impl RngExt) -> Self {
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
	pub fn evolve(&mut self, evolution_rate: f, rng: &mut impl RngExt, algo_weights_clamp: f) {
		let mut ws = self.to_array();
		for w in ws.iter_mut() {
			if rng.random_bool(evolution_rate as f64) {
				// toggle zero / non-zero
				*w = if *w == 0. { rng.random_range(0. .. 1.) } else { 0. };
			}
			if rng.random_bool(evolution_rate as f64) {
				evolve_value(w, rng);
			}
			*w = w.clamp(0., algo_weights_clamp);
		}
		if ws.iter().all(|&w| w == 0.) {
			// let random_index = rng.random_range(0 .. ws.len());
			// ws[random_index] = 1.; // TODO?
			ws = Self::new_random(rng).to_array();
		} else {
			let ws_sum: f = ws.iter().sum();
			if ws_sum != 0. {
				ws = ws.map(|v| v / ws_sum);
			}
		}
		*self = Self::from_array(ws);
	}
	pub fn calc_hash_to_string(&self) -> String {
		let hash: u64 = self.calc_hash();
		hash_to_string(hash)
	}
	fn calc_hash(&self) -> u64 {
		let mut hash = MyHash::from_seed(0x_7dc29f45_3decba81);
		hash.hash(self.to_array().as_slice());
		hash.finish()
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





struct MyHash { value: u64 }
impl MyHash {
	// fn new() -> Self { ... }
	fn from_seed(seed: u64) -> Self { Self { value: seed } }
	fn finish(self) -> u64 { self.value }
}
pub trait MyHash_<T> {
	fn hash(&mut self, obj: T);
}
impl MyHash_<u64> for MyHash {
	fn hash(&mut self, obj: u64) {
		self.value ^= obj;
	}
}
impl MyHash_<&[f]> for MyHash {
	fn hash(&mut self, obj: &[f]) {
		// let mut hash: u64 = 0x_2e7ef108_6fce8375;
		let mut hash: u64 = self.value;
		hash ^= obj.len() as u64;
		for v in obj {
			let bits = v.to_bits() as u64;
			// hash-in value's bits:
			hash ^= if hash.count_ones() % 2 == 0 { bits } else { bits << 32 };
			// shuffle bits and bytes:
			let [b0,b1,b2,b3, b4,b5,b6,b7] = hash.to_le_bytes();
			const N: u64 = 11;
			hash = match hash % N {
				0 => !hash,
				1 => hash.reverse_bits(),
				2 => hash.rotate_left(1), // rotate bits
				3 => u64::from_le_bytes([b7,b6,b5,b4, b3,b2,b1,b0]), // reverse bytes
				4 => u64::from_le_bytes([b1,b2,b3,b4, b5,b6,b7,b0]), // rotate bytes
				5 => u64::from_le_bytes([b0,b2,b4,b6, b1,b3,b5,b7]), // 2-braid
				6 => u64::from_le_bytes([b0,b3,b6, b1,b4,b7, b2,b5]), // 3-braid
				7 => u64::from_le_bytes([b0,b4, b1,b5, b2,b6, b3,b7]), // 4-braid
				8 => u64::from_le_bytes([!b0,b1,!b2,b3, !b4,b5,!b6,b7]), // inverse bytes % 2
				9 => u64::from_le_bytes([!b0,b1,b2,!b3, b4,b5,!b6,b7]), // inverse bytes % 3
				10 => u64::from_le_bytes([!b0,b1,b2,b3, !b4,b5,b6,b7]), // inverse bytes % 4
				N.. => unreachable!()
			}
		}
		self.value = hash;
	}
}

pub fn hash_to_string(hash: u64) -> String {
	// hash_to_string_hex(hash)
	hash_to_string_base32(hash)
}

fn hash_to_string_hex(hash: u64) -> String {
	format!("{:016x}", hash)
}

fn hash_to_string_base32(mut hash: u64) -> String {
	const ALPHABET: [char; 32] = ['0','1','2','3','4','5','6','7','8','9','a','b','c','d','e','f','g','h','i','j','k',/*'l',*/'m','n','o','p',/*'q',*/'r',/*'s',*/'t','u','v','w','x','y',/*'z',*/]; // removed 4 most confusable letters
	const N: usize = 13; // len = ceil(64/5)
	let mut chars = Vec::with_capacity(N);
	for _ in 0..13 {
		// TODO?: if hash < 32 => ?
		chars.push(ALPHABET[(hash % 32) as usize]);
		hash /= 32;
	}
	assert_eq!(13, chars.len());
	assert_eq!(0, hash);
	// chars.reverse(); // TODO?
	chars.into_iter().join("")
}





struct BoardToHumanViewableConfig { beautiful_output: bool, show_files_ranks: bool, show_pieces_diff: bool }
impl BoardToHumanViewableConfig {
	fn all() -> Self {
		Self { beautiful_output: true, show_files_ranks: true, show_pieces_diff: true }
	}
}
fn board_to_human_viewable(board: &Board, config: BoardToHumanViewableConfig) -> String {
	const FILES: [&str; 8] = ["a", "b", "c", "d", "e", "f", "g", "h"];
	const RANKS: [&str; 8] = ["1", "2", "3", "4", "5", "6", "7", "8"];
	let x_line: String = format!("  {}", FILES.join(" "));
	let options_pieces_diff_lost_str = if config.show_pieces_diff {
		Some(get_pieces_diff_lost_str(board, VecPieceToStringConfig { separator: None, is_beautiful: true }))
	} else {
		None
	};
	let approx_capacity: usize = if config.show_files_ranks { 250 } else { 200 }; // 64*2 +? 16*4
	let mut res: String = String::with_capacity(approx_capacity);
	if config.show_files_ranks {
		res += &x_line;
		res += "\n";
	}
	let board_builder: BoardBuilder = board.into();
	for y in (0..8).rev() {
		if y != 7 {
			res += "\n";
		}
		for x in 0..8 {
			let index = y*8 + x;
			let square = ALL_SQUARES[index];
			let option_piece_and_color = board_builder[square];
			if x == 0 {
				res += RANKS[y];
				res += " ";
			}
			res += &if config.beautiful_output {
				chess_pieces_unicode::get(option_piece_and_color)
			} else {
				chess_pieces_ascii::get(option_piece_and_color)
			}.to_string();
			res += " ";
			if x == 7 {
				res += RANKS[y];
				if config.show_pieces_diff {
					match y {
						0 => { res += " "; res += &options_pieces_diff_lost_str.as_ref().unwrap().white_pieces }
						7 => { res += " "; res += &options_pieces_diff_lost_str.as_ref().unwrap().black_pieces }
						_ => {}
					}
				}
			}
		}
	}
	if config.show_files_ranks {
		res += "\n";
		res += &x_line;
	}
	res.shrink_to_fit();
	res
}

#[derive(Clone, Copy)]
struct VecPieceToStringConfig { separator: Option<&'static str>, is_beautiful: bool }
fn vec_piece_to_string(pieces: Vec<Piece>, color: Color, config: VecPieceToStringConfig) -> String {
	pieces
		.into_iter()
		.map(|piece| {
			let option_piece_and_color = Some((piece, color));
			let get_chess_piece = if config.is_beautiful { chess_pieces_unicode::get } else { chess_pieces_ascii::get };
			get_chess_piece(option_piece_and_color).to_string()
		})
		.reduce(|acc, el| acc + config.separator.unwrap_or_default() + &el)
		.unwrap_or_default()
}

fn get_pieces_diff_lost_str(board: &Board, vec_piece_to_string_config: VecPieceToStringConfig) -> PiecesByColor<String> {
	let PiecesByColor { white_pieces, black_pieces } = get_pieces_diff_lost(board);
	PiecesByColor {
		white_pieces: vec_piece_to_string(white_pieces, Color::White, vec_piece_to_string_config),
		black_pieces: vec_piece_to_string(black_pieces, Color::Black, vec_piece_to_string_config),
	}
}
#[test]
fn get_pieces_diff_lost_str_() {
	/* ♚♛♜♝♞♟ ♔♕♖♗♘♙ */
	let expected = PiecesByColor {
		white_pieces: "♟ ♟ ♟ ♟".to_string(),
		black_pieces: "♕ ♖ ♗ ♘".to_string()
	};
	let actual = get_pieces_diff_lost_str(
		&Board::from_fen("r1b1k1n1/pppppppp/8/8/8/8/P1P1P1P1/RNBQKBNR w KQq - 0 1".to_string()).unwrap(),
		VecPieceToStringConfig { separator: Some(" "), is_beautiful: true },
	);
	// let expected_chars: Vec<char> = expected.chars().collect();
	// let actual_chars: Vec<char> = actual.chars().collect();
	assert_eq!(expected, actual);
}

/// Returns `{white_pieces: Vec<Piece>, black_pieces: Vec<Piece>}`,
/// where `white_pieces` - pieces that white have,
/// and black dont, and `black_pieces` - vice versa.
fn get_pieces_diff_have(board: &Board) -> PiecesByColor<Vec<Piece>> {
	let PiecesByColor { mut white_pieces, mut black_pieces } = get_pieces_by_color(board);
	white_pieces.sort();
	black_pieces.sort();
	fn pieces_to_some_pieces(pieces: Vec<Piece>) -> Vec<Option<Piece>> {
		pieces
			.into_iter()
			.map(Some)
			.collect()
	}
	let mut white_pieces: Vec<Option<Piece>> = pieces_to_some_pieces(white_pieces);
	let mut black_pieces: Vec<Option<Piece>> = pieces_to_some_pieces(black_pieces);
	let mut i = 0;
	let mut j = 0;
	while i < white_pieces.len() || j < black_pieces.len() {
		let white_piece = &mut white_pieces[i];
		let black_piece = &mut black_pieces[j];
		match white_piece.cmp(&black_piece) {
			Ordering::Equal => {
				*white_piece = None;
				*black_piece = None;
				i += 1;
				j += 1;
			}
			Ordering::Greater => { j += 1 }
			Ordering::Less => { i += 1 }
		}
	}
	let mut white_pieces: Vec<Piece> = white_pieces.into_iter().flatten().collect();
	let mut black_pieces: Vec<Piece> = black_pieces.into_iter().flatten().collect();
	white_pieces.reverse();
	black_pieces.reverse();
	white_pieces.shrink_to_fit();
	black_pieces.shrink_to_fit();
	PiecesByColor { white_pieces, black_pieces }
}

/// Returns `{white_pieces: Vec<Piece>, black_pieces: Vec<Piece>}`,
/// inversed to [`get_pieces_diff_have`].
fn get_pieces_diff_lost(board: &Board) -> PiecesByColor<Vec<Piece>> {
	let PiecesByColor { white_pieces, black_pieces } = get_pieces_diff_have(board);
	PiecesByColor { white_pieces: black_pieces, black_pieces: white_pieces }
}

#[derive(Debug, PartialEq)]
struct PiecesByColor<T> { white_pieces: T, black_pieces: T }
/// Returns unsorted `(white_pieces, black_pieces)`
fn get_pieces_by_color(board: &Board) -> PiecesByColor<Vec<Piece>> {
	let board_builder: BoardBuilder = board.into();
	let mut white_pieces = Vec::new();
	let mut black_pieces = Vec::new();
	for square in ALL_SQUARES {
		let option_piece_and_color = board_builder[square];
		match option_piece_and_color {
			Some((piece, Color::White)) => { white_pieces.push(piece) }
			Some((piece, Color::Black)) => { black_pieces.push(piece) }
			_ => {}
		}
	}
	PiecesByColor { white_pieces, black_pieces }
}

mod chess_pieces_ascii {
	use chess::{Color, Piece};

	pub const NONE: char = '.';

	pub const PAWN_WHITE  : char = 'P';
	pub const KNIGHT_WHITE: char = 'N';
	pub const BISHOP_WHITE: char = 'B';
	pub const ROOK_WHITE  : char = 'R';
	pub const QUEEN_WHITE : char = 'Q';
	pub const KING_WHITE  : char = 'K';

	pub const PAWN_BLACK  : char = 'p';
	pub const KNIGHT_BLACK: char = 'n';
	pub const BISHOP_BLACK: char = 'b';
	pub const ROOK_BLACK  : char = 'r';
	pub const QUEEN_BLACK : char = 'q';
	pub const KING_BLACK  : char = 'k';

	pub fn get(option_piece_and_color: Option<(Piece, Color)>) -> char {
		let Some(piece_and_color) = option_piece_and_color else { return NONE };
		match piece_and_color {
			(Piece::Pawn  , Color::White) => PAWN_WHITE,
			(Piece::Knight, Color::White) => KNIGHT_WHITE,
			(Piece::Bishop, Color::White) => BISHOP_WHITE,
			(Piece::Rook  , Color::White) => ROOK_WHITE,
			(Piece::Queen , Color::White) => QUEEN_WHITE,
			(Piece::King  , Color::White) => KING_WHITE,

			(Piece::Pawn  , Color::Black) => PAWN_BLACK,
			(Piece::Knight, Color::Black) => KNIGHT_BLACK,
			(Piece::Bishop, Color::Black) => BISHOP_BLACK,
			(Piece::Rook  , Color::Black) => ROOK_BLACK,
			(Piece::Queen , Color::Black) => QUEEN_BLACK,
			(Piece::King  , Color::Black) => KING_BLACK,
		}
	}
}

mod chess_pieces_unicode {
	use chess::{Color, Piece};

	pub const NONE: char = '.';

	/* ♚♛♜♝♞♟ ♔♕♖♗♘♙ */

	pub const PAWN_WHITE  : char = '♟';
	pub const KNIGHT_WHITE: char = '♞';
	pub const BISHOP_WHITE: char = '♝';
	pub const ROOK_WHITE  : char = '♜';
	pub const QUEEN_WHITE : char = '♛';
	pub const KING_WHITE  : char = '♚';

	pub const PAWN_BLACK  : char = '♙';
	pub const KNIGHT_BLACK: char = '♘';
	pub const BISHOP_BLACK: char = '♗';
	pub const ROOK_BLACK  : char = '♖';
	pub const QUEEN_BLACK : char = '♕';
	pub const KING_BLACK  : char = '♔';

	pub fn get(option_piece_and_color: Option<(Piece, Color)>) -> char {
		let Some(piece_and_color) = option_piece_and_color else { return NONE };
		match piece_and_color {
			(Piece::Pawn  , Color::White) => PAWN_WHITE,
			(Piece::Knight, Color::White) => KNIGHT_WHITE,
			(Piece::Bishop, Color::White) => BISHOP_WHITE,
			(Piece::Rook  , Color::White) => ROOK_WHITE,
			(Piece::Queen , Color::White) => QUEEN_WHITE,
			(Piece::King  , Color::White) => KING_WHITE,

			(Piece::Pawn  , Color::Black) => PAWN_BLACK,
			(Piece::Knight, Color::Black) => KNIGHT_BLACK,
			(Piece::Bishop, Color::Black) => BISHOP_BLACK,
			(Piece::Rook  , Color::Black) => ROOK_BLACK,
			(Piece::Queen , Color::Black) => QUEEN_BLACK,
			(Piece::King  , Color::Black) => KING_BLACK,
		}
	}
}





#[derive(Debug, Clone, Copy)]
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
		let moves_strs: Vec<String> = self.actions().iter().flat_map(|action| {
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

	#[allow(non_snake_case)]
	mod calc_elo_rating_delta {
		use super::*;
		#[test] fn strong_wins() { assert_eq!(9.090909, calc_elo_rating_delta(1200., 800.)) }
		#[test] fn weak_wins  () { assert_eq!(90.90909, calc_elo_rating_delta(800., 1200.)) }
		#[test] fn white_1200__black_800__white_wins() { assert_eq!(9.090909, calc_elo_rating_delta(1200., 800.)) }
		#[test] fn white_1200__black_800__black_wins() { assert_eq!(90.90909, calc_elo_rating_delta(800., 1200.)) }
		#[test] fn white_800__black_1200__white_wins() { assert_eq!(90.90909, calc_elo_rating_delta(800., 1200.)) }
		#[test] fn white_800__black_1200__black_wins() { assert_eq!(9.090909, calc_elo_rating_delta(1200., 800.)) }
	}

	mod activation_fns {
		use super::*;
		#[test]
		fn to_from_hash_identity() {
			for af in ActivationFn::get_all_variants() {
				assert_eq!(af, ActivationFn::from_hash(af.to_hash()));
			}
		}
	}

	mod nn {
		use super::*;
		#[test]
		fn to_from_bytes() {
			let mut rng = rng();
			for _ in 0..10 {
				let inner_layers_sizes = Vec::from_fn(rng.random_range(1..10), |_i| rng.random_range(1. .. 10_f32).powi(2).round() as u32);
				let nn = NN::new_random(&inner_layers_sizes, &mut rng);
				// assert_eq!(nn, NN::from_bytes(&nn.to_bytes()));
				if nn != NN::from_bytes(&nn.to_bytes()) { panic!() }
			}
		}
	}

	mod hash_to_string_base32 {
		use super::*;
		#[test] fn _0x_0000_0000_0000_0000() { assert_eq!("0000000000000", hash_to_string_base32(0x_0000_0000_0000_0000)); }
		#[test] fn _0x_0000_0000_0000_0145() { assert_eq!("5a00000000000", hash_to_string_base32(0x_0000_0000_0000_0145)); }
		#[test] fn _0x_1370_0000_0000_0000() { assert_eq!("0000000000v61", hash_to_string_base32(0x_1370_0000_0000_0000)); }
	}
}

