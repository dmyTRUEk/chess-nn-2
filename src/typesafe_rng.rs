//! Typesafe RNG

use num_enum::{IntoPrimitive, TryFromPrimitive};
use rand::{Rng, RngExt, distr::weighted::WeightedIndex, prelude::Distribution};



pub trait TypesafeRNG<const N: usize, T> {
	fn random_variant(&mut self) -> T;
	fn random_variant_weighted(&mut self, weights: [f32; N]) -> T;
}

macro_rules! impl_gen_with_weights {
	($num:literal, $name:ident, $elems:tt) => {
		#[derive(Debug, Clone, Copy, IntoPrimitive, TryFromPrimitive)]
		#[repr(u8)]
		pub enum $name $elems
		impl<R: Rng> TypesafeRNG<$num, $name> for R {
			fn random_variant(&mut self) -> $name {
				let n: u8 = self.random_range(0..$num);
				$name::try_from(n).unwrap()
			}
			fn random_variant_weighted(&mut self, weights: [f32; $num]) -> $name {
				let n = WeightedIndex::new(weights).unwrap().sample(self);
				// `u8` because #[repr(u8)]
				let n: u8 = n.try_into().unwrap();
				$name::try_from(n).unwrap()
			}
		}
	}
}



/// match probability => outcome
///
/// Example:
/// ```
/// let x: char = match_random_weighted! { &mut rng,
///     1. => { 'a' },
///     2. => { 'b' },
///     4. => { 'c' },
/// }
/// ```
#[macro_export]
macro_rules! match_random_weighted {
	(
		$rng:expr,
		$( $weight:expr => $body:expr ),+ $(,)?
	) => {{
		use rand::{distr::weighted::WeightedIndex, prelude::Distribution};
		let weights = [$( $weight ),+];
		let i = WeightedIndex::new(weights).unwrap().sample($rng);
		match_random_weighted!(@arms i, 0; $( $body ),+)
	}};

	// recursive case (at least 2 items)
	(@arms $i:ident, $idx:expr; $body:expr, $( $rest:expr ),+ ) => {
		if $i == $idx {
			$body
		} else {
			match_random_weighted!(@arms $i, $idx + 1; $( $rest ),+)
		}
	};

	// base case (last item)
	(@arms $i:ident, $idx:expr; $body:expr ) => {
		if $i == $idx {
			$body
		} else {
			unreachable!()
		}
	};
}



// TODO: somehow use `cargo expand` to see the output of only this file/macro?
impl_gen_with_weights!(1, V1, { _1 });
impl_gen_with_weights!(2, V2, { _1, _2 });
impl_gen_with_weights!(3, V3, { _1, _2, _3 });
impl_gen_with_weights!(4, V4, { _1, _2, _3, _4 });
impl_gen_with_weights!(5, V5, { _1, _2, _3, _4, _5 });
impl_gen_with_weights!(6, V6, { _1, _2, _3, _4, _5, _6 });
impl_gen_with_weights!(7, V7, { _1, _2, _3, _4, _5, _6, _7 });
impl_gen_with_weights!(8, V8, { _1, _2, _3, _4, _5, _6, _7, _8 });
impl_gen_with_weights!(9, V9, { _1, _2, _3, _4, _5, _6, _7, _8, _9 });
impl_gen_with_weights!(10, V10, { _1, _2, _3, _4, _5, _6, _7, _8, _9, _10 });
impl_gen_with_weights!(11, V11, { _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11 });
impl_gen_with_weights!(12, V12, { _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12 });





#[cfg(test)]
mod tests {
	// use super::*;

	mod match_random_weighted {
		// use super::*;
		use rand::rng;
		mod with_braces {
			#![allow(unused_braces)]
			use super::*;
			#[test]
			fn exec_block() {
				let mut rng = rng();
				for _ in 0..100 {
					match_random_weighted! { &mut rng,
						1. => { println!("1.") },
						2. => { println!("2.") },
						4. => { println!("4.") },
					}
				}
				// panic!()
			}
			#[test]
			fn return_value() {
				let mut rng = rng();
				for _ in 0..100 {
					let x: i32 = match_random_weighted! { &mut rng,
						1. => { 1 },
						2. => { 2 },
						4. => { 4 },
					};
					println!("{x}");
				}
				// panic!()
			}
		}
		mod without_braces {
			use super::*;
			#[test]
			fn exec_block() {
				let mut rng = rng();
				for _ in 0..100 {
					match_random_weighted! { &mut rng,
						1. => println!("1."),
						2. => println!("2."),
						4. => println!("4."),
					}
				}
				// panic!()
			}
			#[test]
			fn return_value() {
				let mut rng = rng();
				for _ in 0..100 {
					let x: i32 = match_random_weighted! { &mut rng,
						1. => 1,
						2. => 2,
						4. => 4,
					};
					println!("{x}");
				}
				// panic!()
			}
		}
	}

}

