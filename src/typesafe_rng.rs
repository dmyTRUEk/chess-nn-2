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
		impl $name {
			pub const NUMBER_OF_VARIANTS: u8 = $num;
		}
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
impl_gen_with_weights!(13, V13, { _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13 });
impl_gen_with_weights!(14, V14, { _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14 });
impl_gen_with_weights!(15, V15, { _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15 });
impl_gen_with_weights!(16, V16, { _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16 });
impl_gen_with_weights!(17, V17, { _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17 });
impl_gen_with_weights!(18, V18, { _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18 });
impl_gen_with_weights!(19, V19, { _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19 });
impl_gen_with_weights!(20, V20, { _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20 });
impl_gen_with_weights!(21, V21, { _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21 });
impl_gen_with_weights!(22, V22, { _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22 });
impl_gen_with_weights!(23, V23, { _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23 });
impl_gen_with_weights!(24, V24, { _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24 });
impl_gen_with_weights!(25, V25, { _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25 });
impl_gen_with_weights!(26, V26, { _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26 });
impl_gen_with_weights!(27, V27, { _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27 });
impl_gen_with_weights!(28, V28, { _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28 });
impl_gen_with_weights!(29, V29, { _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29 });
impl_gen_with_weights!(30, V30, { _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30 });
impl_gen_with_weights!(31, V31, { _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31 });
impl_gen_with_weights!(32, V32, { _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32 });
impl_gen_with_weights!(33, V33, { _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, _33 });
impl_gen_with_weights!(34, V34, { _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, _33, _34 });
impl_gen_with_weights!(35, V35, { _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, _33, _34, _35 });
impl_gen_with_weights!(36, V36, { _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, _33, _34, _35, _36 });
impl_gen_with_weights!(37, V37, { _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, _33, _34, _35, _36, _37 });
impl_gen_with_weights!(38, V38, { _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, _33, _34, _35, _36, _37, _38 });
impl_gen_with_weights!(39, V39, { _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, _33, _34, _35, _36, _37, _38, _39 });
impl_gen_with_weights!(40, V40, { _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, _33, _34, _35, _36, _37, _38, _39, _40 });
impl_gen_with_weights!(41, V41, { _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, _33, _34, _35, _36, _37, _38, _39, _40, _41 });
impl_gen_with_weights!(42, V42, { _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, _33, _34, _35, _36, _37, _38, _39, _40, _41, _42 });
// NOTE: 255 is max bc of `#[repr(u8)]`





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

