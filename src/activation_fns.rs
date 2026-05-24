//! activation fns

use crate::{f, math::*, math_aliases::*};


pub fn relu(x: f) -> f {
	x.max(0.)
}

pub fn leaky_relu_01(x: f) -> f {
	if x.is_sign_positive() { x } else { 0.1 * x }
}
pub fn leaky_relu_001(x: f) -> f {
	if x.is_sign_positive() { x } else { 0.01 * x }
}

pub fn sign(x: f) -> f {
	x.signum()
}

pub fn step(x: f) -> f {
	if x.is_sign_positive() { 1. } else { 0. }
}

pub fn sigmoid(x: f) -> f {
	1. / (1. + exp(-x))
}

// in `math_aliases.rs`
// pub fn tanh(x: f) -> f {
// 	x.tanh()
// }

pub fn soft_sign(x: f) -> f {
	x / (1. + abs(x))
}

pub fn soft_plus(x: f) -> f {
	ln(1. + exp(x))
}

pub fn explu(x: f) -> f {
	if x.is_sign_positive() { x } else { 0.1 * (exp(x) - 1.) }
}

pub fn silu(x: f) -> f {
	x / (1. + exp(-x))
}

pub fn elish(x: f) -> f {
	(if x.is_sign_positive() { x } else { exp(x) - 1. }) / (1. + exp(-x))
}

pub fn gaussian(x: f) -> f {
	exp(-x.powi(2))
}

// in `math_aliases.rs`
// pub fn sin(x: f) -> f {
// 	x.sin()
// }

pub fn clamp01(x: f) -> f {
	x.clamp(0., 1.)
}

pub fn resqrt(x: f) -> f {
	if x.is_sign_positive() { sqrt(x) } else { 0. }
}

// in `math.rs`
// pub fn signed_sqrt(x: f) -> f {
// 	x.signum() * sqrt(abs(x))
// }

pub fn leaky_resqrt_01(x: f) -> f {
	signed_sqrt(x) * if x.is_sign_positive() { 1. } else { 0.1 }
}
pub fn leaky_resqrt_001(x: f) -> f {
	signed_sqrt(x) * if x.is_sign_positive() { 1. } else { 0.01 }
}

pub fn signed_sqrt_p1(x: f) -> f {
	signed_sqrt(x) + 1.
}

pub fn resquare(x: f) -> f {
	if x.is_sign_positive() { x.powi(2) } else { 0. }
}

// in `math.rs`
// pub fn signed_square(x: f) -> f {
// 	x.signum() * x.powi(2)
// }

pub fn leaky_resquare_01(x: f) -> f {
	signed_square(x) * if x.is_sign_positive() { 1. } else { 0.1 }
}
pub fn leaky_resquare_001(x: f) -> f {
	signed_square(x) * if x.is_sign_positive() { 1. } else { 0.01 }
}

pub fn sinc(x: f) -> f {
	// credit: chatgpt
	if x.abs() < 1e-4 {
		// Taylor expansion around 0:
		// sin(x)/x ≈ 1 - x²/6 + x⁴/120
		let x2 = x * x;
		1.0 - x2 * (1.0 / 6.0) + x2 * x2 * (1.0 / 120.0)
	} else {
		sin(x) / x
	}
}

pub fn resinc(x: f) -> f {
	if x.is_sign_positive() { sinc(x) } else { 0. }
}

