//! math functions

use crate::{f, math_aliases::{abs, sqrt}};


pub fn signed_sqrt(x: f) -> f {
	x.signum() * sqrt(abs(x))
}

pub fn signed_square(x: f) -> f {
	x.signum() * x.powi(2)
}

