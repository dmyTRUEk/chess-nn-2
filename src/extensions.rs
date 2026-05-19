//! extensions

pub trait Pushed<T> {
	fn pushed(self, other: T) -> Self;
}
impl<T> Pushed<T> for Vec<T> {
	fn pushed(mut self, other: T) -> Self {
		self.push(other);
		self
	}
}

pub trait Extended<T> {
	fn extended(self, other: impl IntoIterator<Item=T>) -> Self;
}
impl<T> Extended<T> for Vec<T> {
	fn extended(mut self, other: impl IntoIterator<Item=T>) -> Self {
		self.extend(other);
		self
	}
}

