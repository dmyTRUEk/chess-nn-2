//! extensions

pub trait IndexOfMaxMin<T> {
	fn index_of_max(&self) -> Option<usize>;
	fn index_of_min(&self) -> Option<usize>;
}
impl<T: PartialOrd> IndexOfMaxMin<T> for &[T] {
	fn index_of_max(&self) -> Option<usize> {
		let mut option_index_of_max = None;
		for i in 0..self.len() {
			match option_index_of_max {
				None => {
					option_index_of_max = Some(i);
				}
				Some(index_of_max) if self[i] > self[index_of_max] => {
					option_index_of_max = Some(i);
				}
				_ => {}
			}
		}
		option_index_of_max
	}
	fn index_of_min(&self) -> Option<usize> {
		let mut option_index_of_min = None;
		for i in 0..self.len() {
			match option_index_of_min {
				None => {
					option_index_of_min = Some(i);
				}
				Some(index_of_min) if self[i] < self[index_of_min] => {
					option_index_of_min = Some(i);
				}
				_ => {}
			}
		}
		option_index_of_min
	}
}



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

