//! Input/Output utils.

#![allow(dead_code)]

pub fn press_enter_to_continue() {
	print("PRESS ENTER TO CONTINUE");
	wait_for_enter();
}

pub fn wait_for_enter() {
	use std::io::stdin;
	let mut line: String = String::new();
	let _ = stdin().read_line(&mut line).unwrap();
}

pub fn flush() {
	use std::io::{Write, stdout};
	stdout().flush().unwrap();
}

/// print and flush
pub fn print(msg: impl ToString) {
	print!("{}", msg.to_string());
	flush();
}

pub fn prompt(text: &str) -> String {
	use std::io::{BufRead, stdin};
	print(text);
	let mut line = String::new();
	let _ = stdin().lock().read_line(&mut line).expect("Could not read line");
	line.trim().to_string()
}

