//! Input/Output utils.

#![allow(dead_code)]

use std::{fmt::{Debug, Display}, str::FromStr};

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

pub fn prompt_str(text: &str) -> String {
	use std::io::{BufRead, stdin};
	print(text);
	let mut line = String::new();
	let _ = stdin().lock().read_line(&mut line).expect("Could not read line");
	line.trim().to_string()
}

/// user better dont make mistakes or it will crash
pub fn prompt_once_unwrap<T: FromStr>(text: &str) -> T where <T as FromStr>::Err: Debug {
	let input = prompt_str(text);
	input.parse().unwrap()
}

pub fn prompt_once<T: FromStr>(text: &str) -> Result<T, <T as FromStr>::Err> {
	let input = prompt_str(text);
	input.parse()
}

pub fn prompt<T: FromStr>(text: &str) -> T where <T as FromStr>::Err: Debug {
	loop {
		match prompt_once(text) {
			Ok(input) => { return input }
			Err(err) => { println!("Error: {err:?}"); }
		}
	}
}

pub fn prompt_with_default<T: FromStr>(text: &str, default: T) -> T where <T as FromStr>::Err: Debug {
	loop {
		let input = prompt_str(text);
		if input.is_empty() { return default }
		match input.parse() {
			Ok(input) => { return input }
			Err(err) => { println!("Error: {err:?}"); }
		}
	}
}

pub fn prompt_with_name_and_default<T: FromStr + Display>(name: &str, default: T) -> T where <T as FromStr>::Err: Debug {
	prompt_with_default(&format!("{name} (default: {default}): "), default)
}

pub fn prompt_bool_with_name_and_default(name: &str, default: bool) -> bool {
	let default_str = if default { "yes" } else { "no" };
	loop {
		break match prompt_with_name_and_default(&format!("{name} (Yes/No)"), default_str.to_string()).as_str() {
			"y" | "yes" => true,
			"n" | "no" => false,
			_ => continue
		}
	}
}

