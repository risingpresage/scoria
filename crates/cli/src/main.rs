mod gate;
mod uring;

use std::env;
use std::process;

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    match args.first().map(|s| s.as_str()) {
        Some("gate") => {
            let quick = args.iter().any(|a| a == "--quick");
            if let Err(e) = gate::run(quick) {
                eprintln!("scoria gate failed: {e}");
                process::exit(1);
            }
        }
        _ => {
            eprintln!("Usage: scoria gate [--quick]");
            process::exit(2);
        }
    }
}
