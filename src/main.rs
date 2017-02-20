#![allow(dead_code, unused_variables, unused_imports, unused_mut, unreachable_code)]

extern crate libc;
extern crate futures;
extern crate futures_cpupool;
extern crate tokio_core;
extern crate tokio_timer;
extern crate rand;
extern crate chrono;
extern crate ocl;
#[macro_use] extern crate lazy_static;
#[macro_use] extern crate colorify;

mod extras_proto;
mod async_process_proto;
mod main_orig_proto;
mod switches;

pub fn main() {
    main_orig_proto::main();
    // async_process_proto::main();

}