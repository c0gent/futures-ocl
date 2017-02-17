extern crate ocl;

use std::mem;
use ocl::Event;
use ocl::core::{Event as EventCore, UserEvent as UserEventCore};

pub fn main() {
    println!("Event: {}", mem::size_of::<[Event; 10]>());
    println!("EventCore: {}", mem::size_of::<[EventCore; 10]>());
    println!("UserEventCore: {}", mem::size_of::<[UserEventCore; 10]>());
    return;

}