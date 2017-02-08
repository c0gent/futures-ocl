extern crate futures;

use std::io;
use std::time::Duration;
use futures::future::{Future, Map};


// A future is actually a trait implementation, so we can generically take a
// future of any integer and return back a future that will resolve to that
// value plus 10 more.
//
// Note here that like iterators, we're returning the `Map` combinator in
// the futures crate, not a boxed abstraction. This is a zero-cost
// construction of a future.
fn add_ten<F>(future: F) -> Map<F, fn(i32) -> i32>
    where F: Future<Item=i32>,
{
    fn add(a: i32) -> i32 { a + 10 }
    future.map(add)
}


fn main() {
    add_ten(5);
}