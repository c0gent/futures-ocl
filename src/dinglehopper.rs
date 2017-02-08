use futures::{Future, Poll, Async};

#[derive(Debug)]
pub struct Dinglehopper {
    val: i32,
    complete: bool,
    terribad: bool,
}

impl Dinglehopper {
    pub fn new(val: i32) -> Dinglehopper {
        Dinglehopper {
            val: val,
            complete: false,
            terribad: false,
        }
    }
}

impl Future for Dinglehopper {
    type Item = i32;
    type Error = ();

    fn poll(&mut self) -> Poll<Self::Item, Self::Error> {

        if self.terribad {
            Err(())
        } else {
            if self.complete{
                Ok(Async::Ready(self.val))
            } else {
                Ok(Async::NotReady)
            }
        }
    }
}

