#![allow(dead_code, unused_variables, unused_imports, unused_mut, unreachable_code)]

extern crate libc;
extern crate crossbeam;
extern crate futures;
extern crate futures_cpupool;
extern crate tokio_core;
extern crate tokio_timer;
extern crate rand;
extern crate chrono;
extern crate ocl;
#[macro_use] extern crate lazy_static;
#[macro_use] extern crate colorify;

mod extras;
mod switches;

use std::io::Write;
// use std::thread;
use std::rc::Rc;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use libc::c_void;
use crossbeam::sync::MsQueue;
use futures::{future, stream, Async, Sink, Stream};
use futures::future::*;
use futures::sync::mpsc::{self, Receiver, Sender, UnboundedSender};
use futures_cpupool::{CpuPool, CpuFuture};
use tokio_core::reactor::{Core, Handle};
use tokio_timer::{Timer, Sleep, TimerError};

use rand::{Rng, XorShiftRng};
use rand::distributions::{IndependentSample, Range as RandRange};
use std::collections::{LinkedList, HashMap, BTreeSet};
use ocl::{core, async, Platform, Device, Context, Queue, Program, Buffer, Kernel, SubBuffer, OclPrm,
    Event, EventList, FutureMemMap, MemMap, Error as OclError};
use ocl::flags::{MemFlags, MapFlags, CommandQueueProperties};
use ocl::aliases::ClFloat4;
use ocl::async::{FutureResult as FutureAsyncResult, Error as AsyncError};

use extras::{BufferPool, CommandGraph, Command, CommandDetails, KernelArgBuffer, RwCmdIdxs};
use switches::{Switches, SWITCHES};

const INITIAL_BUFFER_LEN: u32 = 2 << 23; // 256MiB of ClFloat4
const SUB_BUF_MIN_LEN: u32 = 2 << 11; // 64KiB of ClFloat4
const SUB_BUF_MAX_LEN: u32 = 2 << 15; // 1MiB of ClFloat4


static KERN_SRC: &'static str = r#"
    __kernel void add(
        __global float4* in,
        float4 values,
        __global float4* out)
    {
        uint idx = get_global_id(0);
        out[idx] = in[idx] + values;
    }
"#;


fn fmt_duration(duration: chrono::Duration) -> String {
    let el_sec = duration.num_seconds();
    let el_ms = duration.num_milliseconds() - (el_sec * 1000);
    format!("{}.{} seconds", el_sec, el_ms)
}



pub fn main() {
    use std::mem;
    use ocl::core::{Event as EventCore};


    let buffer_size_range = RandRange::new(SUB_BUF_MIN_LEN, SUB_BUF_MAX_LEN);
    let mut rng = rand::weak_rng();

    let platform = Platform::default();
    println!("Platform: {}", platform.name());

    let device_idx = 1;

    let device = Device::specifier()
        .wrapping_indices(vec![device_idx])
        .to_device_list(Some(&platform)).unwrap()[0];

    println!("Device: {} {}", device.vendor(), device.name());

    let context = Context::builder()
        .platform(platform)
        .devices(device)
        .build().unwrap();

    let queue_flags = Some(SWITCHES.queue_flags);
    let io_queue = Queue::new(&context, device, queue_flags).unwrap();
    let kern_queue = Queue::new(&context, device, queue_flags).unwrap();

    // let mut buf_pool: BufferPool<ClFloat4> = BufferPool::new(INITIAL_BUFFER_LEN, io_queue.clone());
    // let mut pool_full = false;

    let start_time = chrono::Local::now();
    println!("Creating tasks...");


    let thread_pool = CpuPool::new_num_cpus();
    let mut core = Core::new().unwrap();
    let handle = core.handle();
    // let remote  = core.remote();

    let mut correct_val_count = 0usize;
    let (tx, mut rx) = mpsc::channel(1);
    // let (tx, mut rx) = mpsc::unbounded();

    let mut offloads = MsQueue::new();


    println!("Enqueuing tasks...");

    for _ in 0..4 {
        let work_size = 2 << 15;

        let write_buf_flags = Some(MemFlags::read_only() | MemFlags::host_write_only());
        let read_buf_flags = Some(MemFlags::write_only() | MemFlags::host_read_only());

        let write_buf: Buffer<ClFloat4> = Buffer::new(io_queue.clone(),
            write_buf_flags, work_size, None).unwrap();
        let read_buf: Buffer<ClFloat4> = Buffer::new(io_queue.clone(),
            read_buf_flags, work_size, None).unwrap();

        let program = Program::builder()
            .devices(device)
            .src(KERN_SRC)
            .build(&context).unwrap();

        let kern = Kernel::new("add", &program, kern_queue.clone()).unwrap()
            .gws(work_size)
            .arg_buf(&write_buf)
            .arg_vec(ClFloat4(100., 100., 100., 100.))
            .arg_buf(&read_buf);


        // (-1) INIT: With -500's:
        let mut fill_event = Event::empty();
        write_buf.cmd().fill(ClFloat4(-500., -500., -500., -500.), None).enew(&mut fill_event).enq().unwrap();

        // (0) WRITE: Write a bunch of 50's:
        let mut future_write_data = write_buf.cmd().map().flags(MapFlags::write_invalidate_region())
            .ewait(&fill_event)
            .enq_async().unwrap();

        let unmap_event = future_write_data.create_unmap_event().unwrap().clone();

        let write =
            future_write_data.and_then(move |mut data| {
                for _ in 0..1024 {
                    for val in data.iter_mut() {
                        *val = ClFloat4(50., 50., 50., 50.);
                    }
                }

                println!("Data has been written. ");

                data.enqueue_unmap::<(), ()>(None, None, None).unwrap();

                Ok(())
            })
                // .then(|_| Ok::<_, ()>(()))
            // .wait().unwrap();
        ;

        let spawned_write = thread_pool.spawn(write);

        io_queue.finish().unwrap();
        println!("I/O queue finished.");

        // write.wait().unwrap();
        // thread_pool.spawn(write).wait().unwrap();

        // (1) KERNEL: Run kernel (adds 100 to everything):
        let mut kern_event = Event::empty();

        kern.cmd()
            .enew(&mut kern_event)
            .ewait(&unmap_event)
            .enq().unwrap();

        // kern_queue.flush().unwrap();
        // println!("Kernel queue flushed.");

        kern_queue.finish().unwrap();
        println!("Kernel queue finished.");

        // (2) READ: Read results and verify them:
        let mut future_read_data = read_buf.cmd().map().flags(MapFlags::read())
            .ewait(&kern_event)
            .enq_async().unwrap();

        let unmap_event = future_read_data.create_unmap_event().unwrap().clone();
        // self.cmd_graph.set_cmd_event(cmd_idx, unmap_event_target.into()).unwrap();

        let tx_c = tx.clone();

        let verify =
            // .and_then(|_| {
                future_read_data.and_then(move |mut data| {
                    let mut val_count = 0usize;

                    for _ in 0..1024 {
                        for val in data.iter() {
                            let correct_val = ClFloat4(150., 150., 150., 150.);
                            if *val != correct_val {
                                return Err(format!("Result value mismatch: {:?} != {:?}", val, correct_val).into())
                            }
                            val_count += 1;
                        }
                    }

                    // print!(".");
                    println!("Verify done. ");

                    // Ok(val_count)
                    // Ok(tx.send(val_count).and_then(|_| Ok(()))) // <---- DOESN'T WORK (not sure why)
                    Ok(tx_c.send(val_count).wait())
                })
            // })
            .and_then(|_| Ok(()))
        ;

        io_queue.finish().unwrap();
        println!("I/O queue finished.");

        // verify.wait().unwrap();
        // thread_pool.spawn(verify).wait().unwrap();
        let spawned_verify = thread_pool.spawn(verify);

        let offload = spawned_write.join(spawned_verify);

        offloads.push(offload);

        print!("[.] ");
        std::io::stdout().flush().unwrap();
    }

    // std::io::stdout().flush().unwrap();

    let create_duration = chrono::Local::now() - start_time;

    print!("\n");

    kern_queue.flush().unwrap();
    io_queue.flush().unwrap();

    let enqueue_duration = chrono::Local::now() - start_time - create_duration;

    while let Some(offload) = offloads.try_pop() {
        println!("Waiting on offload");
        // kern_queue.flush().unwrap();
        // io_queue.flush().unwrap();

        offload.wait().unwrap();
    }

    // kern_queue.finish().unwrap();
    // io_queue.finish().unwrap();

    let run_duration = chrono::Local::now() - start_time - create_duration - enqueue_duration;

    print!("\n");

    let _ = tx;

    rx.close();

    for count in rx.wait() {
        // println!("Count: {}", count.unwrap());
        correct_val_count += count.unwrap();
    }

    let final_duration = chrono::Local::now() - start_time - create_duration - enqueue_duration - run_duration;


    printlnc!(yellow_bold: "All {} (float4) result values are correct! \n\
        Durations => | Create: {} | Enqueue: {} | Run: {} | Final: {} | ",
        correct_val_count, fmt_duration(create_duration),
        fmt_duration(enqueue_duration), fmt_duration(run_duration),
        fmt_duration(final_duration));
}







