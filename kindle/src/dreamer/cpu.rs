//! Bounded CPU row parallelism for independent sampling and scalar decoding.
//! GPU work stays on the calling thread; this does not split actor and learner.

use std::sync::OnceLock;

fn pool() -> &'static rayon::ThreadPool {
    static POOL: OnceLock<rayon::ThreadPool> = OnceLock::new();
    POOL.get_or_init(|| {
        let threads = std::thread::available_parallelism().map_or(1, |n| n.get().min(8));
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .thread_name(|index| format!("kindle-cpu-{index}"))
            .build()
            .expect("create Kindle CPU workers")
    })
}

pub fn threads() -> usize {
    pool().current_num_threads()
}

pub fn parallel<T: Send>(work: impl FnOnce() -> T + Send) -> T {
    pool().install(work)
}
