//! Timings of the first window's creation, asked for with `GPUI_TIMING`: marks
//! taken along the way, printed in one block on the standard error once the
//! first frame is presented, in the format of Rature's `RATURE_TIMING` block.
//! Without the variable, a mark is one atomic load. Bench instrumentation.

use std::borrow::Cow;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock, PoisonError};
use std::time::{Duration, Instant};

/// Whether `GPUI_TIMING` asks for the timings (set, neither empty nor `0`).
static ASKED: OnceLock<bool> = OnceLock::new();
/// The platform's `open_window` was entered: the marks count from there.
static STARTED: OnceLock<Instant> = OnceLock::new();
/// What happened, when (since `STARTED`), and how long it took when it is a span.
static MARKS: Mutex<Vec<(Cow<'static, str>, Duration, Option<Duration>)>> =
    Mutex::new(Vec::new());
/// The block is printed: later marks are not kept.
static PRINTED: AtomicBool = AtomicBool::new(false);

fn asked() -> bool {
    *ASKED.get_or_init(|| {
        std::env::var_os("GPUI_TIMING").is_some_and(|value| !value.is_empty() && value != "0")
    })
}

/// Whether marks are kept now.
pub fn on() -> bool {
    asked() && !PRINTED.load(Ordering::Relaxed)
}

/// The platform's `open_window` is entered: the marks count from here. Only the
/// first call counts.
pub fn start() {
    if on() {
        let _ = STARTED.set(Instant::now());
    }
}

/// `what` happens now.
pub fn mark(what: impl Into<Cow<'static, str>>) {
    if on() {
        keep(what.into(), None);
    }
}

/// `what` took `took`, and ends now.
pub fn span(what: impl Into<Cow<'static, str>>, took: Duration) {
    if on() {
        keep(what.into(), Some(took));
    }
}

/// Times `work` as `what`.
pub fn time<T>(what: impl Into<Cow<'static, str>>, work: impl FnOnce() -> T) -> T {
    if !on() {
        return work();
    }
    let begun = Instant::now();
    let done = work();
    span(what, begun.elapsed());
    done
}

/// Times `work` as what `what` names, `what` being called only when timing.
pub fn time_named<T>(what: impl FnOnce() -> String, work: impl FnOnce() -> T) -> T {
    if !on() {
        return work();
    }
    let begun = Instant::now();
    let done = work();
    span(what(), begun.elapsed());
    done
}

fn keep(what: Cow<'static, str>, took: Option<Duration>) {
    let at = STARTED.get_or_init(Instant::now).elapsed();
    MARKS
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .push((what, at, took));
}

/// The first frame was presented: prints the marks taken so far, once.
pub fn presented() {
    if !on() {
        return;
    }
    mark("first frame presented");
    if PRINTED.swap(true, Ordering::Relaxed) {
        return;
    }
    let marks = std::mem::take(&mut *MARKS.lock().unwrap_or_else(PoisonError::into_inner));
    let ms = |d: Duration| d.as_secs_f64() * 1000.;
    let mut out = String::from(
        "gpui: first window, in ms since the platform's open_window (+ since the mark above)\n",
    );
    let mut last = Duration::ZERO;
    for (what, at, took) in marks {
        let took = took.map_or(String::new(), |t| format!(" (took {:.2})", ms(t)));
        out.push_str(&format!(
            "  {:>8.1}  {:>+7.1}  {what}{took}\n",
            ms(at),
            ms(at.saturating_sub(last))
        ));
        last = at;
    }
    out.push_str("gpui: end\n");
    eprint!("{out}");
}
