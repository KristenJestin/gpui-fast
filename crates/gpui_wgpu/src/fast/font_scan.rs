//! The system fonts, scanned on a thread of their own.
//!
//! [`cosmic_text::FontSystem::new`] reads every font file fontconfig knows
//! about before it returns: a few milliseconds with a few hundred faces, whole
//! seconds with thousands. Upstream calls it when the Linux platform starts,
//! on the main thread, so a window cannot open before the scan has ended.
//!
//! [`FontScan`] holds the text system's state with a placeholder font system
//! and runs the scan on a thread. Every access to the state goes through
//! [`FontScan`]'s `Deref`, which waits for the scan and swaps its font system in
//! first, so no query ever sees a partial database: the first shaping, font id
//! or font name query waits for the scan, and later ones find it done. The
//! scan overlaps whatever the application does before its first text query:
//! connecting to the display server, opening the window, starting the GPU.
//!
//! Fonts the application adds meanwhile are kept, in the order they came, and
//! loaded after the system fonts once the scan is done, which is when they
//! would have been loaded had the scan run on the main thread. A panic in the
//! scan is raised again on the thread that waits for it, and on every access
//! after that.

use std::{
    borrow::Cow,
    ops::Deref,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    thread::JoinHandle,
};

use anyhow::Result;
use cosmic_text::FontSystem;
use gpui::MissingGlyphSink;
use parking_lot::{Mutex, RwLock};

use crate::cosmic_text_system::CosmicTextSystemState;

/// A text system's state, and the scan of the system fonts its font system
/// waits for.
pub(crate) struct FontScan {
    state: RwLock<CosmicTextSystemState>,
    scan: Mutex<Scan>,
    /// Set once `scan` is [`Scan::Done`], so that waiting for a finished scan
    /// takes no lock.
    done: AtomicBool,
}

enum Scan {
    Running {
        thread: JoinHandle<FontSystem>,
        /// What [`add_fonts`] was given while the scan ran, in order.
        fonts: Vec<Cow<'static, [u8]>>,
    },
    Done,
    Panicked,
}

/// A font system with no fonts, which costs next to nothing to make, for the
/// state to hold until the scan's replaces it.
pub(crate) fn empty_font_system() -> FontSystem {
    FontSystem::new_with_locale_and_db("en-US".to_string(), cosmic_text::fontdb::Database::new())
}

impl FontScan {
    /// Scans the system fonts on a thread into `state`'s font system, or on
    /// this one where no thread can be started.
    pub(crate) fn system_fonts(state: RwLock<CosmicTextSystemState>) -> Self {
        match spawn(FontSystem::new) {
            Ok(thread) => Self::running(state, thread),
            Err(error) => {
                log::warn!("scanning the system fonts on this thread: {error}");
                state.write().font_system = FontSystem::new();
                Self::ready(state)
            }
        }
    }

    /// Holds a state whose font system is complete.
    pub(crate) fn ready(state: RwLock<CosmicTextSystemState>) -> Self {
        Self {
            state,
            scan: Mutex::new(Scan::Done),
            done: AtomicBool::new(true),
        }
    }

    fn running(state: RwLock<CosmicTextSystemState>, thread: JoinHandle<FontSystem>) -> Self {
        Self {
            state,
            scan: Mutex::new(Scan::Running {
                thread,
                fonts: Vec::new(),
            }),
            done: AtomicBool::new(false),
        }
    }

    fn wait(&self) {
        if self.done.load(Ordering::Acquire) {
            return;
        }
        let mut scan = self.scan.lock();
        match std::mem::replace(&mut *scan, Scan::Done) {
            Scan::Done => {}
            Scan::Panicked => {
                *scan = Scan::Panicked;
                drop(scan);
                panic!("the scan of the system fonts panicked");
            }
            Scan::Running { thread, fonts } => match thread.join() {
                Ok(font_system) => {
                    let mut state = self.state.write();
                    state.font_system = font_system;
                    if !fonts.is_empty()
                        && let Err(error) = state.add_fonts(fonts)
                    {
                        log::error!("adding fonts after the system font scan: {error:?}");
                    }
                }
                Err(panic) => {
                    *scan = Scan::Panicked;
                    drop(scan);
                    std::panic::resume_unwind(panic);
                }
            },
        }
        self.done.store(true, Ordering::Release);
    }
}

impl Deref for FontScan {
    type Target = RwLock<CosmicTextSystemState>;

    /// Waits for the scan, so that what reads the state sees every font.
    fn deref(&self) -> &Self::Target {
        self.wait();
        &self.state
    }
}

/// Adds fonts, or keeps them for when the scan is done.
pub(crate) fn add_fonts(scan: &FontScan, fonts: Vec<Cow<'static, [u8]>>) -> Result<()> {
    if let Scan::Running { fonts: queued, .. } = &mut *scan.scan.lock() {
        queued.extend(fonts);
        return Ok(());
    }
    scan.write().add_fonts(fonts)
}

/// Sets where missing glyphs are reported, without waiting for the scan: the
/// sink is only read while shaping, which waits for it.
pub(crate) fn set_missing_glyph_sink(scan: &FontScan, sink: Option<Arc<dyn MissingGlyphSink>>) {
    scan.state.write().missing_glyph_sink = sink;
}

fn spawn(
    scan: impl FnOnce() -> FontSystem + Send + 'static,
) -> std::io::Result<JoinHandle<FontSystem>> {
    std::thread::Builder::new()
        .name("gpui-font-scan".to_string())
        .spawn(scan)
}

#[cfg(test)]
mod tests {
    use std::{
        borrow::Cow,
        panic::{AssertUnwindSafe, catch_unwind},
        sync::{Arc, mpsc},
        thread,
        time::Duration,
    };

    use cosmic_text::FontSystem;
    use gpui::{FontRun, PlatformTextSystem as _};

    use super::{FontScan, spawn};
    use crate::CosmicTextSystem;

    const IBM_PLEX: &[u8] =
        include_bytes!("../../../../assets/fonts/ibm-plex-sans/IBMPlexSans-Regular.ttf");
    const LILEX: &[u8] = include_bytes!("../../../../assets/fonts/lilex/Lilex-Regular.ttf");
    const LILEX_BOLD: &[u8] = include_bytes!("../../../../assets/fonts/lilex/Lilex-Bold.ttf");

    /// A font system standing for the system's, holding `fonts`.
    fn system_fonts(fonts: &[&'static [u8]]) -> FontSystem {
        let mut db = cosmic_text::fontdb::Database::new();
        for font in fonts {
            db.load_font_data(font.to_vec());
        }
        FontSystem::new_with_locale_and_db("en-US".to_string(), db)
    }

    /// A text system whose scan returns `scan`'s font system.
    fn scanning(scan: impl FnOnce() -> FontSystem + Send + 'static) -> CosmicTextSystem {
        let FontScan { state, .. } = CosmicTextSystem::new_without_system_fonts("IBM Plex Sans").0;
        CosmicTextSystem(FontScan::running(state, spawn(scan).unwrap()))
    }

    /// A text system whose scan returns `fonts` once `release` is sent to.
    fn gated(fonts: &'static [&'static [u8]]) -> (CosmicTextSystem, mpsc::Sender<()>) {
        let (release, released) = mpsc::channel();
        let text_system = scanning(move || {
            released.recv().ok();
            system_fonts(fonts)
        });
        (text_system, release)
    }

    fn is_done(text_system: &CosmicTextSystem) -> bool {
        text_system
            .0
            .done
            .load(std::sync::atomic::Ordering::Acquire)
    }

    fn face_families(text_system: &CosmicTextSystem) -> Vec<String> {
        text_system
            .0
            .read()
            .font_system
            .db()
            .faces()
            .map(|face| face.families[0].0.clone())
            .collect()
    }

    #[test]
    fn queries_wait_for_the_scan() {
        let (text_system, release) = gated(&[IBM_PLEX]);
        let text_system = Arc::new(text_system);
        let (sender, answers) = mpsc::channel();
        let query = thread::spawn({
            let text_system = text_system.clone();
            move || sender.send(text_system.all_font_names()).unwrap()
        });

        assert!(
            answers.recv_timeout(Duration::from_millis(100)).is_err(),
            "a query answered before the scan was done"
        );
        assert!(!is_done(&text_system));

        release.send(()).unwrap();
        let names = answers.recv_timeout(Duration::from_secs(10)).unwrap();
        query.join().unwrap();
        assert_eq!(names, ["IBM Plex Sans"]);
        assert!(is_done(&text_system));
    }

    #[test]
    fn a_query_before_the_scan_starts_sees_every_font() {
        // The scan thread may not have run at all when the query comes.
        let text_system = scanning(|| system_fonts(&[IBM_PLEX]));
        let font_id = text_system.font_id(&gpui::font("IBM Plex Sans")).unwrap();
        let text = "hello";
        let layout = text_system.layout_line(
            text,
            gpui::px(14.),
            &[FontRun {
                len: text.len(),
                font_id,
            }],
        );
        assert_eq!(layout.len, text.len());
        assert!(layout.runs.iter().any(|run| !run.glyphs.is_empty()));
    }

    #[test]
    fn fonts_added_during_the_scan_are_loaded_after_the_system_fonts_in_order() {
        let (text_system, release) = gated(&[IBM_PLEX]);
        text_system.add_fonts(vec![Cow::Borrowed(LILEX)]).unwrap();
        text_system
            .add_fonts(vec![Cow::Borrowed(LILEX_BOLD), Cow::Borrowed(IBM_PLEX)])
            .unwrap();
        assert!(!is_done(&text_system), "adding fonts waited for the scan");
        assert!(text_system.0.state.read().font_system.db().is_empty());

        release.send(()).unwrap();
        assert_eq!(
            face_families(&text_system),
            ["IBM Plex Sans", "Lilex", "Lilex", "IBM Plex Sans"]
        );
        assert_eq!(text_system.all_font_names(), ["IBM Plex Sans", "Lilex"]);
        let bold = text_system
            .font_id(&gpui::font("Lilex").bold())
            .and_then(|font_id| text_system.font_weight_and_style(font_id))
            .unwrap();
        assert_eq!(bold.0, gpui::FontWeight::BOLD);

        // Once the scan is done, fonts are added at once.
        text_system.add_fonts(vec![Cow::Borrowed(LILEX)]).unwrap();
        assert_eq!(face_families(&text_system).len(), 5);
    }

    #[test]
    fn the_missing_glyph_sink_is_set_without_waiting() {
        #[derive(Default)]
        struct Sink(parking_lot::Mutex<Vec<gpui::MissingGlyph>>);
        impl gpui::MissingGlyphSink for Sink {
            fn report(&self, missing_glyphs: Vec<gpui::MissingGlyph>) {
                self.0.lock().extend(missing_glyphs);
            }
        }

        let (text_system, release) = gated(&[IBM_PLEX]);
        let sink = Arc::new(Sink::default());
        text_system.set_missing_glyph_sink(Some(sink.clone()));
        assert!(
            !is_done(&text_system),
            "setting the sink waited for the scan"
        );

        release.send(()).unwrap();
        let font_id = text_system.font_id(&gpui::font("IBM Plex Sans")).unwrap();
        let text = "\u{754c}";
        text_system.layout_line(
            text,
            gpui::px(14.),
            &[FontRun {
                len: text.len(),
                font_id,
            }],
        );
        let reported = sink.0.lock();
        assert_eq!(reported.len(), 1);
        assert_eq!(reported[0].grapheme(), text);
    }

    #[test]
    fn a_panic_in_the_scan_reaches_every_query() {
        let text_system = scanning(|| panic!("no fonts here"));

        let first = catch_unwind(AssertUnwindSafe(|| text_system.all_font_names()))
            .expect_err("the scan's panic was swallowed");
        assert_eq!(first.downcast_ref::<&str>(), Some(&"no fonts here"));

        let later = catch_unwind(AssertUnwindSafe(|| {
            text_system.font_id(&gpui::font("IBM Plex Sans"))
        }))
        .expect_err("a query after the panic went on with no fonts");
        assert_eq!(
            later.downcast_ref::<&str>(),
            Some(&"the scan of the system fonts panicked")
        );
        let added = catch_unwind(AssertUnwindSafe(|| {
            text_system.add_fonts(vec![Cow::Borrowed(LILEX)])
        }));
        assert!(added.is_err(), "fonts were added after the scan panicked");
    }

    #[test]
    fn concurrent_first_queries_wait_for_one_scan() {
        let (text_system, release) = gated(&[IBM_PLEX, LILEX]);
        text_system
            .add_fonts(vec![Cow::Borrowed(LILEX_BOLD)])
            .unwrap();
        let text_system = Arc::new(text_system);
        let queries = (0..4)
            .map(|_| {
                let text_system = text_system.clone();
                thread::spawn(move || text_system.all_font_names())
            })
            .collect::<Vec<_>>();
        release.send(()).unwrap();
        for query in queries {
            assert_eq!(query.join().unwrap(), ["IBM Plex Sans", "Lilex"]);
        }
        assert_eq!(face_families(&text_system).len(), 3);
    }
}
