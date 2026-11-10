//! The first window's GPU context, looked for in a Vulkan instance first.
//!
//! Upstream creates one instance with `Backends::VULKAN | Backends::GL`
//! (`WgpuContext::instance`). Creating it loads and initializes every Vulkan
//! driver and every GL/EGL driver of the machine, and enumerating it opens an
//! adapter on each, before one is chosen (zed-industries/zed#63327,
//! zed-industries/zed#61439). Where Vulkan works, the GL half is never used.
//!
//! So the first window first tries an instance with Vulkan alone, ranked and
//! tested the way upstream does it (`WgpuContext::sort_adapters`, then
//! `WgpuContext::try_adapter_with_surface` on each in turn). Only the Vulkan
//! adapters that `sort_adapters` would put before every GL adapter are tried
//! there (see [`outranks_gl`]), so the adapter taken is the one upstream's
//! instance of both backends would have taken.
//!
//! When none of them configures the surface, the context is looked for again
//! in a new instance with a surface of its own, since a surface belongs to the
//! instance that created it:
//!
//! - an instance of GL alone, when every Vulkan adapter was tried and failed
//!   (a surface no Vulkan driver can present to), or when there was none (no
//!   Vulkan driver). The instance of both backends would have tried the same
//!   Vulkan adapters first, failed the same way, and gone on to the GL
//!   adapters in the same order, so GL alone takes the same adapter;
//! - an instance of Vulkan and GL, as upstream's, when a Vulkan adapter was
//!   left untried because a GL adapter may rank before it (a software, virtual
//!   or unknown Vulkan device): there the two backends' adapters have to be
//!   ranked together, as before.
//!
//! The Vulkan instance stays alive until then. NVIDIA's Vulkan and GL drivers
//! share libraries, which an instance of both backends initializes once; with
//! the Vulkan instance dropped first, the GL instance would initialize them
//! again, and a machine where Vulkan cannot present would start slower than
//! before.
//!
//! `WGPU_BACKEND`, wgpu's own variable, narrows the instance to the backends it
//! names among Vulkan and GL, without falling back to the others.

use std::ops::ControlFlow;
use std::rc::Rc;
use std::sync::Arc;

use raw_window_handle::{HasDisplayHandle, HasWindowHandle};

use crate::wgpu_renderer::create_surface;
use crate::{
    CompositorGpuHint, GpuContext, WgpuAtlas, WgpuContext, WgpuRenderer, WgpuSurfaceConfig,
};

/// The backends of upstream's instance, `WgpuContext::instance`.
const UPSTREAM_BACKENDS: wgpu::Backends = wgpu::Backends::VULKAN.union(wgpu::Backends::GL);

/// How the first window's context is looked for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Plan {
    /// Vulkan alone, then GL alone or Vulkan and GL.
    VulkanFirst,
    /// One instance of these backends, as `WGPU_BACKEND` asks.
    Only(wgpu::Backends),
}

impl Plan {
    fn from_env() -> Self {
        Self::new(wgpu::Backends::from_env())
    }

    /// `requested` is what `WGPU_BACKEND` names; backends other than Vulkan
    /// and GL are left out, and a request for none of them is ignored.
    fn new(requested: Option<wgpu::Backends>) -> Self {
        match requested.map(|backends| backends & UPSTREAM_BACKENDS) {
            Some(backends) if !backends.is_empty() => Self::Only(backends),
            _ => Self::VulkanFirst,
        }
    }
}

/// Creates the renderer of the first window, its context with it: `Break`
/// with the renderer. For any later window, and when `WGPU_BACKEND` asks for
/// upstream's backends, hands `config` back for upstream's code to go on
/// with: `Continue`.
pub(crate) fn new_renderer<W>(
    gpu_context: &GpuContext,
    window: &W,
    config: WgpuSurfaceConfig,
    compositor_gpu: Option<CompositorGpuHint>,
) -> anyhow::Result<ControlFlow<WgpuRenderer, WgpuSurfaceConfig>>
where
    W: HasWindowHandle + HasDisplayHandle + std::fmt::Debug + Send + Sync + Clone + 'static,
{
    if gpu_context.borrow().is_some() {
        return Ok(ControlFlow::Continue(config));
    }
    let plan = Plan::from_env();
    if plan == Plan::Only(UPSTREAM_BACKENDS) {
        return Ok(ControlFlow::Continue(config));
    }
    let target = window
        .window_handle()
        .map_err(|e| anyhow::anyhow!("Failed to get window handle: {e}"))
        .map(|handle| (window, handle.as_raw()))?;

    let (context, surface) = match plan {
        Plan::Only(backends) => {
            log::info!("WGPU_BACKEND: creating a {backends:?} instance");
            context_in(backends, target, &config, compositor_gpu)?
        }
        Plan::VulkanFirst => {
            // Kept until the context is made: a GL driver that shares its
            // libraries with the Vulkan driver (NVIDIA's) then finds them
            // loaded and initialized, as in an instance of both backends.
            let vulkan = new_instance(wgpu::Backends::VULKAN, target.0);
            match vulkan_context(&vulkan, target, &config, compositor_gpu.as_ref()) {
                Ok(found) => found,
                Err(VulkanMiss::GlAlone) => {
                    log::info!("No Vulkan adapter configured the surface; trying GL");
                    context_in(wgpu::Backends::GL, target, &config, compositor_gpu)?
                }
                Err(VulkanMiss::RankWithGl) => {
                    log::info!("A GL adapter may rank before a Vulkan one; trying Vulkan and GL");
                    context_in(UPSTREAM_BACKENDS, target, &config, compositor_gpu)?
                }
            }
        }
    };

    // From here on, as upstream's `WgpuRenderer::new` goes on once adapter
    // selection has configured the surface.
    let mut context_ref = gpu_context.borrow_mut();
    let context = context_ref.insert(context);
    let atlas = Arc::new(WgpuAtlas::from_context(context));
    WgpuRenderer::new_internal(
        Some(Rc::clone(gpu_context)),
        context,
        surface,
        config,
        true,
        compositor_gpu,
        atlas,
    )
    .map(ControlFlow::Break)
}

/// Why no Vulkan adapter was taken, and where to look next.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum VulkanMiss {
    /// Every Vulkan adapter, if any, was tried and failed: an instance of GL
    /// alone takes what upstream's instance of both backends would.
    GlAlone,
    /// A Vulkan adapter was left untried, as a GL adapter may rank before it:
    /// both backends' adapters have to be ranked together.
    RankWithGl,
}

impl VulkanMiss {
    /// Where to look once every adapter that outranks GL failed. `outranks_gl`
    /// gives each adapter of the Vulkan instance.
    fn after(outranks_gl: impl IntoIterator<Item = bool>) -> Self {
        if outranks_gl.into_iter().all(|outranks| outranks) {
            Self::GlAlone
        } else {
            Self::RankWithGl
        }
    }
}

/// A window, and its handle to make surfaces for.
type Target<'a, W> = (&'a W, raw_window_handle::RawWindowHandle);

/// The context and its configured surface, from an instance of Vulkan alone.
fn vulkan_context<W>(
    instance: &wgpu::Instance,
    target: Target<'_, W>,
    config: &WgpuSurfaceConfig,
    compositor_gpu: Option<&CompositorGpuHint>,
) -> Result<(WgpuContext, wgpu::Surface<'static>), VulkanMiss>
where
    W: HasDisplayHandle + std::fmt::Debug + Send + Sync + Clone + 'static,
{
    // Without a Vulkan driver the instance has no backend to make a surface
    // with, and upstream's instance would have no Vulkan adapter.
    let surface = create_surface(instance, target.1).map_err(|error| {
        log::info!("No Vulkan surface: {error}");
        VulkanMiss::GlAlone
    })?;
    let (adapter, device, queue, dual_source_blending, color_texture_format) = gpui::block_on(
        select_vulkan_adapter(instance, &surface, config, compositor_gpu),
    )?;
    let context = WgpuContext::from_native_adapter_and_device(
        instance.clone(),
        adapter,
        device,
        queue,
        dual_source_blending,
        color_texture_format,
    );
    Ok((context, surface))
}

/// The context and its configured surface, from an instance of `backends`,
/// as upstream's `WgpuRenderer::new` finds them.
fn context_in<W>(
    backends: wgpu::Backends,
    target: Target<'_, W>,
    config: &WgpuSurfaceConfig,
    compositor_gpu: Option<CompositorGpuHint>,
) -> anyhow::Result<(WgpuContext, wgpu::Surface<'static>)>
where
    W: HasDisplayHandle + std::fmt::Debug + Send + Sync + Clone + 'static,
{
    let instance = new_instance(backends, target.0);
    let surface = create_surface(&instance, target.1)
        .map_err(|e| anyhow::anyhow!("Failed to create surface: {e}"))?;
    let context = WgpuContext::new(instance, &surface, config, compositor_gpu)?;
    Ok((context, surface))
}

/// `WgpuContext::instance`'s instance, with `backends`.
fn new_instance<W>(backends: wgpu::Backends, window: &W) -> wgpu::Instance
where
    W: HasDisplayHandle + std::fmt::Debug + Send + Sync + Clone + 'static,
{
    wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends,
        flags: wgpu::InstanceFlags::default(),
        backend_options: wgpu::BackendOptions::default(),
        memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
        display: Some(Box::new(window.clone())),
    })
}

/// `WgpuContext::select_adapter_and_device` on a Vulkan instance, keeping to
/// the adapters no GL adapter could have been ranked before.
async fn select_vulkan_adapter(
    instance: &wgpu::Instance,
    surface: &wgpu::Surface<'_>,
    config: &WgpuSurfaceConfig,
    compositor_gpu: Option<&CompositorGpuHint>,
) -> Result<
    (
        wgpu::Adapter,
        wgpu::Device,
        wgpu::Queue,
        bool,
        wgpu::TextureFormat,
    ),
    VulkanMiss,
> {
    let device_id_filter = WgpuContext::device_id_filter();
    let mut adapters = instance.enumerate_adapters(wgpu::Backends::VULKAN).await;
    WgpuContext::sort_adapters(&mut adapters, device_id_filter, compositor_gpu);

    log::info!("Found {} Vulkan adapter(s):", adapters.len());
    for adapter in &adapters {
        let info = adapter.get_info();
        log::info!(
            "  - {} (vendor={:#06x}, device={:#06x}, type={:?}){}",
            info.name,
            info.vendor,
            info.device,
            info.device_type,
            if outranks_gl(&info, device_id_filter, compositor_gpu) {
                ""
            } else {
                ", a GL adapter may rank before it"
            },
        );
    }

    let miss = VulkanMiss::after(
        adapters
            .iter()
            .map(|adapter| outranks_gl(&adapter.get_info(), device_id_filter, compositor_gpu)),
    );
    for adapter in adapters {
        let info = adapter.get_info();
        if !outranks_gl(&info, device_id_filter, compositor_gpu) {
            continue;
        }
        log::info!("Testing adapter: {} ({:?})...", info.name, info.backend);
        match WgpuContext::try_adapter_with_surface(&adapter, surface, config).await {
            Ok((device, queue, dual_source_blending, color_texture_format)) => {
                log::info!(
                    "Selected GPU (passed configuration test): {} ({:?})",
                    info.name,
                    info.backend
                );
                return Ok((
                    adapter,
                    device,
                    queue,
                    dual_source_blending,
                    color_texture_format,
                ));
            }
            Err(error) => {
                log::info!(
                    "  Adapter {} ({:?}) failed: {error}, trying next...",
                    info.name,
                    info.backend
                );
            }
        }
    }
    Err(miss)
}

/// Whether `WgpuContext::sort_adapters` ranks this Vulkan adapter before any
/// adapter wgpu's GL backend can report. GL adapters have device id 0, so they
/// match neither `ZED_DEVICE_ID` nor the compositor's GPU, and wgpu types them
/// from their renderer string as integrated, other or CPU, never discrete; an
/// integrated GL adapter and an integrated Vulkan one tie on type, and the tie
/// goes to Vulkan. The test is upstream's ranking key, which it must follow.
fn outranks_gl(
    info: &wgpu::AdapterInfo,
    device_id_filter: Option<u32>,
    compositor_gpu: Option<&CompositorGpuHint>,
) -> bool {
    let device_known = info.device != 0;
    let user_override = device_id_filter.is_some_and(|id| device_known && info.device == id);
    let compositor_match = compositor_gpu.is_some_and(|hint| {
        device_known && info.vendor == hint.vendor_id && info.device == hint.device_id
    });
    user_override
        || compositor_match
        || matches!(
            info.device_type,
            wgpu::DeviceType::DiscreteGpu | wgpu::DeviceType::IntegratedGpu
        )
}

#[cfg(test)]
mod tests {
    use super::{CompositorGpuHint, Plan, UPSTREAM_BACKENDS, VulkanMiss, outranks_gl};
    use wgpu::{AdapterInfo, Backend, Backends, DeviceType};

    const TYPES: [DeviceType; 5] = [
        DeviceType::DiscreteGpu,
        DeviceType::IntegratedGpu,
        DeviceType::Other,
        DeviceType::VirtualGpu,
        DeviceType::Cpu,
    ];
    /// The types wgpu's GL backend gives its adapters.
    const GL_TYPES: [DeviceType; 3] = [
        DeviceType::IntegratedGpu,
        DeviceType::Other,
        DeviceType::Cpu,
    ];

    fn adapter(backend: Backend, device_type: DeviceType, vendor: u32, device: u32) -> AdapterInfo {
        AdapterInfo {
            vendor,
            device,
            ..AdapterInfo::new(device_type, backend)
        }
    }

    fn gl(device_type: DeviceType) -> AdapterInfo {
        adapter(Backend::Gl, device_type, 0x10de, 0)
    }

    /// A copy of the key `WgpuContext::sort_adapters` sorts by.
    fn upstream_key(
        info: &AdapterInfo,
        device_id_filter: Option<u32>,
        compositor_gpu: Option<&CompositorGpuHint>,
    ) -> (u8, u8, u8, u8) {
        let device_known = info.device != 0;
        let user_override: u8 = match device_id_filter {
            Some(id) if device_known && info.device == id => 0,
            _ => 1,
        };
        let compositor_match: u8 = match compositor_gpu {
            Some(hint)
                if device_known
                    && info.vendor == hint.vendor_id
                    && info.device == hint.device_id =>
            {
                0
            }
            _ => 1,
        };
        let type_priority: u8 = match info.device_type {
            DeviceType::DiscreteGpu => 0,
            DeviceType::IntegratedGpu => 1,
            DeviceType::Other => 2,
            DeviceType::VirtualGpu => 3,
            DeviceType::Cpu => 4,
        };
        let backend_priority: u8 = match info.backend {
            Backend::Vulkan | Backend::Metal | Backend::Dx12 => 0,
            _ => 1,
        };
        (
            user_override,
            compositor_match,
            type_priority,
            backend_priority,
        )
    }

    #[test]
    fn a_vulkan_adapter_is_tried_alone_exactly_when_no_gl_adapter_ranks_before_it() {
        let hint = CompositorGpuHint {
            vendor_id: 0x10de,
            device_id: 0x2482,
        };
        for device_id_filter in [None, Some(0x2482), Some(0x1234)] {
            for compositor_gpu in [None, Some(&hint)] {
                for device_type in TYPES {
                    for device in [0, 0x2482] {
                        let vulkan = adapter(Backend::Vulkan, device_type, 0x10de, device);
                        let vulkan_key = upstream_key(&vulkan, device_id_filter, compositor_gpu);
                        let first = GL_TYPES.iter().all(|&gl_type| {
                            vulkan_key
                                < upstream_key(&gl(gl_type), device_id_filter, compositor_gpu)
                        });
                        assert_eq!(
                            outranks_gl(&vulkan, device_id_filter, compositor_gpu),
                            first,
                            "{vulkan:?}, ZED_DEVICE_ID {device_id_filter:?}, compositor {:?}",
                            compositor_gpu.map(|hint| (hint.vendor_id, hint.device_id)),
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn hardware_vulkan_adapters_are_taken_without_gl() {
        let discrete = adapter(Backend::Vulkan, DeviceType::DiscreteGpu, 0x10de, 0x2482);
        let integrated = adapter(Backend::Vulkan, DeviceType::IntegratedGpu, 0x8086, 0x46a6);
        assert!(outranks_gl(&discrete, None, None));
        assert!(outranks_gl(&integrated, None, None));
    }

    #[test]
    fn software_and_virtual_vulkan_adapters_fall_back_to_gl() {
        // lavapipe: a hardware GL adapter (integrated, or "other" for a
        // discrete GPU GL cannot tell apart) ranks before it upstream.
        let lavapipe = adapter(Backend::Vulkan, DeviceType::Cpu, 0x10005, 0);
        let virtio = adapter(Backend::Vulkan, DeviceType::VirtualGpu, 0x1af4, 0x1050);
        let other = adapter(Backend::Vulkan, DeviceType::Other, 0x1234, 0x5678);
        for info in [lavapipe, virtio, other] {
            assert!(!outranks_gl(&info, None, None), "{info:?}");
        }
    }

    #[test]
    fn a_pinned_vulkan_adapter_is_taken_without_gl_whatever_its_type() {
        let virtio = adapter(Backend::Vulkan, DeviceType::VirtualGpu, 0x1af4, 0x1050);
        let hint = CompositorGpuHint {
            vendor_id: 0x1af4,
            device_id: 0x1050,
        };
        assert!(outranks_gl(&virtio, Some(0x1050), None));
        assert!(outranks_gl(&virtio, None, Some(&hint)));
        assert!(!outranks_gl(&virtio, Some(0x1051), None));

        // A device id of 0 is unknown, and never matches.
        let lavapipe = adapter(Backend::Vulkan, DeviceType::Cpu, 0x10005, 0);
        let hint = CompositorGpuHint {
            vendor_id: 0x10005,
            device_id: 0,
        };
        assert!(!outranks_gl(&lavapipe, Some(0), Some(&hint)));
    }

    #[test]
    fn gl_alone_follows_a_miss_only_when_every_vulkan_adapter_was_tried() {
        let discrete = adapter(Backend::Vulkan, DeviceType::DiscreteGpu, 0x10de, 0x2482);
        let integrated = adapter(Backend::Vulkan, DeviceType::IntegratedGpu, 0x8086, 0x46a6);
        let lavapipe = adapter(Backend::Vulkan, DeviceType::Cpu, 0x10005, 0);
        let after = |adapters: &[&AdapterInfo]| {
            VulkanMiss::after(adapters.iter().map(|info| outranks_gl(info, None, None)))
        };
        // No Vulkan driver, or none that can present to the surface.
        assert_eq!(after(&[]), VulkanMiss::GlAlone);
        assert_eq!(after(&[&discrete]), VulkanMiss::GlAlone);
        assert_eq!(after(&[&discrete, &integrated]), VulkanMiss::GlAlone);
        // lavapipe was not tried: a hardware GL adapter may rank before it,
        // and if none configures the surface it may still be the one taken.
        assert_eq!(after(&[&lavapipe]), VulkanMiss::RankWithGl);
        assert_eq!(after(&[&discrete, &lavapipe]), VulkanMiss::RankWithGl);
    }

    #[test]
    fn wgpu_backend_narrows_the_instance_to_vulkan_or_gl() {
        let requested = |list: &str| Plan::new(Some(Backends::from_comma_list(list)));
        assert_eq!(Plan::new(None), Plan::VulkanFirst);
        assert_eq!(requested("vulkan"), Plan::Only(Backends::VULKAN));
        assert_eq!(requested("gl"), Plan::Only(Backends::GL));
        assert_eq!(requested("vulkan,gl"), Plan::Only(UPSTREAM_BACKENDS));
        assert_eq!(requested("gl,dx12"), Plan::Only(Backends::GL));
        assert_eq!(requested("dx12"), Plan::VulkanFirst);
        assert_eq!(requested(""), Plan::VulkanFirst);
    }
}
