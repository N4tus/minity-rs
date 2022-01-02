#![feature(maybe_uninit_uninit_array)]
#![feature(generic_const_exprs)]
#![feature(maybe_uninit_array_assume_init)]
#![feature(maybe_uninit_slice)]
#![feature(negative_impls)]
#![feature(maybe_uninit_extra)]
#![feature(slice_take)]
#![feature(allocator_api)]

use crate::array_vec::ArrayVec;
use crate::graphics::WGPURenderer;
use crate::objects::{load_model, LoadError, Model, UNIFORM_ALIGNMENT};
use crate::renderer::{
    CameraBuilder, LightRendererBuilder, ModelRendererBuilder, RayTracerBuilder,
};
use crate::window::Window;
use bitflags::bitflags;
use cgmath::SquareMatrix;
use egui::epaint::ClippedShape;
use egui::{CtxRef, Ui};
use egui_winit_platform::Platform;
use std::cell::RefCell;
use std::ops::Range;
use std::process::exit;
use tuple_list::tuple_list;
use winit::dpi::PhysicalSize;
use winit::event::{VirtualKeyCode, WindowEvent};

mod array_vec;
mod graphics;
mod objects;
mod renderer;
mod window;

const MAX_BIND_GROUPS: usize = wgpu_core::MAX_BIND_GROUPS;

trait RenderBackend {
    fn resize(&mut self, new_size: PhysicalSize<u32>);
    fn input(&mut self, event: &WindowEvent) -> bool;
    fn update(&mut self);
    fn init(&mut self);
    fn render(
        &mut self,
        scale_factor: f32,
        platform: &mut Platform,
        end_frame: &dyn Fn(&mut Platform) -> Vec<ClippedShape>,
    ) -> Result<UiActions, wgpu::SurfaceError>;
    fn size(&self) -> PhysicalSize<u32>;
}

#[derive(Debug, Default)]
struct UiActions {
    do_screenshot: bool,
    quit: bool,
}

trait RendererBuilder<Data> {
    type Output: Renderer<Data>;
    fn build(self, data: &mut Data, device: &wgpu::Device, size: PhysicalSize<u32>)
        -> Self::Output;
}

type Bga = ArrayVec<Range<usize>, MAX_BIND_GROUPS>;

#[allow(clippy::upper_case_acronyms)]
#[derive(Clone)]
pub(crate) struct WGPU<'a> {
    pub(crate) depth_view: &'a wgpu::TextureView,
    pub(crate) queue: &'a wgpu::Queue,
    pub(crate) command_encoder: &'a RefCell<wgpu::CommandEncoder>,
    pub(crate) view: &'a wgpu::TextureView,

    render_pipelines: &'a [Option<wgpu::RenderPipeline>],
    bind_groups: &'a [&'a wgpu::BindGroup],
    bind_group_association: &'a [Bga],
    // uniforms: &'a [Option<(&'static str, wgpu::BindGroup, wgpu::BindGroupLayout)>],
    // current_uniforms: &'a [(usize, [usize; wgpu_core::MAX_BIND_GROUPS])],
    renderer_index: usize,
}

impl<'a> WGPU<'a> {
    pub(crate) fn current_render_pipeline(&self) -> &'a Option<wgpu::RenderPipeline> {
        unsafe { self.render_pipelines.get_unchecked(self.renderer_index) }
    }
    pub(crate) fn current_uniforms(&self) -> impl Iterator<Item = &'a [&'a wgpu::BindGroup]> {
        unsafe {
            self.bind_group_association
                .get_unchecked(self.renderer_index)
        }
        .iter()
        .cloned()
        .map(|idx| unsafe { self.bind_groups.get_unchecked(idx) })
    }
}

struct CreateBindGroup {
    name: &'static str,
    groups: Vec<wgpu::BindGroup>,
    layout: wgpu::BindGroupLayout,
}

struct CreatePipeline {
    src: &'static str,
    layout: &'static [wgpu::VertexBufferLayout<'static>],
    uniforms: ArrayVec<&'static str, MAX_BIND_GROUPS>,
    topology: wgpu::PrimitiveTopology,
}

#[derive(Default)]
struct ShaderAction {
    create_bind_groups: Vec<CreateBindGroup>,
    create_pipeline: Option<CreatePipeline>,
}

trait Renderer<Data> {
    /// do never override this method
    fn visit(
        &self,
        data: &mut Data,
        actions: &mut [ShaderAction],
        index: usize,
        device: &wgpu::Device,
    ) {
        *unsafe { actions.get_unchecked_mut(index) } = self.shader(data, device);
    }
    fn update(&mut self, _data: &mut Data) {}
    fn render(&mut self, _data: &mut Data, _wgpu: WGPU) {}
    fn render_ui(
        &mut self,
        _data: &mut Data,
        _ctx: &CtxRef,
        _menu_ui: &mut Ui,
        _actions: &mut UiActions,
    ) {
    }
    fn input(&mut self, _data: &mut Data, _event: &WindowEvent) -> bool {
        false
    }
    fn shader(&self, _data: &mut Data, _device: &wgpu::Device) -> ShaderAction {
        Default::default()
    }
    fn resize(&mut self, _data: &mut Data, _size: PhysicalSize<u32>) {}
}

bitflags! {
   struct Dirty: u8 {
        const CAMERA     = 0b0000_0001;
        const LIGHT      = 0b0000_0010;
        const RAY_TRACER = 0b0000_0100;
        const MATERIAL   = 0b0000_1000;
    }
}

struct App {
    model: Option<Model>,
    bg: [f32; 3],
    view_proj: cgmath::Matrix4<f32>,
    light_data: cgmath::Point3<f32>,
    dirty: Dirty,
}

fn main() {
    env_logger::init();
    if wgpu::Limits::default().min_uniform_buffer_offset_alignment
        != UNIFORM_ALIGNMENT.try_into().unwrap()
    {
        log::error!(
            "min uniform buffer offset alignment is not {}",
            UNIFORM_ALIGNMENT
        );
        exit(1);
    }
    let model_result = load_model();
    let mut model = None;
    match model_result {
        Err(LoadError::NoModelToLoad) => log::error!("no model to load"),
        Err(LoadError::NativeError(err)) => log::error!("Error showing file dialog: {:?}", err),
        Err(LoadError::Other(err)) => log::error!("{}", err),
        Err(LoadError::TObjError(err)) => log::error!("Error loading obj file: {}", err),
        Err(LoadError::IOError(err)) => log::error!("Error loading file: {}", err),
        Err(LoadError::ImageError(err)) => log::error!("Error loading file: {}", err),
        Ok(m) => model = Some(m),
    }
    let window = Window::new();
    let model_renderer = ModelRendererBuilder;
    let camera = CameraBuilder {
        eye: (0.0, 0.0, 2.0).into(),
        target: (0.0, 0.0, 0.0).into(),
        fovy: 45.0,
        znear: 0.1,
        zfar: 100.0,
    };
    let ray_tracer = RayTracerBuilder;
    let light = LightRendererBuilder;
    let renderer = WGPURenderer::new(
        App {
            model,
            bg: [0.0; 3],
            view_proj: cgmath::Matrix4::identity(),
            light_data: cgmath::Point3::new(0.0, 0.5, 0.0),
            dirty: Dirty::empty(),
        },
        Some(VirtualKeyCode::R),
        &window,
        tuple_list!(model_renderer, camera, ray_tracer, light),
    );
    window.run(renderer);
}
