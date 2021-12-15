#![feature(maybe_uninit_uninit_array)]
#![feature(generic_const_exprs)]
#![feature(maybe_uninit_array_assume_init)]
#![feature(maybe_uninit_slice)]
#![feature(negative_impls)]

use crate::graphics::WGPURenderer;
use crate::objects::{load_model, LoadError, Model};
use crate::renderer::{CameraBuilder, ModelRendererBuilder};
use crate::window::Window;
use egui::paint::ClippedShape;
use egui::{CtxRef, Ui};
use egui_winit_platform::Platform;
use std::cell::RefCell;
use tuple_list::tuple_list;
use wgpu::{BindGroup, BindGroupLayout, CommandEncoder, RenderPipeline, TextureView};
use winit::dpi::PhysicalSize;
use winit::event::VirtualKeyCode;

mod graphics;
mod objects;
mod renderer;
mod window;

trait RenderBackend {
    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>);
    fn input(&mut self, event: &winit::event::WindowEvent) -> bool;
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

#[allow(clippy::upper_case_acronyms)]
#[derive(Clone)]
pub(crate) struct WGPU<'a> {
    pub(crate) queue: &'a wgpu::Queue,
    pub(crate) command_encoder: &'a RefCell<CommandEncoder>,
    pub(crate) view: &'a TextureView,
    pub(crate) render_pipelines: &'a [Option<RenderPipeline>],
    uniforms: &'a [Option<(&'static str, BindGroup, BindGroupLayout)>],
    current_uniforms: &'a [(usize, [usize; wgpu_core::MAX_BIND_GROUPS])],
    pub(crate) renderer_index: usize,
}

impl<'a> WGPU<'a> {
    pub(crate) fn current_render_pipeline(&self) -> &'a Option<RenderPipeline> {
        &self.render_pipelines[self.renderer_index]
    }
    pub(crate) fn current_uniforms(&self) -> impl Iterator<Item = &'a BindGroup> {
        let (size, uniform_indices) =
            unsafe { self.current_uniforms.get_unchecked(self.renderer_index) };
        (0..*size).map(|i| {
            &unsafe {
                self.uniforms
                    .get_unchecked(*uniform_indices.get_unchecked(i))
                    .as_ref()
                    .unwrap_unchecked()
            }
            .1
        })
    }
}

enum ShaderAction<'a> {
    CreatePipeline {
        src: &'static str,
        layout: &'static [wgpu::VertexBufferLayout<'static>],
        uniforms: &'static [&'static str],
    },
    AddUniform {
        name: &'static str,
        binding: u32,
        shader_stage: wgpu::ShaderStages,
        buffer: &'a wgpu::Buffer,
    },
}

trait Renderer<Data> {
    /// do never override this method
    fn visit<'a>(
        &'a self,
        data: &mut Data,
        actions: &mut [Option<ShaderAction<'a>>],
        index: usize,
    ) {
        let s = self.shader(data);
        if s.is_some() {
            *unsafe { actions.get_unchecked_mut(index) } = s;
        }
    }
    fn render(&mut self, _data: &mut Data, _wgpu: WGPU) {}
    fn render_ui(
        &mut self,
        _data: &mut Data,
        _ctx: &CtxRef,
        _menu_ui: &mut Ui,
        _actions: &mut UiActions,
    ) {
    }
    fn input(&mut self, _data: &mut Data, _event: &winit::event::WindowEvent) -> bool {
        false
    }
    fn shader(&self, _data: &mut Data) -> Option<ShaderAction> {
        None
    }
    fn resize(&mut self, _data: &mut Data, _size: PhysicalSize<u32>) {}
}

fn main() {
    env_logger::init();
    let model_result = load_model();
    let mut model = None;
    match model_result {
        Err(LoadError::NoModelToLoad) => log::error!("no model to load"),
        Err(LoadError::ObjError(err)) => log::error!("Error parsing files: {:?}", err),
        Err(LoadError::NativeError(err)) => log::error!("Error showing file dialog: {:?}", err),
        Err(LoadError::Other(err)) => log::error!("{}", err),
        Err(LoadError::IoError(err)) => log::error!("Error loading files: {}", err),
        Err(LoadError::FormatError(err)) => log::error!("Error converting obj file: {}", err),
        Ok(m) => model = Some(m),
    }
    let window = Window::new();
    let model_renderer = ModelRendererBuilder;
    let camera = CameraBuilder {
        eye: (0.0, 1.0, 2.0).into(),
        target: (0.0, 0.0, 0.0).into(),
        fovy: 45.0,
        znear: 0.1,
        zfar: 100.0,
    };
    let renderer = WGPURenderer::new(
        model,
        Some(VirtualKeyCode::R),
        &window,
        tuple_list!(model_renderer, camera),
    );
    window.run(renderer);
}
