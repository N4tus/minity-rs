#![feature(maybe_uninit_uninit_array)]
#![feature(generic_const_exprs)]
#![feature(maybe_uninit_array_assume_init)]

use crate::graphics::WGPURenderer;
use crate::objects::{load_model, LoadError, Model};
use crate::renderer::ModelRendererBuilder;
use crate::window::Window;
use egui::paint::ClippedShape;
use egui::{CtxRef, Ui};
use egui_winit_platform::Platform;
use std::cell::RefCell;
use std::rc::Rc;
use tuple_list::tuple_list;
use wgpu::{CommandEncoder, RenderPipeline, TextureView};
use winit::dpi::PhysicalSize;

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

trait RendererBuilder {
    type Output: Renderer;
    fn build(self, device: &wgpu::Device) -> Self::Output;
}

#[derive(Clone)]
pub(crate) struct WGPU<'a> {
    pub(crate) command_encoder: &'a RefCell<CommandEncoder>,
    pub(crate) view: &'a TextureView,
    pub(crate) render_pipelines: &'a [Option<RenderPipeline>],
    pub(crate) renderer_index: usize,
}

impl<'a> WGPU<'a> {
    pub(crate) fn current_render_pipeline(&'a self) -> &'a Option<RenderPipeline> {
        &self.render_pipelines[self.renderer_index]
    }
}

trait Renderer {
    fn render(&mut self, wgpu: WGPU);
    fn render_ui(&mut self, ctx: &CtxRef, menu_ui: &mut Ui, actions: &mut UiActions);
    fn input(&mut self, event: &winit::event::WindowEvent) -> bool;
    fn shader(&self) -> Option<(&str, wgpu::VertexBufferLayout)>;
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
    let model_renderer = ModelRendererBuilder::new(Rc::new(RefCell::new(model)));
    let renderer = WGPURenderer::new(&window, tuple_list!(model_renderer));
    window.run(renderer);
}
