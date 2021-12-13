use crate::{RenderBackend, Renderer, RendererBuilder, UiActions, Window, WGPU};
use egui::paint::ClippedShape;
use egui::{Color32, CtxRef, Frame, Style, Ui};
use egui_wgpu_backend::wgpu::{CommandEncoder, Device};
use egui_wgpu_backend::{RenderPass, ScreenDescriptor};
use egui_winit_platform::Platform;
use std::cell::RefCell;
use std::mem::MaybeUninit;
use std::time::Instant;
use tuple_list::TupleList;
use wgpu::RenderPipeline;
use winit::dpi::PhysicalSize;
use winit::event::WindowEvent;

impl Renderer for () {
    fn render(&mut self, _wgpu: WGPU) {}

    fn render_ui(&mut self, _ctx: &CtxRef, _menu_ui: &mut Ui, _actions: &mut UiActions) {}

    fn input(&mut self, _event: &WindowEvent) -> bool {
        false
    }

    fn shader(&self) -> Option<(&str, wgpu::VertexBufferLayout)> {
        None
    }
}

pub(crate) trait RendererBuilderImpl {
    type Output: Renderer;

    fn build_impl(
        self,
        device: &wgpu::Device,
        pipelines: &mut [Option<RenderPipeline>],
        format: wgpu::TextureFormat,
        index: usize,
    ) -> Self::Output;
}

impl RendererBuilderImpl for () {
    type Output = ();

    fn build_impl(
        self,
        _device: &Device,
        _pipelines: &mut [Option<RenderPipeline>],
        _format: wgpu::TextureFormat,
        _index: usize,
    ) -> Self::Output {
    }
}

impl<Head, Tail> Renderer for (Head, Tail)
where
    Head: Renderer,
    Tail: Renderer + TupleList,
{
    fn render(&mut self, mut wgpu: WGPU) {
        self.0.render(wgpu.clone());
        wgpu.renderer_index += 1;
        self.1.render(wgpu);
    }

    fn render_ui(&mut self, ctx: &CtxRef, menu_ui: &mut Ui, actions: &mut UiActions) {
        self.0.render_ui(ctx, menu_ui, actions);
        self.1.render_ui(ctx, menu_ui, actions);
    }

    fn input(&mut self, event: &WindowEvent) -> bool {
        let l = self.0.input(event);
        let r = self.1.input(event);
        l || r
    }

    fn shader(&self) -> Option<(&str, wgpu::VertexBufferLayout)> {
        None
    }
}

impl<Head, Tail> RendererBuilderImpl for (Head, Tail)
where
    Head: RendererBuilderImpl,
    Tail: RendererBuilderImpl + TupleList,
    <Tail as RendererBuilderImpl>::Output: TupleList,
{
    type Output = (Head::Output, Tail::Output);

    fn build_impl(
        self,
        device: &Device,
        pipelines: &mut [Option<RenderPipeline>],
        format: wgpu::TextureFormat,
        index: usize,
    ) -> Self::Output {
        let head_output = self.0.build_impl(device, pipelines, format, index);
        if let Some((name, desc)) = head_output.shader() {
            match std::fs::read_to_string(name) {
                Ok(shader_data) => {
                    let shader = device.create_shader_module(&wgpu::ShaderModuleDescriptor {
                        label: Some(name),
                        source: wgpu::ShaderSource::Wgsl(shader_data.into()),
                    });
                    let render_pipeline_layout =
                        device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                            label: Some(name),
                            bind_group_layouts: &[],
                            push_constant_ranges: &[],
                        });
                    let render_pipeline =
                        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                            label: Some(name),
                            layout: Some(&render_pipeline_layout),
                            vertex: wgpu::VertexState {
                                module: &shader,
                                entry_point: "vs_main", // 1.
                                buffers: &[desc],       // 2.
                            },
                            fragment: Some(wgpu::FragmentState {
                                // 3.
                                module: &shader,
                                entry_point: "fs_main",
                                targets: &[wgpu::ColorTargetState {
                                    // 4.
                                    format,
                                    blend: Some(wgpu::BlendState::REPLACE),
                                    write_mask: wgpu::ColorWrites::ALL,
                                }],
                            }),
                            primitive: wgpu::PrimitiveState {
                                topology: wgpu::PrimitiveTopology::TriangleList, // 1.
                                strip_index_format: None,
                                front_face: wgpu::FrontFace::Ccw, // 2.
                                cull_mode: Some(wgpu::Face::Back),
                                // Setting this to anything other than Fill requires Features::NON_FILL_POLYGON_MODE
                                polygon_mode: wgpu::PolygonMode::Fill,
                                // Requires Features::DEPTH_CLAMPING
                                clamp_depth: false,
                                // Requires Features::CONSERVATIVE_RASTERIZATION
                                conservative: false,
                            },
                            depth_stencil: None, // 1.
                            multisample: wgpu::MultisampleState {
                                count: 1,                         // 2.
                                mask: !0,                         // 3.
                                alpha_to_coverage_enabled: false, // 4.
                            },
                        });
                    *unsafe { pipelines.get_unchecked_mut(index) } = Some(render_pipeline);
                }
                Err(err) => {
                    log::error!("Error reading shader source code: {:?}", err);
                }
            }
        }
        (
            head_output,
            self.1.build_impl(device, pipelines, format, index + 1),
        )
    }
}

impl<R: RendererBuilder> RendererBuilderImpl for R {
    type Output = <R as RendererBuilder>::Output;

    fn build_impl(
        self,
        device: &Device,
        _pipelines: &mut [Option<RenderPipeline>],
        _format: wgpu::TextureFormat,
        _index: usize,
    ) -> Self::Output {
        self.build(device)
    }
}

pub(crate) struct WGPURenderer<Renderers: RendererBuilderImpl + TupleList>
where
    [(); <Renderers as TupleList>::TUPLE_LIST_SIZE]:,
{
    state: State,
    egui_rpass: RenderPass,
    previous_frame_time: Option<f32>,
    renderer_builders: Option<Renderers>,
    renderers: Option<Renderers::Output>,
    render_pipelines: [Option<wgpu::RenderPipeline>; <Renderers as TupleList>::TUPLE_LIST_SIZE],
}

impl<Renderers: RendererBuilderImpl + TupleList> WGPURenderer<Renderers>
where
    [(); <Renderers as TupleList>::TUPLE_LIST_SIZE]:,
{
    pub(crate) fn new(window: &Window, renderers: Renderers) -> Self {
        let state = pollster::block_on(State::new(window));
        let egui_rpass = RenderPass::new(&state.device, state.config.format, 1);
        let mut rp: [MaybeUninit<Option<RenderPipeline>>;
            <Renderers as TupleList>::TUPLE_LIST_SIZE] = MaybeUninit::uninit_array();
        for render_pipeline in rp.iter_mut() {
            render_pipeline.write(None);
        }
        Self {
            state,
            egui_rpass,
            previous_frame_time: None,
            renderer_builders: Some(renderers),
            renderers: None,
            render_pipelines: unsafe { MaybeUninit::array_assume_init(rp) },
        }
    }

    fn create_command_encoder(&self) -> CommandEncoder {
        self.state
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            })
    }
}

struct State {
    surface: wgpu::Surface,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
}

impl State {
    // Creating some of the wgpu types requires async code
    async fn new(window: &Window) -> Self {
        let window = &window.window;
        let size = window.inner_size();

        // The instance is a handle to our GPU
        // Backends::all => Vulkan + Metal + DX12 + Browser WebGPU
        let instance = wgpu::Instance::new(wgpu::Backends::all());
        let surface = unsafe { instance.create_surface(window) };
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    features: wgpu::Features::empty(),
                    limits: wgpu::Limits::default(),
                    label: None,
                },
                None, // Trace path
            )
            .await
            .unwrap();

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface.get_preferred_format(&adapter).unwrap(),
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
        };
        surface.configure(&device, &config);
        Self {
            surface,
            device,
            queue,
            config,
            size,
        }
    }
}

impl<Renderers: RendererBuilderImpl + TupleList> RenderBackend for WGPURenderer<Renderers>
where
    [(); <Renderers as TupleList>::TUPLE_LIST_SIZE]:,
{
    fn resize(&mut self, mut new_size: PhysicalSize<u32>) {
        if new_size.width == 0 || new_size.height == 0 {
            new_size = self.state.size;
        }
        self.state.size = new_size;
        self.state.config.width = new_size.width;
        self.state.config.height = new_size.height;
        self.state
            .surface
            .configure(&self.state.device, &self.state.config);
    }

    fn input(&mut self, event: &WindowEvent) -> bool {
        self.renderers
            .as_mut()
            .expect("WGPURenderer::renderers was None in input")
            .input(event)
    }

    fn update(&mut self) {}

    fn init(&mut self) {
        self.renderers = Some(
            self.renderer_builders
                .take()
                .expect("WGPURenderer::renderer_builder was None on init")
                .build_impl(
                    &self.state.device,
                    &mut self.render_pipelines,
                    self.state.config.format,
                    0,
                ),
        );
    }

    fn render(
        &mut self,
        scale_factor: f32,
        platform: &mut Platform,
        end_frame: &dyn Fn(&mut Platform) -> Vec<ClippedShape>,
    ) -> Result<UiActions, wgpu::SurfaceError> {
        let output = self.state.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let physical_size = self.state.size;
        let egui_start = Instant::now();
        platform.begin_frame();

        let mut actions = UiActions::default();
        let ctx = platform.context();
        let encoder = self.create_command_encoder();
        let encoder = RefCell::new(encoder);
        let renderers = self
            .renderers
            .as_mut()
            .expect("WGPURenderer::renderers was None in render");
        let mut style = Style::default();
        // style.visuals.widgets.noninteractive.bg_fill = Color32::from_white_alpha(255);
        egui::TopBottomPanel::top("menu")
            .frame(Frame::menu(&style))
            .max_height(200.0)
            .height_range(0.0..=100.0)
            .show(&ctx, |ui| {
                egui::menu::bar(ui, |ui| {
                    renderers.render_ui(&ctx, ui, &mut actions);
                });
            });

        renderers.render(WGPU {
            command_encoder: &encoder,
            view: &view,
            render_pipelines: &self.render_pipelines,
            renderer_index: 0,
        });

        let paint_commands = end_frame(platform);
        let paint_jobs = platform.context().tessellate(paint_commands);

        let frame_time = (Instant::now() - egui_start).as_secs_f64() as f32;
        self.previous_frame_time = Some(frame_time);

        // Upload all resources for the GPU.
        let screen_descriptor = ScreenDescriptor {
            physical_width: physical_size.width,
            physical_height: physical_size.height,
            scale_factor,
        };
        self.egui_rpass.update_texture(
            &self.state.device,
            &self.state.queue,
            &platform.context().texture(),
        );
        self.egui_rpass
            .update_user_textures(&self.state.device, &self.state.queue);
        self.egui_rpass.update_buffers(
            &self.state.device,
            &self.state.queue,
            &paint_jobs,
            &screen_descriptor,
        );

        let mut ui_encoder = self.create_command_encoder();

        self.egui_rpass
            .execute(
                &mut ui_encoder,
                &view,
                &paint_jobs,
                &screen_descriptor,
                None,
            )
            .unwrap();

        // submit will accept anything that implements IntoIter
        self.state
            .queue
            .submit([encoder.into_inner().finish(), ui_encoder.finish()]);
        output.present();

        Ok(actions)
    }

    fn size(&self) -> PhysicalSize<u32> {
        self.state.size
    }
}
