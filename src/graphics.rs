use crate::{RenderBackend, Renderer, RendererBuilder, ShaderAction, UiActions, Window, WGPU};
use egui::paint::ClippedShape;
use egui::{CtxRef, Ui};
use egui_wgpu_backend::{RenderPass, ScreenDescriptor};
use egui_winit_platform::Platform;
use std::cell::RefCell;
use std::mem::MaybeUninit;
use std::time::Instant;
use tuple_list::TupleList;
use wgpu::{BindGroup, BindGroupLayout, RenderPipeline};
use winit::dpi::PhysicalSize;
use winit::event::WindowEvent;

pub(crate) trait RendererBuilderImpl {
    type Output: RendererExt;

    fn build_impl(
        self,
        device: &wgpu::Device,
        size: PhysicalSize<u32>,
        index: usize,
    ) -> Self::Output;
}

pub(crate) trait RendererExt {
    fn visit<'a>(&'a self, actions: &mut [Option<ShaderAction<'a>>], index: usize);
    fn render(&mut self, _wgpu: WGPU) {}
    fn render_ui(&mut self, _ctx: &CtxRef, _menu_ui: &mut Ui, _actions: &mut UiActions) {}
    fn input(&mut self, _event: &winit::event::WindowEvent) -> bool {
        false
    }
    fn resize(&mut self, _size: PhysicalSize<u32>) {}
}

impl RendererBuilderImpl for () {
    type Output = ();

    fn build_impl(
        self,
        _device: &wgpu::Device,
        _size: PhysicalSize<u32>,
        _index: usize,
    ) -> Self::Output {
    }
}

impl RendererExt for () {
    fn visit<'a>(&'a self, _actions: &mut [Option<ShaderAction<'a>>], _index: usize) {}
}

impl<Head, Tail> RendererExt for (Head, Tail)
where
    Head: RendererExt,
    Tail: RendererExt + TupleList,
{
    fn visit<'a>(&'a self, actions: &mut [Option<ShaderAction<'a>>], index: usize) {
        self.0.visit(actions, index);
        self.1.visit(actions, index + 1);
    }

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

    fn resize(&mut self, size: PhysicalSize<u32>) {
        self.0.resize(size);
        self.1.resize(size);
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
        device: &wgpu::Device,
        size: PhysicalSize<u32>,
        index: usize,
    ) -> Self::Output {
        let head_output = self.0.build_impl(device, size, index);
        (head_output, self.1.build_impl(device, size, index + 1))
    }
}

impl<R: Renderer> RendererExt for R {
    fn visit<'a>(&'a self, actions: &mut [Option<ShaderAction<'a>>], index: usize) {
        let s = self.shader();
        if s.is_some() {
            *unsafe { actions.get_unchecked_mut(index) } = s;
        }
    }

    fn render(&mut self, wgpu: WGPU) {
        Renderer::render(self, wgpu);
    }

    fn render_ui(&mut self, ctx: &CtxRef, menu_ui: &mut Ui, actions: &mut UiActions) {
        Renderer::render_ui(self, ctx, menu_ui, actions);
    }

    fn input(&mut self, event: &WindowEvent) -> bool {
        Renderer::input(self, event)
    }

    fn resize(&mut self, size: PhysicalSize<u32>) {
        Renderer::resize(self, size);
    }
}

impl<R: RendererBuilder> RendererBuilderImpl for R {
    type Output = <R as RendererBuilder>::Output;

    fn build_impl(
        self,
        device: &wgpu::Device,
        size: PhysicalSize<u32>,
        _index: usize,
    ) -> Self::Output {
        self.build(device, size)
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
    render_pipelines: [Option<RenderPipeline>; <Renderers as TupleList>::TUPLE_LIST_SIZE],
    /// all bind groups of one RenderPipeline are next to each other.
    /// starting from bind_group_association.0 with bind_group_association.1 count elements.
    bind_groups: [MaybeUninit<BindGroup>; <Renderers as TupleList>::TUPLE_LIST_SIZE],
    bind_group_association: [(usize, usize); <Renderers as TupleList>::TUPLE_LIST_SIZE],
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
        let bg: [MaybeUninit<BindGroup>; <Renderers as TupleList>::TUPLE_LIST_SIZE] =
            MaybeUninit::uninit_array();
        Self {
            state,
            egui_rpass,
            previous_frame_time: None,
            renderer_builders: Some(renderers),
            renderers: None,
            render_pipelines: unsafe { MaybeUninit::array_assume_init(rp) },
            bind_groups: bg,
            bind_group_association: [(0, 0); <Renderers as TupleList>::TUPLE_LIST_SIZE],
        }
    }

    fn create_command_encoder(&self) -> wgpu::CommandEncoder {
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
        let mut actions: [MaybeUninit<Option<ShaderAction>>;
            <Renderers as TupleList>::TUPLE_LIST_SIZE] = MaybeUninit::uninit_array();
        for action in actions.iter_mut() {
            action.write(None);
        }
        let mut actions = unsafe { MaybeUninit::array_assume_init(actions) };
        self.renderers = Some(
            self.renderer_builders
                .take()
                .expect("WGPURenderer::renderer_builder was None on init")
                .build_impl(&self.state.device, self.state.size, 0),
        );

        unsafe { self.renderers.as_ref().unwrap_unchecked() }.visit(&mut actions, 0);

        let mut bind_group_index = 0_usize;
        for action_index in 0..actions.len() {
            let is_create_pipeline = matches!(
                &unsafe { actions.get_unchecked(action_index) },
                Some(ShaderAction::CreatePipeline(..))
            );
            // ripping out the action
            if is_create_pipeline {
                let action = unsafe { actions[action_index].take().unwrap_unchecked() };
                if let ShaderAction::CreatePipeline(name, desc) = action {
                    let mut bindings: [MaybeUninit<BindGroupLayout>;
                        <Renderers as TupleList>::TUPLE_LIST_SIZE] = MaybeUninit::uninit_array();
                    let mut len = 0_usize;
                    let bind_group_start_index = bind_group_index;
                    for action in actions.iter().flatten() {
                        if let ShaderAction::AddUniform(n, binding, stage, buffer) = action {
                            if *n == name {
                                let bind_group_layout = self.state.device.create_bind_group_layout(
                                    &wgpu::BindGroupLayoutDescriptor {
                                        entries: &[wgpu::BindGroupLayoutEntry {
                                            binding: *binding,
                                            visibility: *stage,
                                            ty: wgpu::BindingType::Buffer {
                                                ty: wgpu::BufferBindingType::Uniform,
                                                has_dynamic_offset: false,
                                                min_binding_size: None,
                                            },
                                            count: None,
                                        }],
                                        label: Some(name),
                                    },
                                );
                                let bind_group = self.state.device.create_bind_group(
                                    &wgpu::BindGroupDescriptor {
                                        layout: &bind_group_layout,
                                        entries: &[wgpu::BindGroupEntry {
                                            binding: *binding,
                                            resource: buffer.as_entire_binding(),
                                        }],
                                        label: Some("camera_bind_group"),
                                    },
                                );

                                let _ =
                                    unsafe { self.bind_groups.get_unchecked_mut(bind_group_index) }
                                        .write(bind_group);
                                bind_group_index += 1;

                                let _ = *unsafe { bindings.get_unchecked_mut(len) }
                                    .write(bind_group_layout);
                                len += 1;
                            }
                        }
                    }
                    let bindings = unsafe { MaybeUninit::slice_assume_init_ref(&bindings[0..len]) };

                    let mut bindings_ref: [MaybeUninit<&BindGroupLayout>;
                        <Renderers as TupleList>::TUPLE_LIST_SIZE] = MaybeUninit::uninit_array();

                    for (idx, binding) in bindings.iter().enumerate() {
                        unsafe { bindings_ref.get_unchecked_mut(idx) }.write(binding);
                    }

                    let bindings_ref = unsafe {
                        MaybeUninit::slice_assume_init_ref(&bindings_ref[0..bindings.len()])
                    };

                    match std::fs::read_to_string(name) {
                        Ok(shader_data) => {
                            let shader = self.state.device.create_shader_module(
                                &wgpu::ShaderModuleDescriptor {
                                    label: Some(name),
                                    source: wgpu::ShaderSource::Wgsl(shader_data.into()),
                                },
                            );
                            let render_pipeline_layout = self.state.device.create_pipeline_layout(
                                &wgpu::PipelineLayoutDescriptor {
                                    label: Some(name),
                                    bind_group_layouts: bindings_ref,
                                    push_constant_ranges: &[],
                                },
                            );
                            let render_pipeline = self.state.device.create_render_pipeline(
                                &wgpu::RenderPipelineDescriptor {
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
                                            format: self.state.config.format,
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
                                },
                            );
                            *unsafe { self.render_pipelines.get_unchecked_mut(action_index) } =
                                Some(render_pipeline);
                            *unsafe {
                                self.bind_group_association.get_unchecked_mut(action_index)
                            } = (
                                bind_group_start_index,
                                bind_group_index - bind_group_start_index,
                            );
                        }
                        Err(err) => {
                            log::error!("Error reading shader source code: {:?}", err);
                        }
                    }
                }
            }
        }
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
        egui::TopBottomPanel::top("menu").show(&ctx, |ui| {
            egui::menu::bar(ui, |ui| {
                renderers.render_ui(&ctx, ui, &mut actions);
            });
        });

        renderers.render(WGPU {
            queue: &self.state.queue,
            command_encoder: &encoder,
            view: &view,
            render_pipelines: &self.render_pipelines,
            uniforms: &self.bind_groups,
            current_uniforms: &self.bind_group_association,
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
