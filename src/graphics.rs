use crate::{RenderBackend, Renderer, RendererBuilder, ShaderAction, UiActions, Window, WGPU};
use egui::epaint::ClippedShape;
use egui::{CtxRef, Ui};
use egui_wgpu_backend::{RenderPass, ScreenDescriptor};
use egui_winit_platform::Platform;
use std::cell::RefCell;
use std::mem::MaybeUninit;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;
use tuple_list::TupleList;
use wgpu::{BindGroup, BindGroupLayout, RenderPipeline};
use winit::dpi::PhysicalSize;
use winit::event::{ElementState, KeyboardInput, VirtualKeyCode, WindowEvent};

impl<Data> RendererBuilder<Data> for () {
    type Output = ();

    fn build(
        self,
        _data: &mut Data,
        _device: &wgpu::Device,
        _size: PhysicalSize<u32>,
    ) -> Self::Output {
    }
}

impl<Data> Renderer<Data> for () {}

impl<Data, Head, Tail> Renderer<Data> for (Head, Tail)
where
    Head: Renderer<Data>,
    Tail: Renderer<Data> + TupleList,
{
    fn visit<'a>(
        &'a self,
        data: &mut Data,
        actions: &mut [Option<ShaderAction<'a>>],
        index: usize,
    ) {
        self.0.visit(data, actions, index);
        self.1.visit(data, actions, index + 1);
    }

    fn render(&mut self, data: &mut Data, mut wgpu: WGPU) {
        self.0.render(data, wgpu.clone());
        wgpu.renderer_index += 1;
        self.1.render(data, wgpu);
    }

    fn render_ui(
        &mut self,
        data: &mut Data,
        ctx: &CtxRef,
        menu_ui: &mut Ui,
        actions: &mut UiActions,
    ) {
        self.0.render_ui(data, ctx, menu_ui, actions);
        self.1.render_ui(data, ctx, menu_ui, actions);
    }

    fn input(&mut self, data: &mut Data, event: &WindowEvent) -> bool {
        let l = self.0.input(data, event);
        let r = self.1.input(data, event);
        l || r
    }

    fn resize(&mut self, data: &mut Data, size: PhysicalSize<u32>) {
        self.0.resize(data, size);
        self.1.resize(data, size);
    }
}

impl<Data, Head, Tail> RendererBuilder<Data> for (Head, Tail)
where
    Head: RendererBuilder<Data>,
    Tail: RendererBuilder<Data> + TupleList,
    <Tail as RendererBuilder<Data>>::Output: TupleList,
{
    type Output = (Head::Output, Tail::Output);

    fn build(
        self,
        data: &mut Data,
        device: &wgpu::Device,
        size: PhysicalSize<u32>,
    ) -> Self::Output {
        (
            self.0.build(data, device, size),
            self.1.build(data, device, size),
        )
    }
}

pub(crate) struct WGPURenderer<Data: 'static, Renderers: RendererBuilder<Data> + TupleList>
where
    [(); <Renderers as TupleList>::TUPLE_LIST_SIZE]:,
{
    state: State,
    egui_rpass: RenderPass,
    previous_frame_time: Option<f32>,
    renderer_builders: Option<Renderers>,
    renderers: Option<Renderers::Output>,
    render_pipelines: [Option<RenderPipeline>; <Renderers as TupleList>::TUPLE_LIST_SIZE],
    #[allow(clippy::type_complexity)]
    bind_groups: [Option<(&'static str, BindGroup, BindGroupLayout)>;
        <Renderers as TupleList>::TUPLE_LIST_SIZE],
    #[allow(clippy::type_complexity)]
    bind_group_association:
        [(usize, [usize; wgpu_core::MAX_BIND_GROUPS]); <Renderers as TupleList>::TUPLE_LIST_SIZE],
    #[allow(clippy::type_complexity)]
    shader_creation_info: [Option<(
        &'static str,
        &'static [wgpu::VertexBufferLayout<'static>],
        &'static [&'static str],
        wgpu::PrimitiveTopology,
    )>; <Renderers as TupleList>::TUPLE_LIST_SIZE],
    reload_shader_key: Option<winit::event::VirtualKeyCode>,
    data: Data,
}

impl<Data: 'static, Renderers: RendererBuilder<Data> + TupleList> WGPURenderer<Data, Renderers>
where
    [(); <Renderers as TupleList>::TUPLE_LIST_SIZE]:,
{
    pub(crate) fn new(
        data: Data,
        reload_shader_key: Option<VirtualKeyCode>,
        window: &Window,
        renderers: Renderers,
    ) -> Self {
        let state = pollster::block_on(State::new(window));
        let egui_rpass = RenderPass::new(&state.device, state.config.format, 1);
        #[allow(clippy::type_complexity)]
        let mut rp: [MaybeUninit<Option<RenderPipeline>>;
            <Renderers as TupleList>::TUPLE_LIST_SIZE] = MaybeUninit::uninit_array();
        #[allow(clippy::type_complexity)]
        let mut bg: [MaybeUninit<Option<(&'static str, BindGroup, BindGroupLayout)>>;
            <Renderers as TupleList>::TUPLE_LIST_SIZE] = MaybeUninit::uninit_array();
        #[allow(clippy::type_complexity)]
        let mut shader_creation_info: [MaybeUninit<
            Option<(
                &'static str,
                &'static [wgpu::VertexBufferLayout<'static>],
                &'static [&'static str],
                wgpu::PrimitiveTopology,
            )>,
        >;
            <Renderers as TupleList>::TUPLE_LIST_SIZE] = MaybeUninit::uninit_array();
        for ((render_pipeline, bind_group), shader_creation_info) in rp
            .iter_mut()
            .zip(bg.iter_mut())
            .zip(shader_creation_info.iter_mut())
        {
            render_pipeline.write(None);
            bind_group.write(None);
            shader_creation_info.write(None);
        }

        Self {
            data,
            state,
            egui_rpass,
            previous_frame_time: None,
            renderer_builders: Some(renderers),
            renderers: None,
            render_pipelines: unsafe { MaybeUninit::array_assume_init(rp) },
            bind_groups: unsafe { MaybeUninit::array_assume_init(bg) },
            bind_group_association: [(0, [0; wgpu_core::MAX_BIND_GROUPS]);
                <Renderers as TupleList>::TUPLE_LIST_SIZE],
            reload_shader_key,
            shader_creation_info: unsafe { MaybeUninit::array_assume_init(shader_creation_info) },
        }
    }

    fn create_command_encoder(&self) -> wgpu::CommandEncoder {
        self.state
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            })
    }

    fn init_shader(&mut self) {
        // create render pipelines

        let mut render_pipelines: [MaybeUninit<Option<RenderPipeline>>;
            <Renderers as TupleList>::TUPLE_LIST_SIZE] = MaybeUninit::uninit_array();
        for rp in render_pipelines.iter_mut() {
            rp.write(None);
        }
        let mut pipeline_count = 0_usize;
        let mut render_pipelines = unsafe { MaybeUninit::array_assume_init(render_pipelines) };
        for (index, action) in self.shader_creation_info.iter().enumerate() {
            if let Some((src, layout, uniforms, topology)) = action {
                match std::fs::read_to_string(src) {
                    Ok(shader_data) => {
                        // arming shader compilation error handler.
                        SHADER_COMPILATION_ERROR.store(true, Ordering::Relaxed);
                        let shader =
                            self.state
                                .device
                                .create_shader_module(&wgpu::ShaderModuleDescriptor {
                                    label: Some(src),
                                    source: wgpu::ShaderSource::Wgsl(shader_data.into()),
                                });

                        // if false error occurred,
                        if !SHADER_COMPILATION_ERROR.load(Ordering::Relaxed) {
                            // disarm error handler.
                            SHADER_COMPILATION_ERROR.store(false, Ordering::Relaxed);
                            return;
                        }

                        let mut bind_group_layouts: [MaybeUninit<&BindGroupLayout>;
                            wgpu_core::MAX_BIND_GROUPS] = MaybeUninit::uninit_array();

                        let (uniform_size, uniform_indices) =
                            unsafe { self.bind_group_association.get_unchecked(index) };

                        for uniform_index in 0..(*uniform_size) {
                            unsafe { bind_group_layouts.get_unchecked_mut(uniform_index) }.write(
                                &unsafe {
                                    self.bind_groups
                                        .get_unchecked(
                                            *uniform_indices.get_unchecked(uniform_index),
                                        )
                                        .as_ref()
                                        .unwrap_unchecked()
                                }
                                .2,
                            );
                        }

                        let bind_group_layouts = unsafe {
                            MaybeUninit::slice_assume_init_ref(
                                &bind_group_layouts[0..uniforms.len()],
                            )
                        };

                        let render_pipeline_layout = self.state.device.create_pipeline_layout(
                            &wgpu::PipelineLayoutDescriptor {
                                label: Some(src),
                                bind_group_layouts,
                                push_constant_ranges: &[],
                            },
                        );
                        let render_pipeline = self.state.device.create_render_pipeline(
                            &wgpu::RenderPipelineDescriptor {
                                label: Some(src),
                                layout: Some(&render_pipeline_layout),
                                vertex: wgpu::VertexState {
                                    module: &shader,
                                    entry_point: "vs_main", // 1.
                                    buffers: layout,        // 2.
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
                                    topology: *topology, // 1.
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
                        pipeline_count += 1;
                        *unsafe { render_pipelines.get_unchecked_mut(index) } =
                            Some(render_pipeline);
                    }
                    Err(err) => {
                        log::error!("Error reading shader source code from `{}`: {}", src, err);
                    }
                }
            }
        }

        log::info!("created {} pipelines", pipeline_count);
        self.render_pipelines = render_pipelines;
    }
}

struct State {
    surface: wgpu::Surface,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
}

static SHADER_COMPILATION_ERROR: std::sync::atomic::AtomicBool = AtomicBool::new(false);

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

        device.on_uncaptured_error(|err| {
            let err: wgpu::Error = err;
            if SHADER_COMPILATION_ERROR.load(Ordering::Relaxed) {
                // armed error handler, check it validation error,
                // and don't panic just lock and disarm handler, to signal there was an error.
                if let wgpu::Error::ValidationError { description, .. } = &err {
                    log::warn!("{}", description);
                    SHADER_COMPILATION_ERROR.store(false, Ordering::Relaxed);
                    return;
                }
            }
            panic!("{}", err);
        });

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

impl<Data, Renderers: RendererBuilder<Data> + TupleList> RenderBackend
    for WGPURenderer<Data, Renderers>
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
        if let Some(renderers) = &mut self.renderers {
            renderers.resize(&mut self.data, new_size);
        }
    }

    fn input(&mut self, event: &WindowEvent) -> bool {
        let handled = self
            .renderers
            .as_mut()
            .expect("WGPURenderer::renderers was None in input")
            .input(&mut self.data, event);
        if !handled {
            if let Some(key) = &self.reload_shader_key {
                if let WindowEvent::KeyboardInput {
                    input:
                        KeyboardInput {
                            state: ElementState::Released,
                            virtual_keycode: Some(pressed_key),
                            ..
                        },
                    ..
                } = event
                {
                    if key == pressed_key {
                        self.init_shader();
                        return true;
                    }
                }
            }
        }
        handled
    }

    fn update(&mut self) {
        let renderers = self
            .renderers
            .as_mut()
            .expect("WGPURenderer::renderers was None in render");
        renderers.update(&mut self.data);
    }

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
                .build(&mut self.data, &self.state.device, self.state.size),
        );

        unsafe { self.renderers.as_ref().unwrap_unchecked() }.visit(
            &mut self.data,
            &mut actions,
            0,
        );

        // gather uniform bind groups
        for (index, action) in actions.iter().enumerate() {
            match action {
                Some(ShaderAction::AddUniform {
                    name,
                    binding,
                    shader_stage,
                    buffer,
                })
                | Some(ShaderAction::CreateShaderWithUniform {
                    name,
                    binding,
                    shader_stage,
                    buffer,
                    ..
                }) => {
                    let bind_group_layout = self.state.device.create_bind_group_layout(
                        &wgpu::BindGroupLayoutDescriptor {
                            entries: &[wgpu::BindGroupLayoutEntry {
                                binding: *binding,
                                visibility: *shader_stage,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Uniform,
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            }],
                            label: Some(*name),
                        },
                    );
                    let bind_group =
                        self.state
                            .device
                            .create_bind_group(&wgpu::BindGroupDescriptor {
                                layout: &bind_group_layout,
                                entries: &[wgpu::BindGroupEntry {
                                    binding: *binding,
                                    resource: buffer.as_entire_binding(),
                                }],
                                label: Some(*name),
                            });
                    *unsafe { self.bind_groups.get_unchecked_mut(index) } =
                        Some((*name, bind_group, bind_group_layout));
                }
                _ => {}
            }
        }

        // resolve uniform bind groups
        for (index, action) in actions.iter().enumerate() {
            match action {
                Some(ShaderAction::CreatePipeline { uniforms, .. })
                | Some(ShaderAction::CreateShaderWithUniform { uniforms, .. }) => {
                    if uniforms.len() > wgpu_core::MAX_BIND_GROUPS {
                        panic!(
                            "Cannot have more than {} bind groups per shader",
                            wgpu_core::MAX_BIND_GROUPS
                        );
                    }
                    let mut uniform_indices = [0_usize; wgpu_core::MAX_BIND_GROUPS];
                    let mut uniform_size = 0_usize;
                    for &uniform_name in *uniforms {
                        for (uniform_index, uniform) in self.bind_groups.iter().enumerate() {
                            if let Some((name, _, _)) = uniform {
                                if uniform_name == *name {
                                    *unsafe { uniform_indices.get_unchecked_mut(uniform_size) } =
                                        uniform_index;
                                    uniform_size += 1;
                                }
                            }
                        }
                    }
                    *unsafe { self.bind_group_association.get_unchecked_mut(index) } =
                        (uniforms.len(), uniform_indices);
                }
                _ => {}
            }
        }

        for (i, action) in actions.into_iter().enumerate() {
            if let Some(ShaderAction::CreatePipeline {
                src,
                layout,
                uniforms,
                topology,
            })
            | Some(ShaderAction::CreateShaderWithUniform {
                src,
                layout,
                uniforms,
                topology,
                ..
            }) = action
            {
                *unsafe { self.shader_creation_info.get_unchecked_mut(i) } =
                    Some((src, layout, uniforms, topology));
            }
        }
        self.init_shader();
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
                renderers.render_ui(&mut self.data, &ctx, ui, &mut actions);
            });
        });

        renderers.render(
            &mut self.data,
            WGPU {
                queue: &self.state.queue,
                command_encoder: &encoder,
                view: &view,
                render_pipelines: &self.render_pipelines,
                uniforms: &self.bind_groups,
                current_uniforms: &self.bind_group_association,
                renderer_index: 0,
            },
        );

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
