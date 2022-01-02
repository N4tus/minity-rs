use crate::array_vec::{default_array, optional_array};
use crate::graphics::bind_group::BindGroupStorage;
use crate::{
    ArrayVec, CreatePipeline, RenderBackend, Renderer, RendererBuilder, ShaderAction, UiActions,
    Window, MAX_BIND_GROUPS, WGPU,
};
use egui::epaint::ClippedShape;
use egui_wgpu_backend::{RenderPass, ScreenDescriptor};
use egui_winit_platform::Platform;
use std::cell::RefCell;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;
use tuple_list::TupleList;
use wgpu::RenderPipeline;
use winit::dpi::PhysicalSize;
use winit::event::{ElementState, KeyboardInput, VirtualKeyCode, WindowEvent};

mod bind_group;
mod renderer_impls;

pub(crate) struct WGPURenderer<Data: 'static, Renderers: RendererBuilder<Data> + TupleList>
where
    [(); <Renderers as TupleList>::TUPLE_LIST_SIZE]:,
{
    state: State,
    egui_rpass: RenderPass,
    depth_texture: (wgpu::Texture, wgpu::TextureView, wgpu::Sampler),
    previous_frame_time: Option<f32>,
    renderer_builders: Option<Renderers>,
    renderers: Option<Renderers::Output>,
    render_pipelines: [Option<RenderPipeline>; <Renderers as TupleList>::TUPLE_LIST_SIZE],
    render_pipeline_layouts:
        [Option<wgpu::PipelineLayout>; <Renderers as TupleList>::TUPLE_LIST_SIZE],
    bg_association: BindGroupStorage<{ <Renderers as TupleList>::TUPLE_LIST_SIZE }>,
    bind_group_ref_storage: bumpalo::Bump,
    #[allow(clippy::type_complexity)]
    shader_creation_info: [Option<(
        &'static str,
        &'static [wgpu::VertexBufferLayout<'static>],
        ArrayVec<&'static str, MAX_BIND_GROUPS>,
        wgpu::PrimitiveTopology,
    )>; <Renderers as TupleList>::TUPLE_LIST_SIZE],
    reload_shader_key: Option<VirtualKeyCode>,
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
        let depth_texture = Self::create_depth_texture(&state.device, &state.config);

        Self {
            data,
            state,
            depth_texture,
            egui_rpass,
            previous_frame_time: None,
            renderer_builders: Some(renderers),
            renderers: None,
            render_pipelines: optional_array(),
            render_pipeline_layouts: optional_array(),
            bg_association: BindGroupStorage::empty(),
            bind_group_ref_storage: bumpalo::Bump::new(),
            reload_shader_key,
            shader_creation_info: optional_array(),
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
        let mut pipeline_count = 0_usize;
        let mut render_pipelines: [Option<wgpu::RenderPipeline>;
            <Renderers as TupleList>::TUPLE_LIST_SIZE] = optional_array();
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
                        log::debug!("create pipeline `{}`", src);

                        let render_pipeline_layout = unsafe {
                            self.render_pipeline_layouts
                                .get_unchecked(index)
                                .as_ref()
                                .unwrap_unchecked()
                        };
                        let render_pipeline = self.state.device.create_render_pipeline(
                            &wgpu::RenderPipelineDescriptor {
                                label: Some(src),
                                layout: Some(render_pipeline_layout),
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
                                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                                        write_mask: wgpu::ColorWrites::ALL,
                                    }],
                                }),
                                primitive: wgpu::PrimitiveState {
                                    topology: *topology, // 1.
                                    strip_index_format: None,
                                    front_face: wgpu::FrontFace::Ccw, // 2.
                                    cull_mode: Some(wgpu::Face::Back),
                                    unclipped_depth: false,
                                    // Setting this to anything other than Fill requires Features::NON_FILL_POLYGON_MODE
                                    polygon_mode: wgpu::PolygonMode::Fill,
                                    // Requires Features::CONSERVATIVE_RASTERIZATION
                                    conservative: false,
                                },
                                depth_stencil: Some(wgpu::DepthStencilState {
                                    format: wgpu::TextureFormat::Depth32Float,
                                    depth_write_enabled: true,
                                    depth_compare: wgpu::CompareFunction::Less,
                                    stencil: Default::default(),
                                    bias: Default::default(),
                                }), // 1.
                                multisample: wgpu::MultisampleState {
                                    count: 1,                         // 2.
                                    mask: !0,                         // 3.
                                    alpha_to_coverage_enabled: false, // 4.
                                },
                                multiview: None,
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

    fn create_depth_texture(
        device: &wgpu::Device,
        config: &wgpu::SurfaceConfiguration,
    ) -> (wgpu::Texture, wgpu::TextureView, wgpu::Sampler) {
        let size = wgpu::Extent3d {
            // 2.
            width: config.width,
            height: config.height,
            depth_or_array_layers: 1,
        };
        let desc = wgpu::TextureDescriptor {
            label: Some("depth texture"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT // 3.
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST,
        };
        let texture = device.create_texture(&desc);

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            // 4.
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            compare: Some(wgpu::CompareFunction::LessEqual), // 5.
            lod_min_clamp: -100.0,
            lod_max_clamp: 100.0,
            ..Default::default()
        });
        (texture, view, sampler)
    }
}

struct State {
    surface: wgpu::Surface,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: PhysicalSize<u32>,
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
                    limits: Default::default(),
                    label: Some("device descriptor"),
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
                if let wgpu::Error::Validation { description, .. } = &err {
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
        self.depth_texture = Self::create_depth_texture(&self.state.device, &self.state.config);
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
                            virtual_keycode: Some(keycode),
                            ..
                        },
                    ..
                } = event
                {
                    if key == keycode {
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
        let mut actions: [ShaderAction; <Renderers as TupleList>::TUPLE_LIST_SIZE] =
            default_array();
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
            &self.state.device,
        );

        let bg_association = BindGroupStorage::taking_from_actions(&mut actions);

        for (i, action) in actions.into_iter().enumerate() {
            if let Some(CreatePipeline {
                src,
                layout,
                uniforms,
                topology,
            }) = action.create_pipeline
            {
                *unsafe { self.render_pipeline_layouts.get_unchecked_mut(i) } =
                    Some(self.state.device.create_pipeline_layout(
                        &wgpu::PipelineLayoutDescriptor {
                            label: Some(src),
                            bind_group_layouts: &*bg_association.bg_ref_list(i),
                            push_constant_ranges: &[],
                        },
                    ));

                *unsafe { self.shader_creation_info.get_unchecked_mut(i) } =
                    Some((src, layout, uniforms, topology));
            }
        }
        self.bg_association = bg_association;
        self.bind_group_ref_storage =
            bumpalo::Bump::with_capacity(self.bg_association.aligned_size_of_ref_bind_groups());
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

        {
            let mut bind_groups_ref = Vec::new_in(&self.bind_group_ref_storage);
            for bg in &self.bg_association.bind_groups {
                bind_groups_ref.push(bg);
            }

            renderers.render(
                &mut self.data,
                WGPU {
                    depth_view: &self.depth_texture.1,
                    queue: &self.state.queue,
                    command_encoder: &encoder,
                    view: &view,
                    render_pipelines: &self.render_pipelines,
                    bind_groups: bind_groups_ref.as_slice(),
                    bind_group_association: &self.bg_association.bind_group_association,
                    renderer_index: 0,
                },
            );
        }

        self.bind_group_ref_storage.reset();

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
