use crate::objects::{MaterialData, MaterialDataPadding, Vertex};
use crate::{
    array_vec, App, CreateBindGroup, CreatePipeline, Dirty, Renderer, RendererBuilder,
    ShaderAction, UiActions, WGPU,
};
use cgmath::{InnerSpace, Rotation, Rotation3, SquareMatrix};
use egui::{CtxRef, Ui, Widget};
use egui_wgpu_backend::wgpu::Device;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{BufferDescriptor, BufferUsages, Color, DynamicOffset, VertexBufferLayout};
use winit::dpi::PhysicalSize;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};

mod shader;

pub(crate) struct ModelRenderer {
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    material_buffer: wgpu::Buffer,
}

pub(crate) struct ModelRendererBuilder;

impl RendererBuilder<App> for ModelRendererBuilder {
    type Output = ModelRenderer;

    fn build(
        self,
        data: &mut App,
        device: &wgpu::Device,
        _size: PhysicalSize<u32>,
    ) -> Self::Output {
        let (vertex_buffer, index_buffer, material_buffer) = {
            let (data, indices, mat_count) = if let Some(m) = &data.model {
                (
                    bytemuck::cast_slice(m.vertices.as_slice()),
                    bytemuck::cast_slice(m.indices.as_slice()),
                    m.materials.shader_data.len(),
                )
            } else {
                ([0u8; 0].as_slice(), [0u8; 0].as_slice(), 1)
            };
            (
                device.create_buffer_init(&BufferInitDescriptor {
                    label: Some("model vertex buffer"),
                    contents: data,
                    usage: BufferUsages::VERTEX,
                }),
                device.create_buffer_init(&BufferInitDescriptor {
                    label: Some("model index buffer"),
                    contents: indices,
                    usage: BufferUsages::INDEX,
                }),
                device.create_buffer(&BufferDescriptor {
                    label: Some("material uniform Buffer"),
                    usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
                    size: (mat_count * std::mem::size_of::<MaterialDataPadding>())
                        as wgpu::BufferAddress,
                    mapped_at_creation: false,
                }),
            )
        };
        data.dirty.insert(Dirty::MATERIAL);
        Self::Output {
            vertex_buffer,
            index_buffer,
            material_buffer,
        }
    }
}

impl Renderer<App> for ModelRenderer {
    fn render(&mut self, data: &mut App, wgpu: WGPU) {
        if let Some(model) = &data.model {
            if data.dirty.contains(Dirty::MATERIAL) {
                data.dirty.remove(Dirty::MATERIAL);
                let d = model.materials.shader_data.as_byte_slice();
                log::debug!("write {}bytes to material uniform", d.len());
                wgpu.queue.write_buffer(&self.material_buffer, 0, d);
            }
            if let Some(pipeline) = wgpu.current_render_pipeline().as_ref() {
                let mut encoder = wgpu.command_encoder.borrow_mut();
                let mut uniforms = wgpu.current_uniforms();
                let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Model Render Pass"),
                    color_attachments: &[
                        // This is what [[location(0)]] in the fragment shader targets
                        wgpu::RenderPassColorAttachment {
                            view: wgpu.view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(Color {
                                    r: data.bg[0] as f64,
                                    g: data.bg[1] as f64,
                                    b: data.bg[2] as f64,
                                    a: 1.0,
                                }),
                                store: true,
                            },
                        },
                    ],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: wgpu.depth_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(1.0),
                            store: true,
                        }),
                        stencil_ops: None,
                    }),
                });

                render_pass.set_pipeline(pipeline); // 2.
                render_pass.set_bind_group(0, uniforms.next().unwrap()[0], &[]);
                render_pass
                    .set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);

                render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));

                let material_uniform = uniforms.next().unwrap()[0];
                for (index, group) in model.groups.iter().enumerate() {
                    let offset = index * std::mem::size_of::<MaterialDataPadding>();
                    render_pass.set_bind_group(1, material_uniform, &[offset as DynamicOffset]);
                    render_pass.draw_indexed(group.index_range(), 0, 0..1);
                }
            }
        }
    }

    fn render_ui(
        &mut self,
        data: &mut App,
        _ctx: &CtxRef,
        menu_ui: &mut Ui,
        actions: &mut UiActions,
    ) {
        egui::menu::menu(menu_ui, "Config", |ui| {
            if ui.button("Quit").clicked() {
                actions.quit = true;
            }
            ui.color_edit_button_rgb(&mut data.bg);
        });
        egui::menu::menu(menu_ui, "Model", |ui| {
            egui::Grid::new("model_menu_grid")
                .num_columns(2)
                .spacing([40.0, 4.0])
                .striped(true)
                .show(ui, |_ui| {})
        });
    }

    fn shader(&self, _data: &mut App, device: &wgpu::Device) -> ShaderAction {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("material bind group layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("material bind group"),
            layout: &layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &self.material_buffer,
                    offset: 0,
                    size: wgpu::BufferSize::new(std::mem::size_of::<MaterialData>() as u64),
                }),
            }],
        });
        ShaderAction {
            create_bind_groups: vec![CreateBindGroup {
                name: "material",
                groups: vec![group],
                layout,
            }],
            create_pipeline: Some(CreatePipeline {
                src: "shader_src/model.wgsl",
                layout: VERTEX_LAYOUT,
                uniforms: array_vec!["camera", "material"],
                topology: wgpu::PrimitiveTopology::TriangleList,
            }),
        }
    }
}

const VERTEX_LAYOUT: &[VertexBufferLayout<'static>] = &[Vertex::desc()];

pub(crate) struct CameraBuilder {
    pub(crate) eye: cgmath::Point3<f32>,
    pub(crate) target: cgmath::Point3<f32>,
    pub(crate) fovy: f32,
    pub(crate) znear: f32,
    pub(crate) zfar: f32,
}

pub(crate) struct Camera {
    eye: cgmath::Point3<f32>,
    target: cgmath::Point3<f32>,
    up: cgmath::Vector3<f32>,
    fovy: f32,
    znear: f32,
    zfar: f32,
    aspect: f32,

    uniform: wgpu::Buffer,

    left_mouse_pressed: bool,
    shift_pressed: bool,
    mouse_pos: cgmath::Point2<f32>,
    x_axis: cgmath::Vector3<f32>,
    y_axis: cgmath::Vector3<f32>,
}

#[rustfmt::skip]
pub const OPENGL_TO_WGPU_MATRIX: cgmath::Matrix4<f32> = cgmath::Matrix4::new(
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.5, 0.0,
    0.0, 0.0, 0.5, 1.0,
);

impl RendererBuilder<App> for CameraBuilder {
    type Output = Camera;

    fn build(self, data: &mut App, device: &wgpu::Device, size: PhysicalSize<u32>) -> Self::Output {
        let aspect = size.width as f32 / size.height as f32;
        let view = cgmath::Matrix4::look_at_rh(self.eye, self.target, cgmath::Vector3::unit_y());
        let proj = cgmath::perspective(cgmath::Deg(self.fovy), aspect, self.znear, self.zfar);

        data.view_proj = proj * view;

        let uniform = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Camera uniform buffer"),
            contents: bytemuck::cast_slice::<[[f32; 4]; 4], u8>(&[(OPENGL_TO_WGPU_MATRIX
                * data.view_proj)
                .into()]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        Self::Output {
            eye: self.eye,
            target: self.target,
            up: cgmath::Vector3::unit_y(),
            fovy: self.fovy,
            znear: self.znear,
            zfar: self.zfar,
            aspect,
            uniform,
            left_mouse_pressed: false,
            shift_pressed: false,
            mouse_pos: cgmath::Point2::new(0.0, 0.0),
            x_axis: cgmath::Vector3::unit_x(),
            y_axis: cgmath::Vector3::unit_y(),
        }
    }
}

impl Camera {
    fn update_camera(&mut self, data: &mut App) {
        let view = cgmath::Matrix4::look_at_rh(self.eye, self.target, self.up);
        let proj = cgmath::perspective(cgmath::Deg(self.fovy), self.aspect, self.znear, self.zfar);

        data.view_proj = proj * view;

        data.dirty.insert(Dirty::CAMERA);
    }
}

impl Renderer<App> for Camera {
    fn render(&mut self, data: &mut App, wgpu: WGPU) {
        if data.dirty.contains(Dirty::CAMERA) {
            data.dirty.remove(Dirty::CAMERA);
            wgpu.queue.write_buffer(
                &self.uniform,
                0,
                bytemuck::cast_slice::<[[f32; 4]; 4], u8>(&[(OPENGL_TO_WGPU_MATRIX
                    * data.view_proj)
                    .into()]),
            );
        }
    }

    fn input(&mut self, data: &mut App, event: &WindowEvent) -> bool {
        match event {
            WindowEvent::MouseInput {
                state,
                button: MouseButton::Left,
                ..
            } => {
                self.left_mouse_pressed = *state == ElementState::Pressed;
                false
            }
            WindowEvent::CursorMoved { position, .. } => {
                if self.left_mouse_pressed {
                    let dir =
                        self.mouse_pos - cgmath::Point2::new(position.x as f32, position.y as f32);

                    let rot_axis = self.x_axis * dir.y + self.y_axis * dir.x;

                    let rot_angle = dir.magnitude() / 2.0;
                    let qrot = cgmath::Quaternion::from_axis_angle(
                        rot_axis.normalize(),
                        cgmath::Deg(rot_angle),
                    );

                    if self.shift_pressed {
                        data.light_data = qrot.conjugate().rotate_point(data.light_data);
                        data.dirty.insert(Dirty::LIGHT);
                    } else {
                        self.eye = qrot.rotate_point(self.eye);
                        self.up = qrot.rotate_vector(self.up);
                        self.x_axis = qrot.rotate_vector(self.x_axis);
                        self.y_axis = qrot.rotate_vector(self.y_axis);
                        self.update_camera(data);
                    }

                    self.mouse_pos = cgmath::Point2::new(position.x as f32, position.y as f32);
                    true
                } else {
                    self.mouse_pos = cgmath::Point2::new(position.x as f32, position.y as f32);
                    false
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let delta = match delta {
                    MouseScrollDelta::LineDelta(_, d) => *d,
                    MouseScrollDelta::PixelDelta(px) => px.y as f32,
                };
                if self.shift_pressed {
                    if delta >= 0.0 {
                        data.light_data /= delta + 1.0;
                    } else {
                        data.light_data *= -delta + 1.0;
                    }
                    data.dirty.insert(Dirty::LIGHT);
                } else {
                    if delta >= 0.0 {
                        self.eye /= delta + 1.0;
                    } else {
                        self.eye *= -delta + 1.0;
                    }
                    self.update_camera(data);
                }
                true
            }
            WindowEvent::ModifiersChanged(state) => {
                self.shift_pressed = state.shift();
                false
            }
            _ => false,
        }
    }

    fn shader(&self, _data: &mut App, device: &wgpu::Device) -> ShaderAction {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("camera bind group layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("camera bind group"),
            layout: &layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: self.uniform.as_entire_binding(),
            }],
        });
        ShaderAction {
            create_bind_groups: vec![CreateBindGroup {
                name: "camera",
                groups: vec![group],
                layout,
            }],
            create_pipeline: None,
        }
    }

    fn resize(&mut self, data: &mut App, size: PhysicalSize<u32>) {
        self.aspect = size.width as f32 / size.height as f32;
        self.update_camera(data);
    }
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable, Default)]
struct LightUniform {
    view_proj: [[f32; 4]; 4],
    position: [f32; 3],
    _0: f32,
    viewport_size: [f32; 2],
    size: f32,
    _1: f32,
}

pub(crate) struct LightRendererBuilder;
pub(crate) struct LightRenderer {
    uniform: wgpu::Buffer,
    uniform_data: LightUniform,
}

impl RendererBuilder<App> for LightRendererBuilder {
    type Output = LightRenderer;

    fn build(self, data: &mut App, device: &Device, size: PhysicalSize<u32>) -> Self::Output {
        // The LightRenderer is built after the CameraRenderer, so data.proj has a proper value
        assert!(!data.view_proj.is_identity());
        let uniform_data = LightUniform {
            view_proj: (OPENGL_TO_WGPU_MATRIX * data.view_proj).into(),
            position: [data.light_data.x, data.light_data.y, data.light_data.z],
            _0: 0.0,
            viewport_size: [size.width as f32, size.height as f32],
            size: 27.0,
            _1: 0.0,
        };
        let uniform = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Light uniform buffer"),
            contents: bytemuck::cast_slice(&[uniform_data]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        LightRenderer {
            uniform,
            uniform_data,
        }
    }
}

impl Renderer<App> for LightRenderer {
    fn update(&mut self, data: &mut App) {
        if data.dirty.contains(Dirty::LIGHT) {
            self.uniform_data.position = [data.light_data.x, data.light_data.y, data.light_data.z];
        }
        if data.dirty.contains(Dirty::CAMERA) {
            data.dirty.insert(Dirty::LIGHT);
            self.uniform_data.view_proj = (OPENGL_TO_WGPU_MATRIX * data.view_proj).into();
        }
    }

    fn render(&mut self, data: &mut App, wgpu: WGPU) {
        if data.dirty.contains(Dirty::LIGHT) {
            data.dirty.remove(Dirty::LIGHT);
            wgpu.queue
                .write_buffer(&self.uniform, 0, bytemuck::cast_slice(&[self.uniform_data]));
        }

        if let Some(pipeline) = wgpu.current_render_pipeline().as_ref() {
            let mut encoder = wgpu.command_encoder.borrow_mut();
            let mut uniforms = wgpu.current_uniforms();
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Light Render Pass"),
                color_attachments: &[
                    // This is what [[location(0)]] in the fragment shader targets
                    wgpu::RenderPassColorAttachment {
                        view: wgpu.view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: true,
                        },
                    },
                ],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: wgpu.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: true,
                    }),
                    stencil_ops: None,
                }),
            });

            render_pass.set_pipeline(pipeline);
            render_pass.set_bind_group(0, uniforms.next().unwrap()[0], &[]);
            render_pass.draw(0..4, 0..1);
        }
    }

    fn render_ui(
        &mut self,
        data: &mut App,
        _ctx: &CtxRef,
        menu_ui: &mut Ui,
        _actions: &mut UiActions,
    ) {
        egui::menu::menu(menu_ui, "Light", |ui| {
            egui::Grid::new("light_menu_grid")
                .num_columns(2)
                .spacing([40.0, 4.0])
                .striped(true)
                .show(ui, |ui| {
                    ui.label("size");
                    if egui::Slider::new(&mut self.uniform_data.size, 5.0..=50.0)
                        .ui(ui)
                        .changed()
                    {
                        data.dirty.insert(Dirty::LIGHT);
                    }
                    ui.end_row();
                })
        });
    }

    fn shader(&self, _data: &mut App, device: &wgpu::Device) -> ShaderAction {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("light bind group layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("light bind group"),
            layout: &layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(self.uniform.as_entire_buffer_binding()),
            }],
        });
        ShaderAction {
            create_bind_groups: vec![CreateBindGroup {
                name: "light",
                groups: vec![group],
                layout,
            }],
            create_pipeline: Some(CreatePipeline {
                src: "shader_src/light.wgsl",
                layout: &[],
                uniforms: array_vec!["light"],
                topology: wgpu::PrimitiveTopology::TriangleStrip,
            }),
        }
    }

    fn resize(&mut self, data: &mut App, size: PhysicalSize<u32>) {
        self.uniform_data.viewport_size = [size.width as f32, size.height as f32];
        data.dirty.insert(Dirty::LIGHT);
    }
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct RayTracerObjects {
    inverse_view_proj: [[f32; 4]; 4],
    view_proj: [[f32; 4]; 4],
}

pub(crate) struct RayTracerBuilder;
pub(crate) struct RayTracer {
    object_data: RayTracerObjects,
    uniform: wgpu::Buffer,
}

impl RendererBuilder<App> for RayTracerBuilder {
    type Output = RayTracer;

    fn build(self, data: &mut App, device: &Device, _size: PhysicalSize<u32>) -> Self::Output {
        let object_data = RayTracerObjects {
            inverse_view_proj: (OPENGL_TO_WGPU_MATRIX * data.view_proj)
                .invert()
                .expect("a view-projection matrix should be invertible")
                .into(),
            view_proj: (OPENGL_TO_WGPU_MATRIX * data.view_proj).into(),
        };
        let uniform = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Ray-tracer uniform buffer"),
            contents: bytemuck::cast_slice(&[object_data]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        Self::Output {
            object_data,
            uniform,
        }
    }
}

impl Renderer<App> for RayTracer {
    fn update(&mut self, data: &mut App) {
        if data.dirty.contains(Dirty::CAMERA) {
            data.dirty.insert(Dirty::RAY_TRACER);
            self.object_data.inverse_view_proj = (OPENGL_TO_WGPU_MATRIX * data.view_proj)
                .invert()
                .expect("a view-projection matrix should be invertible")
                .into();
            self.object_data.view_proj = (OPENGL_TO_WGPU_MATRIX * data.view_proj).into();
        }
    }

    fn render(&mut self, data: &mut App, wgpu: WGPU) {
        if data.dirty.contains(Dirty::RAY_TRACER) {
            data.dirty.remove(Dirty::RAY_TRACER);
            wgpu.queue
                .write_buffer(&self.uniform, 0, bytemuck::cast_slice(&[self.object_data]));
        }

        if let Some(pipeline) = wgpu.current_render_pipeline().as_ref() {
            let mut encoder = wgpu.command_encoder.borrow_mut();
            let mut uniforms = wgpu.current_uniforms();
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Ray Tracer Pass"),
                color_attachments: &[
                    // This is what [[location(0)]] in the fragment shader targets
                    wgpu::RenderPassColorAttachment {
                        view: wgpu.view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: true,
                        },
                    },
                ],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: wgpu.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: true,
                    }),
                    stencil_ops: None,
                }),
            });

            render_pass.set_pipeline(pipeline);
            render_pass.set_bind_group(0, uniforms.next().unwrap()[0], &[]);
            render_pass.draw(0..4, 0..1);
        }
    }

    fn shader(&self, _data: &mut App, device: &wgpu::Device) -> ShaderAction {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ray_tracer bind group layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ray_tracer bind group"),
            layout: &layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(self.uniform.as_entire_buffer_binding()),
            }],
        });
        ShaderAction {
            create_bind_groups: vec![CreateBindGroup {
                name: "ray_tracer",
                groups: vec![group],
                layout,
            }],
            create_pipeline: Some(CreatePipeline {
                src: "shader_src/ray_tracer.wgsl",
                layout: &[],
                uniforms: array_vec!["ray_tracer"],
                topology: wgpu::PrimitiveTopology::TriangleStrip,
            }),
        }
    }
}
