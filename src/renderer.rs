use crate::objects::Vertex;
use crate::{Model, Renderer, RendererBuilder, ShaderAction, UiActions, WGPU};
use cgmath::{InnerSpace, Rotation, Rotation3, SquareMatrix};
use egui::{CtxRef, Ui};
use std::cell::RefCell;
use std::rc::Rc;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{BufferUsages, VertexBufferLayout};
use winit::dpi::PhysicalSize;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};

mod shader;

type ModelData = Rc<RefCell<Option<Model>>>;

pub(crate) struct ModelRenderer {
    model: ModelData,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
}

pub(crate) struct ModelRendererBuilder {
    model: ModelData,
}

impl ModelRendererBuilder {
    pub(crate) fn new(model: ModelData) -> Self {
        Self { model }
    }
}

impl RendererBuilder for ModelRendererBuilder {
    type Output = ModelRenderer;

    fn build(self, device: &wgpu::Device, _size: PhysicalSize<u32>) -> Self::Output {
        let (vertex_buffer, index_buffer) = {
            let model = RefCell::borrow(&*self.model);
            let (data, indices) = if let Some(m) = &*model {
                (
                    bytemuck::cast_slice(m.vertices.as_slice()),
                    bytemuck::cast_slice(m.indices.as_slice()),
                )
            } else {
                ([0u8; 0].as_slice(), [0u8; 0].as_slice())
            };
            (
                device.create_buffer_init(&BufferInitDescriptor {
                    label: Some("model vertex attribute buffer"),
                    contents: data,
                    usage: BufferUsages::VERTEX,
                }),
                device.create_buffer_init(&BufferInitDescriptor {
                    label: Some("model index buffer"),
                    contents: indices,
                    usage: BufferUsages::INDEX,
                }),
            )
        };
        Self::Output {
            model: self.model,
            vertex_buffer,
            index_buffer,
        }
    }
}

impl Renderer for ModelRenderer {
    fn render(&mut self, wgpu: WGPU) {
        if let Some(model) = &*RefCell::borrow(&*self.model) {
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
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: 0.1,
                                g: 0.2,
                                b: 0.3,
                                a: 1.0,
                            }),
                            store: true,
                        },
                    },
                ],
                depth_stencil_attachment: None,
            });

            // NEW!
            render_pass.set_pipeline(wgpu.current_render_pipeline().as_ref().unwrap()); // 2.

            render_pass.set_bind_group(0, uniforms.next().unwrap(), &[]);
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            render_pass.draw_indexed(0..model.indices.len() as u32, 0, 0..1);
            // 3.
        }
    }

    fn render_ui(&mut self, _ctx: &CtxRef, menu_ui: &mut Ui, actions: &mut UiActions) {
        egui::menu::menu(menu_ui, "Config", |ui| {
            if ui.button("Quit").clicked() {
                actions.quit = true;
            }
        });
    }

    fn input(&mut self, _event: &WindowEvent) -> bool {
        false
    }

    fn shader(&self) -> Option<ShaderAction> {
        Some(ShaderAction::CreatePipeline {
            src: "shader_src/model.wgsl",
            layout: VERTEX_LAYOUT,
            uniforms: &["camera"],
        })
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
    mat: cgmath::Matrix4<f32>,

    left_mouse_pressed: bool,
    mouse_pos: cgmath::Point2<f32>,
    x_axis: cgmath::Vector3<f32>,
    y_axis: cgmath::Vector3<f32>,
    dirty: bool,
}

#[rustfmt::skip]
pub const OPENGL_TO_WGPU_MATRIX: cgmath::Matrix4<f32> = cgmath::Matrix4::new(
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.5, 0.0,
    0.0, 0.0, 0.5, 1.0,
);

impl RendererBuilder for CameraBuilder {
    type Output = Camera;

    fn build(self, device: &wgpu::Device, size: PhysicalSize<u32>) -> Self::Output {
        let aspect = size.width as f32 / size.height as f32;
        let view = cgmath::Matrix4::look_at_rh(self.eye, self.target, cgmath::Vector3::unit_y());
        let proj = cgmath::perspective(cgmath::Deg(self.fovy), aspect, self.znear, self.zfar);

        let mat = OPENGL_TO_WGPU_MATRIX * proj * view;

        let uniform = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Camera uniform buffer"),
            contents: bytemuck::cast_slice::<[[f32; 4]; 4], u8>(&[mat.into()]),
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
            mat: cgmath::Matrix4::identity(),
            left_mouse_pressed: false,
            mouse_pos: cgmath::Point2::new(0.0, 0.0),
            x_axis: cgmath::Vector3::unit_x(),
            y_axis: cgmath::Vector3::unit_y(),
            dirty: false,
        }
    }
}

impl Camera {
    fn update_camera(&mut self) {
        let view = cgmath::Matrix4::look_at_rh(self.eye, self.target, self.up);
        let proj = cgmath::perspective(cgmath::Deg(self.fovy), self.aspect, self.znear, self.zfar);

        self.mat = OPENGL_TO_WGPU_MATRIX * proj * view;
        self.dirty = true;
    }
}

impl Renderer for Camera {
    fn render(&mut self, wgpu: WGPU) {
        if self.dirty {
            self.dirty = false;
            wgpu.queue.write_buffer(
                &self.uniform,
                0,
                bytemuck::cast_slice::<[[f32; 4]; 4], u8>(&[self.mat.into()]),
            );
        }
    }

    fn input(&mut self, event: &WindowEvent) -> bool {
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

                    self.eye = qrot.rotate_point(self.eye);
                    self.up = qrot.rotate_vector(self.up);
                    self.x_axis = qrot.rotate_vector(self.x_axis);
                    self.y_axis = qrot.rotate_vector(self.y_axis);

                    self.update_camera();

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
                if delta >= 0.0 {
                    self.eye /= delta + 1.0;
                } else {
                    self.eye *= -delta + 1.0;
                }
                self.update_camera();
                true
            }
            _ => false,
        }
    }

    fn shader(&self) -> Option<ShaderAction> {
        Some(ShaderAction::AddUniform {
            name: "camera",
            binding: 0,
            shader_stage: wgpu::ShaderStages::VERTEX,
            buffer: &self.uniform,
        })
    }

    fn resize(&mut self, size: PhysicalSize<u32>) {
        self.aspect = size.width as f32 / size.height as f32;
        self.update_camera();
    }
}
