use crate::objects::Vertex;
use crate::{Model, Renderer, RendererBuilder, UiActions, WGPU};
use cgmath::Vector3;
use egui::{CtxRef, Ui};
use egui_wgpu_backend::wgpu::VertexBufferLayout;
use std::borrow::Borrow;
use std::cell::RefCell;
use std::ops::Deref;
use std::rc::Rc;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::BufferUsages;
use winit::event::WindowEvent;

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

    fn build(self, device: &wgpu::Device) -> Self::Output {
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
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            render_pass.draw_indexed(0..model.indices.len() as u32, 0, 0..1); // 3.
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

    fn shader(&self) -> Option<(&str, VertexBufferLayout)> {
        Some(("shader_src/model.wgsl", Vertex::desc()))
    }
}
