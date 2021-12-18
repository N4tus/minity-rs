use bytemuck::{Pod, Zeroable};
use cgmath::{Matrix4, SquareMatrix, Vector3};
use egui::CursorIcon::Default;
use itertools::Itertools;
use log::Level::Debug;
use native_dialog::FileDialog;
use std::collections::HashMap;
use std::ops::{Range, RangeBounds};
use std::path::PathBuf;
use tobj::{LoadOptions, Mesh};

#[derive(Copy, Clone, Debug, Pod, Zeroable)]
#[repr(C)]
pub(crate) struct Vertex {
    pos: [f32; 3],
    normal: [f32; 3],
    uv: [f32; 2],
}

impl Vertex {
    pub(crate) const fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x3,
                    offset: 0,
                    shader_location: 0,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x3,
                    offset: std::mem::size_of::<Vector3<f32>>() as wgpu::BufferAddress,
                    shader_location: 1,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x2,
                    offset: (std::mem::size_of::<Vector3<f32>>() * 2) as wgpu::BufferAddress,
                    shader_location: 2,
                },
            ],
        }
    }
}

pub(crate) struct Group {
    indices_start: usize,
    indices_end: usize,
    vertex_start: usize,
    vertex_end: usize,
    transformation: Matrix4<f32>,
    name: String,
    material: Option<usize>,
}

pub(crate) struct Model {
    pub(crate) vertices: Vec<Vertex>,
    pub(crate) indices: Vec<u32>,
    pub(crate) groups: Vec<Group>,
}

impl Group {
    pub(crate) fn vertex_range(&self) -> impl RangeBounds<wgpu::BufferAddress> {
        ((self.vertex_start * std::mem::size_of::<Vertex>()) as wgpu::BufferAddress)
            ..((self.vertex_end * std::mem::size_of::<Vertex>()) as wgpu::BufferAddress)
    }

    pub(crate) fn index_range(&self) -> Range<u32> {
        (self.indices_start as u32)..(self.indices_end as u32)
    }
}

#[derive(Debug)]
pub(crate) enum LoadError {
    NativeError(native_dialog::Error),
    NoModelToLoad,
    Other(String),
    TObjError(tobj::LoadError),
}

impl From<tobj::LoadError> for LoadError {
    fn from(err: tobj::LoadError) -> Self {
        Self::TObjError(err)
    }
}

impl From<native_dialog::Error> for LoadError {
    fn from(err: native_dialog::Error) -> Self {
        Self::NativeError(err)
    }
}

fn load_model_with_path(path: impl Into<PathBuf>) -> Result<Model, LoadError> {
    let mut path = path.into();
    path.set_extension("obj");
    if !path.is_file() {
        return Err(LoadError::Other(format!(
            "`{}` is not a file",
            path.to_string_lossy()
        )));
    }
    if !path.exists() {
        return Err(LoadError::Other(format!(
            "`{}` does not exist",
            path.to_string_lossy()
        )));
    }
    let (model, material) = tobj::load_obj(
        path,
        &LoadOptions {
            single_index: true,
            triangulate: true,
            ignore_lines: true,
            ignore_points: true,
            ..LoadOptions::default()
        },
    )?;
    let material = material?;

    let mut position_count = 0usize;
    let mut normal_count = 0usize;
    let mut uv_count = 0usize;
    let mut index_count = 0usize;
    for model in &model {
        position_count += model.mesh.positions.len();
        normal_count += model.mesh.normals.len();
        uv_count += model.mesh.texcoords.len();
        index_count += model.mesh.indices.len();
    }

    let mut vertex_buffer_data = Vec::with_capacity(position_count + normal_count + uv_count);
    let mut index_buffer_data = vec![0u32; index_count];
    let mut groups = Vec::with_capacity(model.len());

    let mut current_index_buffer_index = 0usize;
    let mut current_vertex_buffer_index = 0usize;
    for model in model {
        let vertex_count = model.mesh.positions.len() / 3;
        for idx in 0..vertex_count {
            vertex_buffer_data.push(Vertex {
                pos: [
                    model.mesh.positions[idx * 3],
                    model.mesh.positions[idx * 3 + 1],
                    model.mesh.positions[idx * 3 + 2],
                ],
                normal: [
                    model.mesh.normals.get(idx * 3).copied().unwrap_or_default(),
                    model
                        .mesh
                        .normals
                        .get(idx * 3 + 1)
                        .copied()
                        .unwrap_or_default(),
                    model
                        .mesh
                        .normals
                        .get(idx * 3 + 2)
                        .copied()
                        .unwrap_or_default(),
                ],
                uv: [
                    model
                        .mesh
                        .texcoords
                        .get(idx * 2)
                        .copied()
                        .unwrap_or_default(),
                    model
                        .mesh
                        .texcoords
                        .get(idx * 2 + 1)
                        .copied()
                        .unwrap_or_default(),
                ],
            });
        }
        index_buffer_data
            [current_index_buffer_index..(current_index_buffer_index + model.mesh.indices.len())]
            .copy_from_slice(model.mesh.indices.as_slice());

        groups.push(Group {
            indices_start: current_index_buffer_index,
            indices_end: current_index_buffer_index + model.mesh.indices.len(),
            vertex_start: current_vertex_buffer_index,
            vertex_end: current_vertex_buffer_index + vertex_count,
            transformation: cgmath::Matrix4::identity(),
            name: model.name,
            material: model.mesh.material_id,
        });

        current_index_buffer_index += model.mesh.indices.len();
        current_vertex_buffer_index += vertex_count;
    }

    Ok(Model {
        vertices: vertex_buffer_data,
        indices: index_buffer_data,
        groups,
    })
}

pub(crate) fn load_model() -> Result<Model, LoadError> {
    std::env::args()
        .nth(1)
        .map(load_model_with_path)
        .unwrap_or_else(load_model_with_dialog)
}

fn load_model_with_dialog() -> Result<Model, LoadError> {
    FileDialog::new()
        .set_location(&std::env::current_dir().unwrap_or_else(|e| {
            log::error!("{}", e);
            PathBuf::new()
        }))
        .add_filter("Object", &["obj"])
        .show_open_single_file()?
        .map(load_model_with_path)
        .unwrap_or(Err(LoadError::NoModelToLoad))
}
