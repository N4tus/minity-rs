use crate::array_vec::ArrayVec;
use bytemuck::{Pod, Zeroable};
use cgmath::{Matrix4, SquareMatrix, Vector3};
use image::{DynamicImage, ImageError, ImageFormat};
use native_dialog::FileDialog;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Error};
use std::ops::Range;
use std::path::PathBuf;
use tobj::LoadOptions;

pub const MAX_MATERIALS: usize = 8;
pub const UNIFORM_ALIGNMENT: usize = 256;

#[derive(Copy, Clone, Debug, Pod, Zeroable)]
#[repr(C)]
pub(crate) struct MaterialData {
    /// Ambient color of the material.
    pub ambient: [f32; 3],
    // _d are padding
    _0: f32,
    /// Diffuse color of the material.
    pub diffuse: [f32; 3],
    _1: f32,
    /// Specular color of the material.
    pub specular: [f32; 3],
    /// Material shininess attribute. Also called `glossiness`.
    pub shininess: f32,
}

pub const PADDING: usize =
    UNIFORM_ALIGNMENT - (std::mem::size_of::<MaterialData>() % UNIFORM_ALIGNMENT);

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub(crate) struct MaterialDataPadding {
    data: MaterialData,
    /// manually align to UNIFORM_ALIGNMENT
    _p: [u8; PADDING],
}

impl MaterialDataPadding {
    pub(crate) fn new(data: MaterialData) -> Self {
        Self {
            data,
            _p: [0; PADDING],
        }
    }
}

// SAFETY:
// data is Pod
// _p is Pod
// there is no padding between data and _p
// there is no padding after _p
unsafe impl Zeroable for MaterialDataPadding {}
unsafe impl Pod for MaterialDataPadding {}

impl Default for MaterialData {
    fn default() -> Self {
        Self {
            ambient: [0.0; 3],
            _0: 0.0,
            diffuse: [0.5; 3],
            _1: 0.0,
            specular: [1.0; 3],
            shininess: 50.0,
        }
    }
}

pub(crate) struct Material {
    /// Material name as specified in the `MTL` file.
    pub name: String,

    /// Ambient texture index for the material
    pub ambient_texture: Option<usize>,
    /// Diffuse texture index for the material.
    pub diffuse_texture: Option<usize>,
    /// Specular texture index for the material.
    pub specular_texture: Option<usize>,
    // ///Normal map texture index for the material.
    // pub normal_texture: DynamicImage,
    /// Shininess map texture index for the material.
    pub shininess_texture: Option<usize>,
}

pub(crate) struct MaterialInfo {
    pub(crate) shader_data: ArrayVec<MaterialDataPadding, MAX_MATERIALS>,
    pub(crate) material_info: ArrayVec<Material, MAX_MATERIALS>,
}

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
    pub(crate) materials: MaterialInfo,
    pub(crate) images_storage: Vec<DynamicImage>,
}

impl Group {
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
    IOError(std::io::Error),
    ImageError(image::ImageError),
}

impl From<image::ImageError> for LoadError {
    fn from(err: ImageError) -> Self {
        Self::ImageError(err)
    }
}

impl From<std::io::Error> for LoadError {
    fn from(err: Error) -> Self {
        Self::IOError(err)
    }
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
    // let mut index_buffer_data = vec![0u32; index_count];
    let mut index_buffer_data = Vec::with_capacity(index_count);
    let mut groups = Vec::with_capacity(model.len());

    let mut current_index_buffer_index = 0usize;
    let mut current_vertex_buffer_index = 0usize;
    for model in model {
        let vertex_count = model.mesh.positions.len() / 3;
        let index_offset = vertex_buffer_data.len() as u32;
        for &idx in &model.mesh.indices {
            index_buffer_data.push(idx + index_offset);
        }
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

    let mut materials = ArrayVec::new();
    let mut images_storage = Vec::new();
    let mut images_set = HashMap::new();
    let mut material_data = ArrayVec::new();
    for material in material.into_iter().take(MAX_MATERIALS) {
        log::info!(
            "Material `{}` images:\n\tambient: {}\n\tdiffuse: {}\n\tspecular: {}\n\tshininess: {}",
            material.name,
            material.ambient_texture,
            material.diffuse_texture,
            material.specular_texture,
            material.shininess_texture
        );
        let ambient_texture = if !material.ambient_texture.is_empty() {
            if let Some(&idx) = images_set.get(&material.ambient_texture) {
                Some(idx)
            } else {
                let img_idx = images_storage.len();
                images_storage.push(image::load(
                    BufReader::new(File::open(&material.ambient_texture)?),
                    ImageFormat::Png,
                )?);
                images_set.insert(material.ambient_texture, img_idx);
                Some(img_idx)
            }
        } else {
            None
        };
        let diffuse_texture = if !material.diffuse_texture.is_empty() {
            if let Some(&idx) = images_set.get(&material.diffuse_texture) {
                Some(idx)
            } else {
                let img_idx = images_storage.len();
                images_storage.push(image::load(
                    BufReader::new(File::open(&material.diffuse_texture)?),
                    ImageFormat::Png,
                )?);
                images_set.insert(material.diffuse_texture, img_idx);
                Some(img_idx)
            }
        } else {
            None
        };
        let specular_texture = if !material.specular_texture.is_empty() {
            if let Some(&idx) = images_set.get(&material.specular_texture) {
                Some(idx)
            } else {
                let img_idx = images_storage.len();
                images_storage.push(image::load(
                    BufReader::new(File::open(&material.specular_texture)?),
                    ImageFormat::Png,
                )?);
                images_set.insert(material.specular_texture, img_idx);
                Some(img_idx)
            }
        } else {
            None
        };
        let shininess_texture = if !material.shininess_texture.is_empty() {
            if let Some(&idx) = images_set.get(&material.shininess_texture) {
                Some(idx)
            } else {
                let img_idx = images_storage.len();
                images_storage.push(image::load(
                    BufReader::new(File::open(&material.shininess_texture)?),
                    ImageFormat::Png,
                )?);
                images_set.insert(material.shininess_texture, img_idx);
                Some(img_idx)
            }
        } else {
            None
        };
        materials.push(Material {
            name: material.name,
            ambient_texture,
            diffuse_texture,
            specular_texture,
            // normal_texture: "".to_string(),
            shininess_texture,
        });
        material_data.push(MaterialDataPadding::new(MaterialData {
            ambient: material.ambient,
            _0: 0.0,
            diffuse: material.diffuse,
            _1: 0.0,
            specular: material.specular,
            shininess: material.shininess,
        }));
    }

    Ok(Model {
        vertices: vertex_buffer_data,
        indices: index_buffer_data,
        groups,
        materials: MaterialInfo {
            shader_data: material_data,
            material_info: materials,
        },
        images_storage,
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
