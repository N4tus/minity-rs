use bytemuck::{Pod, Zeroable};
use cgmath::{Matrix4, SquareMatrix, Vector2, Vector3, Zero};
use itertools::{EitherOrBoth, Itertools};
use native_dialog::FileDialog;
use nobject_rs::{load_mtl, load_obj, ObjError};
use std::collections::HashMap;
use std::path::PathBuf;

#[derive(Copy, Clone, Debug, Pod, Zeroable)]
#[repr(C)]
pub(crate) struct Vertex {
    pos: [f32; 3],
    normal: [f32; 3],
    uv: [f32; 2],
}

impl Vertex {
    pub(crate) fn desc() -> wgpu::VertexBufferLayout<'static> {
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
    indices: Vec<usize>,
    transformation: Matrix4<f32>,
}

pub(crate) struct Model {
    pub(crate) vertices: Vec<Vertex>,
    pub(crate) groups: HashMap<String, Group>,
}

#[derive(Debug)]
pub(crate) enum LoadError {
    NativeError(native_dialog::Error),
    ObjError(ObjError),
    NoModelToLoad,
    Other(String),
    FormatError(String),
    IoError(std::io::Error),
}

impl From<native_dialog::Error> for LoadError {
    fn from(err: native_dialog::Error) -> Self {
        Self::NativeError(err)
    }
}

impl From<ObjError> for LoadError {
    fn from(err: ObjError) -> Self {
        Self::ObjError(err)
    }
}

impl From<std::io::Error> for LoadError {
    fn from(err: std::io::Error) -> Self {
        Self::IoError(err)
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
    let base_path = path
        .parent()
        .ok_or_else(|| {
            LoadError::Other(format!(
                "Cannot get parent of: `{}`",
                path.to_string_lossy()
            ))
        })?
        .to_path_buf();
    let obj_data = std::fs::read_to_string(path)?;
    let obj = load_obj(&obj_data)?;

    let mut mats = Vec::new();
    for lib_path in &obj.material_libs {
        let mut base_path = base_path.clone();
        base_path.push(lib_path);
        let mlt_data = std::fs::read_to_string(base_path)?;
        let mlt = load_mtl(&mlt_data)?;
        mats.push(mlt);
    }

    if (obj.vertices.len() != obj.normals.len() && !obj.normals.is_empty())
        || (obj.vertices.len() != obj.textures.len() && !obj.textures.is_empty())
    {
        return Err(LoadError::Other(format!(
            "different sizes of vertices ({}), normals ({}), and textures ({})",
            obj.vertices.len(),
            obj.normals.len(),
            obj.textures.len()
        )));
    }

    let unwrap = |v: Option<f32>| {
        v.ok_or_else(|| LoadError::FormatError("A uv coordinate has no v component".to_string()))
    };
    let mut vertex_data = Vec::<Vertex>::with_capacity(obj.vertices.len());
    for (index, vertex) in obj.vertices.into_iter().enumerate() {
        let normal = obj
            .normals
            .get(index)
            .map(|n| [n.x, n.y, n.z])
            .unwrap_or([0.0; 3]);
        let uv = obj.textures.get(index);
        let uv = if let Some(uv) = uv {
            [uv.u, unwrap(uv.v)?]
        } else {
            [0.0; 2]
        };
        vertex_data.push(Vertex {
            pos: [vertex.x, vertex.y, vertex.z],
            normal,
            uv,
        });
    }

    let mut groups = HashMap::new();

    for (name, group) in obj.faces {
        let mut indices = Vec::with_capacity(group.len() * 3);
        for face in group {
            if face.elements.len() != 3 {
                return Err(LoadError::FormatError(
                    "A face must be a triangle.".to_string(),
                ));
            }
            for f in face.elements {
                let v = (f.vertex_index - 1) as usize;
                indices.push(v);
            }
        }
        groups.insert(
            name,
            Group {
                indices,
                transformation: Matrix4::identity(),
            },
        );
    }

    Ok(Model {
        vertices: vertex_data,
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
