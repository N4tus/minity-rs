// Vertex shader

struct CameraUniform {
    view_proj: mat4x4<f32>;
};

struct Material {
    ambient: vec3<f32>;
    diffuse: vec3<f32>;
    specular: vec3<f32>;
    shininess: f32;
};

[[group(0), binding(0)]] // 2.
var<uniform> camera: CameraUniform;

[[group(1), binding(0)]]
var<uniform> material: Material;

struct Vertex  {
    [[builtin(vertex_index)]] index: u32;
    [[location(0)]] pos: vec3<f32>;
    [[location(1)]] normal: vec3<f32>;
    [[location(2)]] uv: vec2<f32>;
};

struct VertexOutput {
    [[builtin(position)]] clip_position: vec4<f32>;
    [[location(0)]] uv: vec2<f32>;
};

[[stage(vertex)]]
fn vs_main(
    vertex: Vertex
) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = camera.view_proj * vec4<f32>(vertex.pos, 1.0);
    out.uv = vertex.uv;
    return out;
}

// Fragment shader

[[group(2), binding(0)]]
var t_ambient: texture_2d<f32>;
[[group(2), binding(1)]]
var s_ambient: sampler;
[[group(2), binding(2)]]
var t_diffuse: texture_2d<f32>;
[[group(2), binding(3)]]
var s_diffuse: sampler;
[[group(2), binding(4)]]
var t_specular: texture_2d<f32>;
[[group(2), binding(5)]]
var s_specular: sampler;
[[group(2), binding(6)]]
var t_shininess: texture_2d<f32>;
[[group(2), binding(7)]]
var s_shininess: sampler;

[[stage(fragment)]]
fn fs_main(in: VertexOutput) -> [[location(0)]] vec4<f32> {
    var g = textureSample(t_diffuse, s_diffuse, in.uv);
    if (g.a < 0.1) {
        discard;
    }
    //g.a = 1.0 - g.a;
    return g;
}
