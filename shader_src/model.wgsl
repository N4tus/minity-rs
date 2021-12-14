// Vertex shader

[[block]] // 1.
struct CameraUniform {
    view_proj: mat4x4<f32>;
};

[[group(0), binding(0)]] // 2.
var<uniform> camera: CameraUniform;

struct Vertex  {
    [[location(0)]] pos: vec3<f32>;
    [[location(1)]] normal: vec3<f32>;
    [[location(2)]] uv: vec2<f32>;
};

struct VertexOutput {
    [[builtin(position)]] clip_position: vec4<f32>;
    [[location(0)]] color: vec4<f32>;
};

[[stage(vertex)]]
fn vs_main(
    vertex: Vertex
) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = camera.view_proj * vec4<f32>(vertex.pos, 1.0);
    out.color = vec4<f32>(vertex.pos, 1.0);
    return out;
}

// Fragment shader

[[stage(fragment)]]
fn fs_main(in: VertexOutput) -> [[location(0)]] vec4<f32> {
    return in.color;
}
