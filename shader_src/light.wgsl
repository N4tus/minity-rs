[[block]]
struct LightUniform {
    model_view_proj: mat4x4<f32>;
    position: vec3<f32>;
    viewport_size: vec2<f32>;
    size: f32;
};

[[group(0), binding(0)]]
var<uniform> light: LightUniform;

struct VertexOutput {
    [[builtin(position)]] clip_position: vec4<f32>;
    [[location(0)]] center: vec2<f32>;
    [[location(1)]] size: f32;
};

[[stage(vertex)]]
fn vs_main(
    [[builtin(vertex_index)]] vertex_index: u32
) -> VertexOutput {
    var corner_pos: vec2<f32>;
    switch (vertex_index)
    {
        case 0: {
            corner_pos = vec2<f32>(-1.0, -1.0);
            break;
        }
        case 1: {
            corner_pos = vec2<f32>(1.0, -1.0);
            break;
        }
        case 2: {
            corner_pos = vec2<f32>(-1.0, 1.0);
            break;
        }
        case 3: {
            corner_pos = vec2<f32>(1.0, 1.0);
            break;
        }
        default: {
            corner_pos = vec2<f32>(0.0);
            break;
        }
    }
    let pos = light.model_view_proj * vec4<f32>(light.position, 1.0);

    let center = (pos.xy / pos.w)+vec2<f32>(1.0)*light.viewport_size*0.5;
    let corner = center + corner_pos * light.size / 2.0;

    var out: VertexOutput;
    out.clip_position = vec4<f32>(corner / light.viewport_size * 2.0 - vec2<f32>(1.0), 0.0, 1.0);
    out.center = center;
    out.size = light.size;
    return out;
}

// Fragment shader

[[stage(fragment)]]
fn fs_main(in: VertexOutput) -> [[location(0)]] vec4<f32> {
    let distance = sqrt(4.0)*length(in.clip_position.xy-in.center)/in.size;
    if (distance >= 1.0) {
        discard;
    }
    return vec4<f32>(1.0-distance);
}
