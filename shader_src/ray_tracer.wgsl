struct RayTracerObjects {
    inverse_view_projection: mat4x4<f32>;
    view_projection: mat4x4<f32>;
};

[[group(0), binding(0)]]
var<uniform> objects: RayTracerObjects;

struct VertexOutput {
    [[builtin(position)]] clip_position: vec4<f32>;
    [[location(0)]] frag_position: vec2<f32>;
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

    var out: VertexOutput;
    out.clip_position = vec4<f32>(corner_pos, 0.0, 1.0);
    out.frag_position = corner_pos;
    return out;
}

// Fragment shader

struct FragmentOutput {
    [[builtin(frag_depth)]] depth: f32;
    [[location(0)]] color: vec4<f32>;
};

fn calc_depth(pos: vec3<f32>) -> f32 {
	let far: f32 = 1.0; // gl_DepthRange.far
	let near: f32 = -1.0; // gl_DepthRange.near
	let clip_space_pos: vec4<f32> = objects.view_projection * vec4<f32>(pos, 1.0);
	let ndc_depth: f32 = clip_space_pos.z / clip_space_pos.w;
	return (((far - near) * ndc_depth) + near + far) / 2.0;
}

[[stage(fragment)]]
fn fs_main(in: VertexOutput) -> FragmentOutput {
    var out: FragmentOutput;
    out.color = vec4<f32>(0.0);

	let near_raw = objects.inverse_view_projection * vec4<f32>(in.frag_position,-1.0,1.0);
	let near = near_raw / near_raw.w;

	let far_raw = objects.inverse_view_projection * vec4<f32>(in.frag_position,1.0,1.0);
	let far = far_raw / far_raw.w;

	// this is the setup for our viewing ray
	let ray_origin = near.xyz;
	var ray_direction: vec3<f32> = (far-near).xyz;
    ray_direction = normalize(ray_direction);
    //ray_direction.z = ray_direction.z / 2.0;

    let sphere_center = vec3<f32>(0.0, 2.0, 0.0);
    let sphere_radius = 1.0;


	let o_minus_c = ray_origin - sphere_center;

	let p = dot(ray_direction, o_minus_c);
	let r = sphere_radius;
	let q = dot(o_minus_c, o_minus_c) - (r*r);

	let discriminant = (p * p) - q;
	if (discriminant < 0.0)
	{
		discard;
	}

	let d_root = sqrt(discriminant);
	let t0 = -p - d_root;
	let t1 = -p + d_root ;

	var nearestPositiveHit: f32;

	if (t0 >= 0.0 && t1 >= 0.0) {
		nearestPositiveHit = min(t0, t1);
	} else {
        if (t0 >= 0.0) {
            nearestPositiveHit = t0;
        } else {
            if (t1 >= 0.0) {
                nearestPositiveHit = t1;
            } else {
                // both are negative, return bigger (closer to 0)
                nearestPositiveHit = max(t0, t1);
            }
        }
	}
	let hit_pos = ray_origin + nearestPositiveHit * ray_direction;

	out.color = vec4<f32>(1.0, 0.0, 0.0, 1.0);

	// using calcDepth, you can convert a ray position to an OpenGL z-value, so that intersections/occlusions with the
	// model geometry are handled correctly, e.g.: gl_FragDepth = calcDepth(nearestHit);
	// in case there is no intersection, you should get gl_FragDepth to 1.0, i.e., the output of the shader will be ignored
    out.depth = calc_depth(hit_pos);

    return out;
}
