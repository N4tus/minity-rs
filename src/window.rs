use crate::RenderBackend;
use egui::{egui_assert, FontDefinitions};
use egui_winit_platform::{Platform, PlatformDescriptor};
use std::time::Instant;
use winit::dpi::PhysicalSize;
use winit::event::{self, ElementState, KeyboardInput, VirtualKeyCode, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::WindowBuilder;

pub(crate) struct Window {
    pub(crate) window: winit::window::Window,
    event_loop: EventLoop<()>,
}

impl Window {
    pub(crate) fn new() -> Self {
        let event_loop = EventLoop::with_user_event();
        Self {
            window: WindowBuilder::new().build(&event_loop).unwrap(),
            event_loop,
        }
    }

    pub(crate) fn run(self, mut renderer: impl RenderBackend + 'static) -> ! {
        let size = self.window.inner_size();
        let mut platform = Platform::new(PlatformDescriptor {
            physical_width: size.width,
            physical_height: size.height,
            scale_factor: self.window.scale_factor(),
            font_definitions: FontDefinitions::default(),
            style: Default::default(),
        });
        let start_time = Instant::now();
        renderer.init();
        self.event_loop.run(move |event, _, control_flow| {
            *control_flow = ControlFlow::Wait;
            platform.handle_event(&event);
            let egui_handled_event = platform.captures_event(&event);
            match event {
                event::Event::WindowEvent {
                    ref event,
                    window_id,
                } if window_id == self.window.id() => {
                    if egui_handled_event || !renderer.input(event) {
                        match event {
                            WindowEvent::CloseRequested
                            | WindowEvent::KeyboardInput {
                                input:
                                    KeyboardInput {
                                        state: ElementState::Pressed,
                                        virtual_keycode: Some(VirtualKeyCode::Escape),
                                        ..
                                    },
                                ..
                            } => *control_flow = ControlFlow::Exit,
                            WindowEvent::Resized(physical_size) => {
                                renderer.resize(*physical_size);
                            }
                            WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                                renderer.resize(**new_inner_size);
                            }
                            _ => {}
                        }
                    }
                }
                event::Event::RedrawRequested(_) => {
                    platform.update_time(start_time.elapsed().as_secs_f64());

                    renderer.update();
                    match renderer.render(
                        self.window.scale_factor() as _,
                        &mut platform,
                        &|platform| platform.end_frame(Some(&self.window)).1,
                    ) {
                        Ok(actions) => {
                            if actions.quit {
                                *control_flow = ControlFlow::Exit
                            }
                        }
                        Err(wgpu::SurfaceError::Lost) => renderer.resize(PhysicalSize::new(0, 0)),
                        Err(wgpu::SurfaceError::OutOfMemory) => *control_flow = ControlFlow::Exit,
                        Err(e) => eprintln!("{:?}", e),
                    };
                }
                event::Event::MainEventsCleared => {
                    self.window.request_redraw();
                }
                _ => {}
            }
        })
    }
}
