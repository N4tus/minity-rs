use crate::{Renderer, RendererBuilder, ShaderAction, UiActions, WGPU};
use egui::{CtxRef, Ui};
use tuple_list::TupleList;
use wgpu::Device;
use winit::dpi::PhysicalSize;
use winit::event::WindowEvent;

impl<Data> RendererBuilder<Data> for () {
    type Output = ();

    fn build(
        self,
        _data: &mut Data,
        _device: &wgpu::Device,
        _size: PhysicalSize<u32>,
    ) -> Self::Output {
    }
}

impl<Data> Renderer<Data> for () {
    fn visit(
        &self,
        _data: &mut Data,
        _actions: &mut [ShaderAction],
        _index: usize,
        _device: &Device,
    ) {
    }
}

impl<Data, Head, Tail> Renderer<Data> for (Head, Tail)
where
    Head: Renderer<Data>,
    Tail: Renderer<Data> + TupleList,
{
    fn visit(
        &self,
        data: &mut Data,
        actions: &mut [ShaderAction],
        index: usize,
        device: &wgpu::Device,
    ) {
        self.0.visit(data, actions, index, device);
        self.1.visit(data, actions, index + 1, device);
    }

    fn update(&mut self, data: &mut Data) {
        self.0.update(data);
        self.1.update(data);
    }

    fn render(&mut self, data: &mut Data, mut wgpu: WGPU) {
        self.0.render(data, wgpu.clone());
        wgpu.renderer_index += 1;
        self.1.render(data, wgpu);
    }

    fn render_ui(
        &mut self,
        data: &mut Data,
        ctx: &CtxRef,
        menu_ui: &mut Ui,
        actions: &mut UiActions,
    ) {
        self.0.render_ui(data, ctx, menu_ui, actions);
        self.1.render_ui(data, ctx, menu_ui, actions);
    }

    fn input(&mut self, data: &mut Data, event: &WindowEvent) -> bool {
        let l = self.0.input(data, event);
        let r = self.1.input(data, event);
        l || r
    }

    fn resize(&mut self, data: &mut Data, size: PhysicalSize<u32>) {
        self.0.resize(data, size);
        self.1.resize(data, size);
    }
}

impl<Data, Head, Tail> RendererBuilder<Data> for (Head, Tail)
where
    Head: RendererBuilder<Data>,
    Tail: RendererBuilder<Data> + TupleList,
    <Tail as RendererBuilder<Data>>::Output: TupleList,
{
    type Output = (Head::Output, Tail::Output);

    fn build(
        self,
        data: &mut Data,
        device: &wgpu::Device,
        size: PhysicalSize<u32>,
    ) -> Self::Output {
        (
            self.0.build(data, device, size),
            self.1.build(data, device, size),
        )
    }
}
