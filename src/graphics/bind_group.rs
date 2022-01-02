use crate::array_vec::{default_array, ArrayVec};
use crate::{CreatePipeline, ShaderAction, MAX_BIND_GROUPS};
use std::collections::HashMap;
use std::ops::Range;

pub(crate) struct BindGroupStorage<const RC: usize> {
    ///
    pub(crate) bind_groups: Vec<wgpu::BindGroup>,
    bind_group_layouts: Vec<wgpu::BindGroupLayout>,
    pub(crate) bind_group_association: [ArrayVec<Range<usize>, MAX_BIND_GROUPS>; RC],
    bind_group_layout_association: [ArrayVec<usize, MAX_BIND_GROUPS>; RC],
}

impl<const RC: usize> BindGroupStorage<RC> {
    pub(crate) fn empty() -> Self {
        Self {
            bind_groups: vec![],
            bind_group_layouts: vec![],
            bind_group_association: default_array(),
            bind_group_layout_association: default_array(),
        }
    }

    pub(crate) fn taking_from_actions(actions: &mut [ShaderAction; RC]) -> Self {
        let mut bind_group_count = 0_usize;
        let mut bind_group_layout_count = 0_usize;
        for action in actions.iter() {
            for create_bind_group in &action.create_bind_groups {
                bind_group_count += create_bind_group.groups.len();
            }
            bind_group_layout_count += action.create_bind_groups.len();
        }

        let mut bind_groups = Vec::with_capacity(bind_group_count);
        let mut bind_group_layouts = Vec::with_capacity(bind_group_layout_count);
        let mut bind_group_map = HashMap::new();
        let mut bind_group_layout_map = HashMap::new();
        for action in actions.iter_mut() {
            for crate_bind_group in std::mem::take(&mut action.create_bind_groups) {
                let bg = crate_bind_group.groups;
                bind_group_map.insert(
                    crate_bind_group.name,
                    bind_groups.len()..(bind_groups.len() + bg.len()),
                );
                bind_group_layout_map.insert(crate_bind_group.name, bind_group_layouts.len());
                bind_groups.extend(bg.into_iter());
                bind_group_layouts.push(crate_bind_group.layout);
            }
        }

        let mut bind_group_association: [ArrayVec<Range<usize>, MAX_BIND_GROUPS>; RC] =
            default_array();
        let mut bind_group_layout_association: [ArrayVec<usize, MAX_BIND_GROUPS>; RC] =
            default_array();

        for (index, action) in actions.iter().enumerate() {
            if let Some(CreatePipeline { uniforms, .. }) = &action.create_pipeline {
                for &uniform in uniforms.iter() {
                    let bg = unsafe { bind_group_association.get_unchecked_mut(index) };
                    if let Some(bg_idx) = bind_group_map.get(uniform) {
                        bg.push(bg_idx.clone());
                    } else {
                        log::warn!("`{}` is an unknown bind group name", uniform);
                    }

                    let bgl = unsafe { bind_group_layout_association.get_unchecked_mut(index) };
                    if let Some(&bgl_idx) = bind_group_layout_map.get(uniform) {
                        bgl.push(bgl_idx);
                    } else {
                        log::warn!("`{}` is an unknown bind group layout name", uniform);
                    }
                }
            }
        }
        Self {
            bind_groups,
            bind_group_layouts,
            bind_group_association,
            bind_group_layout_association,
        }
    }

    pub(crate) fn bg_ref_list(
        &self,
        index: usize,
    ) -> ArrayVec<&wgpu::BindGroupLayout, MAX_BIND_GROUPS> {
        let mut res = ArrayVec::new();
        for bgl in self.bind_group_layout_association[index]
            .iter()
            .map(|&idx| unsafe { self.bind_group_layouts.get_unchecked(idx) })
        {
            res.push(bgl);
        }
        res
    }

    pub(crate) fn aligned_size_of_ref_bind_groups(&self) -> usize {
        std::mem::size_of::<&wgpu::BindGroup>() * self.bind_groups.len()
    }
}
