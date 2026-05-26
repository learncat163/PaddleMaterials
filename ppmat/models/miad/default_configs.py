# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

class _Cfg:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, dict):
                setattr(self, k, _Cfg(**v))
            else:
                setattr(self, k, v)


def _make_config(method, lat_diffusion, frac_diffusion, type_diffusion):
    return _Cfg(
        method=method,
        cont_time=False,
        num_steps=1000,
        task='',
        lat_diffusion=lat_diffusion,
        frac_diffusion=frac_diffusion,
        type_diffusion=type_diffusion,
    )


_lat = {
    'diffcsp_ddpm_config': _Cfg(method='ddpm', scheduler='diffcsp_cosine', cond_coef=1.),
    'ddpm_config': _Cfg(method='ddpm', scheduler='cosine', cond_coef=1.),
    'fm_config': _Cfg(method='fm', cond_coef=1., parameterization='eps'),
    'fm_v_config': _Cfg(method='fm', cond_coef=1., parameterization='v'),
    'fm_lenang_config': _Cfg(method='fm_lenang', cond_coef=1.),
}

_coord = {
    'wrapped_normal_config': _Cfg(method='wrapped_normal', scheduler='default_wrapped_normal', cond_coef=1.),
    'pfm_config': _Cfg(method='pfm', cond_coef=1.),
}

_type = {
    'd3pm_config': _Cfg(method='d3pm', scheduler='default_d3pm', cond_coef=1.),
    'ddpm_onehot_config': _Cfg(method='ddpm_onehot', scheduler='diffcsp_cosine', cond_coef=1.),
}

DEFAULT_DIFFUSION_CONFIGS = {
    'DiffCSP:diffcsp_ddpm_config:wrapped_normal_config:d3pm_config':
        _make_config('DiffCSP', _lat['diffcsp_ddpm_config'], _coord['wrapped_normal_config'], _type['d3pm_config']),
    'Default:ddpm_config:wrapped_normal_config:d3pm_config':
        _make_config('Default', _lat['ddpm_config'], _coord['wrapped_normal_config'], _type['d3pm_config']),
    'Default:fm_config:wrapped_normal_config:d3pm_config':
        _make_config('Default', _lat['fm_config'], _coord['wrapped_normal_config'], _type['d3pm_config']),
    'DiffCSP:ddpm_config:wrapped_normal_config:ddpm_onehot_config':
        _make_config('DiffCSP', _lat['ddpm_config'], _coord['wrapped_normal_config'], _type['ddpm_onehot_config']),
    'DiffCSP:fm_config:wrapped_normal_config:d3pm_config':
        _make_config('DiffCSP', _lat['fm_config'], _coord['wrapped_normal_config'], _type['d3pm_config']),
}
