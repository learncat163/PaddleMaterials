import argparse
import random
import warnings
from pathlib import Path

import numpy as np
import paddle
from pymatgen.core import Structure

from ppmat.models.chemeleon2.common.schema import CrystalBatch
from ppmat.models.chemeleon2.ldm_module.dit import DiT
from ppmat.models.chemeleon2.ldm_module.ldm import LDMModule
from ppmat.models.chemeleon2.vae_module.decoder import TransformerDecoder
from ppmat.models.chemeleon2.vae_module.encoder import TransformerEncoder
from ppmat.models.chemeleon2.vae_module.vae import VAEModule

# 从 Chemeleon2 项目的 src/data/num_atom_distributions.py 中定义
NUM_ATOM_DISTRIBUTIONS = {
    "mp-20": {
        1: 0.0021742334905660377,
        2: 0.021079009433962265,
        3: 0.019826061320754717,
        4: 0.15271226415094338,
        5: 0.047132959905660375,
        6: 0.08464770047169812,
        7: 0.021079009433962265,
        8: 0.07808814858490566,
        9: 0.03434551886792453,
        10: 0.0972877358490566,
        11: 0.013303360849056603,
        12: 0.09669811320754718,
        13: 0.02155807783018868,
        14: 0.06522700471698113,
        15: 0.014372051886792452,
        16: 0.06703272405660378,
        17: 0.00972877358490566,
        18: 0.053176591981132074,
        19: 0.010576356132075472,
        20: 0.08995430424528301,
    }
}


def create_empty_batch(num_atoms_list, device='gpu'):
    batch_size = len(num_atoms_list)
    total_atoms = sum(num_atoms_list)
    
    batch = CrystalBatch()
    batch.atom_types = paddle.randint(1, 95, [total_atoms])
    batch.frac_coords = paddle.rand([total_atoms, 3])
    batch.lengths = paddle.rand([batch_size, 3]) * 10 + 5
    batch.angles = paddle.rand([batch_size, 3]) * 60 + 60
    batch.num_atoms = paddle.to_tensor(num_atoms_list, dtype='int64')
    batch.batch = paddle.repeat_interleave(
        paddle.arange(batch_size), 
        paddle.to_tensor(num_atoms_list)
    )
    batch.token_idx = paddle.concat([
        paddle.arange(n) for n in num_atoms_list
    ])
    batch.num_nodes = total_atoms
    batch.num_graphs = batch_size
    
    return batch


def sample(
    num_samples: int = 100,
    batch_size: int = 50,
    num_atom_distribution: str = "mp-20",
    ldm_ckpt_path: str = "test-for-weight/converted_weights/ldm_paddle.pdparams",
    vae_ckpt_path: str = "test-for-weight/converted_weights/vae_paddle.pdparams",
    output_dir: str = "outputs/samples",
    sampling_steps: int = 50,
    device: str = "gpu",
    seed: int = 42,
):
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)
    print(f"设置随机种子: {seed}")
    
    print(f"使用设备: {device}")
    paddle.set_device(device)
    
    print("加载 VAE 模型...")
    encoder = TransformerEncoder(
        max_num_elements=100,
        d_model=512,
        nhead=8,
        dim_feedforward=2048,
        activation='gelu',
        dropout=0.0,
        num_layers=8
    )
    
    decoder = TransformerDecoder(
        max_num_elements=100,
        d_model=512,
        nhead=8,
        dim_feedforward=2048,
        activation='gelu',
        dropout=0.0,
        num_layers=8
    )
    
    vae = VAEModule(
        encoder=encoder,
        decoder=decoder,
        latent_dim=8,
        loss_weights={}
    )
    vae_state = paddle.load(vae_ckpt_path)
    vae.set_state_dict(vae_state)
    vae.eval()
    print(f"VAE 模型加载完成: {vae_ckpt_path}")
    
    print("加载 LDM 模型...")
    denoiser = DiT(
        hidden_dim=768,
        num_layers=12,
        num_heads=12,
        mlp_ratio=4.0,
        input_dim=8,
        learn_sigma=True
    )
    
    ldm = LDMModule(
        denoiser=denoiser,
        normalize_latent=True,
        diffusion_configs={
            'timestep_respacing': '',
            'noise_schedule': 'linear',
            'use_kl': False,
            'sigma_small': False,
            'predict_xstart': False,
            'learn_sigma': True,
            'rescale_learned_sigmas': False,
            'diffusion_steps': 1000,
        }
    )
    ldm_state = paddle.load(ldm_ckpt_path)
    ldm.set_state_dict(ldm_state)
    ldm.vae = vae
    ldm.eval()
    print(f"LDM 模型加载完成: {ldm_ckpt_path}")
    
    num_atom_dist = NUM_ATOM_DISTRIBUTIONS[num_atom_distribution]
    probs = np.array(list(num_atom_dist.values()))
    probs = probs / probs.sum()
    num_atoms = np.random.choice(
        list(num_atom_dist.keys()),
        p=probs,
        size=num_samples,
    ).tolist()
    print(f"生成任务: {num_samples} 个样本")
    
    output_path = Path(output_dir)
    if output_path.exists():
        existing_cif_files = list(output_path.glob("sample_*.cif"))
        if existing_cif_files:
            warnings.warn(
                f"输出目录 '{output_path}' 已存在，包含 {len(existing_cif_files)} 个 CIF 文件。"
                "新样本将与现有文件一起生成。",
                UserWarning,
            )
    else:
        output_path.mkdir(parents=True, exist_ok=True)
    print(f"样本将保存到目录: '{output_path}'")
    
    total_num_samples = len(num_atoms)
    batch_size = min(batch_size, total_num_samples)
    
    sampled_structures = []
    for i in range(0, total_num_samples, batch_size):
        current_batch_size = min(batch_size, total_num_samples - i)
        print(f"生成批次 #{i // batch_size + 1}，包含 {current_batch_size} 个样本")
        
        batch = create_empty_batch(num_atoms[i:i+current_batch_size], device=device)
        
        with paddle.no_grad():
            gen_structures_batch = ldm.sample(
                batch,
                sampler='ddim',
                sampling_steps=sampling_steps,
                return_structure=True,  # Return pymatgen Structure objects directly
            )

        # gen_structures_batch is now a list of pymatgen Structure objects
        gen_structures = gen_structures_batch
        
        for j, st in enumerate(gen_structures):
            params = st.lattice.parameters
            a, b, c = params[:3]
            alpha, beta, gamma = params[3:]
            
            if a < 2.0 or b < 2.0 or c < 2.0:
                print(f"  警告: 样本 {i+j} 晶胞参数较小 (a={a:.3f}, b={b:.3f}, c={c:.3f})")
            
            if alpha > 150 or beta > 150 or gamma > 150:
                print(f"  跳过样本 {i+j}: 晶胞角度异常 (α={alpha:.1f}°, β={beta:.1f}°, γ={gamma:.1f}°)")
                continue
            
            if st.volume < 50.0:
                print(f"  跳过样本 {i+j}: 晶胞体积过小 ({st.volume:.3f} Å³)")
                continue
            
            formula = st.formula.replace(' ', '')
            output_file = output_path / f"sample_{i+j}_{formula}.cif"
            st.to(filename=str(output_file), fmt="cif")
            sampled_structures.append(st)
    
    print(f"\n生成完成！共生成 {len(sampled_structures)} 个晶体结构")
    print(f"CIF 文件已保存到: {output_path}")
    
    gen_st_files = sorted(output_path.glob("sample_*.cif"))
    all_structures = [Structure.from_file(str(f)) for f in gen_st_files]
    
    from monty.serialization import dumpfn
    json_output = output_path / "generated_structures.json.gz"
    dumpfn(all_structures, str(json_output))
    print(f"所有结构已保存为 JSON 格式: {json_output}")
    
    return sampled_structures


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chemeleon2 Structure Generation")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to generate")
    parser.add_argument("--batch_size", type=int, default=10, help="Batch size for generation")
    parser.add_argument("--sampling_steps", type=int, default=50, help="Number of sampling steps")
    parser.add_argument("--ldm_ckpt_path", type=str, default="test-for-weight/converted_weights/ldm_paddle.pdparams", help="Path to LDM checkpoint")
    parser.add_argument("--vae_ckpt_path", type=str, default="test-for-weight/converted_weights/vae_paddle.pdparams", help="Path to VAE checkpoint")
    parser.add_argument("--output_dir", type=str, default="outputs/samples", help="Output directory for generated structures")
    parser.add_argument("--device", type=str, default="gpu", help="Device to use (gpu or cpu)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    sample(
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        ldm_ckpt_path=args.ldm_ckpt_path,
        vae_ckpt_path=args.vae_ckpt_path,
        output_dir=args.output_dir,
        sampling_steps=args.sampling_steps,
        device=args.device,
        seed=args.seed,
    )
