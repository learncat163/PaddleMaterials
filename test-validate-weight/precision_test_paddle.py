import sys
from pathlib import Path

parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))
sys.path.insert(0, str(parent_dir / 'chemeleon2'))

class MockModule:
    def __getattr__(self, name):
        return lambda *args, **kwargs: None

sys.modules['paddle_scatter'] = MockModule()

import paddle
import numpy as np
import json

from ppmat.models.chemeleon2.vae_module.vae import VAEModule
from ppmat.models.chemeleon2.vae_module.encoder import TransformerEncoder
from ppmat.models.chemeleon2.vae_module.decoder import TransformerDecoder
from ppmat.models.chemeleon2.ldm_module.dit import DiT
from ppmat.models.chemeleon2.common.schema import CrystalBatch

def load_npz_data(file_path):
    data = np.load(file_path)
    key = list(data.keys())[0]
    return data[key]

def save_npz_data(file_path, data, key='data'):
    np.savez_compressed(file_path, **{key: data})

def test_vae_encoder():
    print("\n" + "="*80)
    print("Testing VAE Encoder")
    print("="*80)

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

    weight_path = 'test-for-weight/converted_weights/vae_paddle.pdparams'
    print(f"Loading weights from: {weight_path}")
    full_state_dict = paddle.load(weight_path)
    vae.set_state_dict(full_state_dict)
    vae.eval()
    
    data_dir = Path('chemeleon2/cyy_test/outputs/precision_test')
    atom_types = load_npz_data(data_dir / 'vae_encoder_input_atom_types_full.npz')
    frac_coords = load_npz_data(data_dir / 'vae_encoder_input_frac_coords_full.npz')
    lattices = load_npz_data(data_dir / 'vae_encoder_input_lattices_full.npz')
    num_atoms = load_npz_data(data_dir / 'vae_encoder_input_num_atoms_full.npz')
    lengths = load_npz_data(data_dir / 'vae_encoder_input_lengths_full.npz')
    angles = load_npz_data(data_dir / 'vae_encoder_input_angles_full.npz')
    
    batch = CrystalBatch()
    batch.atom_types = paddle.to_tensor(atom_types, dtype='int64')
    batch.frac_coords = paddle.to_tensor(frac_coords, dtype='float32')
    batch.lattices = paddle.to_tensor(lattices, dtype='float32')
    batch.num_atoms = paddle.to_tensor(num_atoms, dtype='int64')
    batch.batch = paddle.zeros([12], dtype='int64')
    batch.token_idx = paddle.arange(12, dtype='int64')
    batch.num_graphs = 1
    
    with paddle.no_grad():
        encoded = vae.encode(batch)
        encoder_output = encoded['x']
        quant_conv_output = encoded['moments']
        latent_mean = encoded['posterior'].mean
        latent_logvar = encoded['posterior'].logvar
        latent_z = encoded['posterior'].sample()
    
    output_dir = Path('test-validate-weight/precision_test')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    save_npz_data(output_dir / 'vae_encoder_output_full.npz', encoder_output.numpy(), 'data')
    save_npz_data(output_dir / 'vae_encoder_quant_conv_full.npz', quant_conv_output.numpy(), 'data')
    save_npz_data(output_dir / 'vae_encoder_latent_mean_full.npz', latent_mean.numpy(), 'data')
    save_npz_data(output_dir / 'vae_encoder_latent_logvar_full.npz', latent_logvar.numpy(), 'data')
    save_npz_data(output_dir / 'vae_encoder_latent_z_full.npz', latent_z.numpy(), 'data')
    
    print(f"encoder_output: {encoder_output.shape}, mean={encoder_output.mean().item():.6f}, std={encoder_output.std().item():.6f}")
    print(f"latent_mean: {latent_mean.shape}, mean={latent_mean.mean().item():.6f}, std={latent_mean.std().item():.6f}")
    print(f"latent_logvar: {latent_logvar.shape}, mean={latent_logvar.mean().item():.6f}, std={latent_logvar.std().item():.6f}")
    print(f"latent_z: {latent_z.shape}, mean={latent_z.mean().item():.6f}, std={latent_z.std().item():.6f}")
    
    return {
        'encoder_output': encoder_output,
        'latent_mean': latent_mean,
        'latent_logvar': latent_logvar,
        'latent_z': latent_z 
    }

def test_vae_decoder():
    print("\n" + "="*80)
    print("Testing VAE Decoder")
    print("="*80)

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

    weight_path = 'test-for-weight/converted_weights/vae_paddle.pdparams'
    print(f"Loading weights from: {weight_path}")
    full_state_dict = paddle.load(weight_path)
    vae.set_state_dict(full_state_dict)
    vae.eval()
    
    data_dir = Path('chemeleon2/cyy_test/outputs/precision_test')
    z = load_npz_data(data_dir / 'vae_decoder_input_z_full.npz')
    num_atoms = load_npz_data(data_dir / 'vae_encoder_input_num_atoms_full.npz')
    
    z_t = paddle.to_tensor(z, dtype='float32')
    batch = CrystalBatch()
    batch.atom_types = paddle.zeros([12], dtype='int64')
    batch.frac_coords = paddle.zeros([12, 3], dtype='float32')
    batch.lattices = paddle.zeros([1, 3, 3], dtype='float32')
    batch.num_atoms = paddle.to_tensor(num_atoms, dtype='int64')
    batch.batch = paddle.zeros([12], dtype='int64')
    batch.token_idx = paddle.arange(12, dtype='int64')
    batch.num_graphs = 1
    
    with paddle.no_grad():
        encoded = {
            'x': z_t,
            'z': z_t,
            'batch': batch.batch,
            'token_idx': batch.token_idx,
            'num_atoms': batch.num_atoms
        }
        decoded = vae.decode(encoded)
        post_quant = vae.post_quant_conv(z_t)
        atom_types = decoded['atom_types']
        frac_coords = decoded['frac_coords']
        lengths = decoded['lengths']
        angles = decoded['angles']
    
    output_dir = Path('test-validate-weight/precision_test')
    save_npz_data(output_dir / 'vae_decoder_post_quant_full.npz', post_quant.numpy(), 'data')
    save_npz_data(output_dir / 'vae_decoder_atom_types_full.npz', atom_types.numpy(), 'data')
    save_npz_data(output_dir / 'vae_decoder_frac_coords_full.npz', frac_coords.numpy(), 'data')
    save_npz_data(output_dir / 'vae_decoder_lengths_full.npz', lengths.numpy(), 'data')
    save_npz_data(output_dir / 'vae_decoder_angles_full.npz', angles.numpy(), 'data')
    
    print(f"post_quant: {post_quant.shape}, mean={post_quant.mean().item():.6f}, std={post_quant.std().item():.6f}")
    print(f"atom_types: {atom_types.shape}, mean={atom_types.mean().item():.6f}, std={atom_types.std().item():.6f}")
    print(f"frac_coords: {frac_coords.shape}, mean={frac_coords.mean().item():.6f}, std={frac_coords.std().item():.6f}")
    print(f"lengths: {lengths.shape}, mean={lengths.mean().item():.6f}")
    print(f"angles: {angles.shape}, mean={angles.mean().item():.6f}")
    
    return {
        'post_quant': post_quant,
        'atom_types': atom_types,
        'frac_coords': frac_coords,
        'lengths': lengths,
        'angles': angles
    }

def test_ldm_denoiser():
    print("\n" + "="*80)
    print("Testing LDM Denoiser")
    print("="*80)
    
    denoiser = DiT(
        hidden_dim=768,
        num_layers=12,
        num_heads=12,
        mlp_ratio=4.0,
        input_dim=8,
        learn_sigma=True
    )
    
    weight_path = 'test-for-weight/converted_weights/ldm_paddle.pdparams'
    print(f"Loading weights from: {weight_path}")
    full_state_dict = paddle.load(weight_path)
    
    denoiser_state_dict = {}
    for key, value in full_state_dict.items():
        if key.startswith('denoiser.'):
            new_key = key.replace('denoiser.', '')
            denoiser_state_dict[new_key] = value
    
    denoiser.set_state_dict(denoiser_state_dict)
    denoiser.eval()
    
    data_dir = Path('chemeleon2/cyy_test/outputs/precision_test')
    z = load_npz_data(data_dir / 'ldm_denoiser_input_z_full.npz')

    z_t = paddle.to_tensor(z, dtype='float32')
    t = paddle.to_tensor([13, 25], dtype='int64')
    batch_size, num_atoms, _ = z.shape
    mask = paddle.ones([batch_size, num_atoms], dtype='bool')

    with paddle.no_grad():
        noise_pred = denoiser(z_t, t, mask=mask)
    
    output_dir = Path('test-validate-weight/precision_test')
    save_npz_data(output_dir / 'ldm_denoiser_output_noise_full.npz', noise_pred.numpy(), 'data')
    
    print(f"noise_pred: {noise_pred.shape}, mean={noise_pred.mean().item():.6f}, std={noise_pred.std().item():.6f}")
    print(f"min={noise_pred.min().item():.6f}, max={noise_pred.max().item():.6f}")
    
    return {'noise_pred': noise_pred}

def compare_results():
    print("\n" + "="*80)
    print("Comparing Paddle vs PyTorch Results (Threshold: 1e-4)")
    print("="*80)
    
    pytorch_dir = Path('chemeleon2/cyy_test/outputs/precision_test')
    paddle_dir = Path('test-validate-weight/precision_test')
    
    test_cases = [
        ('vae_encoder_output_full.npz', 'VAE Encoder Output'),
        ('vae_encoder_quant_conv_full.npz', 'VAE Encoder Quant Conv'),
        ('vae_encoder_latent_mean_full.npz', 'VAE Latent Mean'),
        ('vae_encoder_latent_logvar_full.npz', 'VAE Latent Logvar'),
        ('vae_full_latent_mean_full.npz', 'VAE Full Latent Mean'),
        ('vae_decoder_atom_types_full.npz', 'VAE Decoder Atom Types'),
        ('vae_decoder_frac_coords_full.npz', 'VAE Decoder Frac Coords'),
        ('vae_decoder_lengths_full.npz', 'VAE Decoder Lengths'),
        ('vae_decoder_angles_full.npz', 'VAE Decoder Angles'),
        ('ldm_denoiser_output_noise_full.npz', 'LDM Denoiser Output'),
    ]
    
    results = {}
    all_passed = True
    failed_tests = []
    
    for filename, name in test_cases:
        pytorch_file = pytorch_dir / filename
        paddle_file = paddle_dir / filename
        
        if not pytorch_file.exists():
            print(f"⚠️  {name}: PyTorch file not found: {pytorch_file}")
            continue
            
        if not paddle_file.exists():
            print(f"⚠️  {name}: Paddle file not found: {paddle_file}")
            continue
        
        pytorch_data = load_npz_data(pytorch_file)
        paddle_data = load_npz_data(paddle_file)
        
        if pytorch_data.shape != paddle_data.shape:
            print(f"❌ {name}: Shape mismatch! PyTorch: {pytorch_data.shape}, Paddle: {paddle_data.shape}")
            all_passed = False
            failed_tests.append(name)
            continue
        
        diff = np.abs(pytorch_data - paddle_data)
        max_diff = diff.max()
        mean_diff = diff.mean()
        relative_diff = np.abs((pytorch_data - paddle_data) / (np.abs(pytorch_data) + 1e-8)).mean()
        
        passed = max_diff < 1e-4
        
        if not passed:
            all_passed = False
            failed_tests.append(name)
        
        status = "✅" if passed else "❌"
        print(f"{status} {name}:")
        print(f"   Max diff: {max_diff:.6e}, Mean diff: {mean_diff:.6e}, Relative: {relative_diff:.6e}")
        
        if not passed:
            max_idx = np.unravel_index(diff.argmax(), diff.shape)
            print(f"   Max diff location: {max_idx}")
            print(f"   PyTorch value: {pytorch_data[max_idx]:.6f}, Paddle value: {paddle_data[max_idx]:.6f}")
        
        results[name] = {
            'max_diff': float(max_diff),
            'mean_diff': float(mean_diff),
            'relative_diff': float(relative_diff),
            'passed': bool(passed)
        }
    
    output_file = paddle_dir / 'comparison_results.json'
    with open(output_file, 'w') as f:
        json.dump({
            'all_passed': all_passed,
            'threshold': 1e-4,
            'failed_tests': failed_tests,
            'results': results
        }, f, indent=2)
    
    print(f"\n{'='*80}")
    if all_passed:
        print(f"✅ ALL TESTS PASSED (< 1e-4)")
    else:
        print(f"❌ FAILED TESTS ({len(failed_tests)}):")
        for test in failed_tests:
            print(f"   - {test}")
    print(f"Results saved to: {output_file}")
    
    return all_passed

if __name__ == '__main__':
    paddle.seed(42)
    np.random.seed(42)
    
    print("Paddle Precision Test")
    print("="*80)
    
    test_vae_encoder()
    test_vae_decoder()
    test_ldm_denoiser()
    compare_results()
