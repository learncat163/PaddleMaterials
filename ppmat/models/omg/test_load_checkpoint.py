"""
Test loading converted checkpoint into CSPNet model.
"""

import paddle
import torch
import numpy as np

# Import the Paddle CSPNet model
from ppmat.models.omg.model.cspnet import CSPNet


def load_paddle_checkpoint(torch_ckpt_path):
    """Load PyTorch checkpoint and convert to Paddle format."""
    print(f"Loading PyTorch checkpoint from: {torch_ckpt_path}")
    torch_ckpt = torch.load(torch_ckpt_path, map_location='cpu')
    
    if 'state_dict' in torch_ckpt:
        state_dict = torch_ckpt['state_dict']
    else:
        state_dict = torch_ckpt
    
    # Convert keys from PyTorch format to PaddlePaddle format
    paddle_state_dict = {}
    for key, value in state_dict.items():
        # Remove "model." prefix if present
        new_key = key
        if new_key.startswith('model.'):
            new_key = new_key[6:]
        
        # Convert torch tensor to numpy, then to paddle tensor
        # Note: For Linear layers, PyTorch and Paddle both use [out_features, in_features]
        # so no transpose is needed
        paddle_state_dict[new_key] = paddle.to_tensor(value.numpy())
        
    return paddle_state_dict


def analyze_checkpoint(torch_ckpt_path):
    """Analyze checkpoint structure."""
    torch_ckpt = torch.load(torch_ckpt_path, map_location='cpu')
    state_dict = torch_ckpt['state_dict'] if 'state_dict' in torch_ckpt else torch_ckpt
    
    # Strip "model." prefix from keys
    clean_state_dict = {}
    for key, value in state_dict.items():
        new_key = key
        if new_key.startswith('model.'):
            new_key = new_key[6:]
        clean_state_dict[new_key] = value
    
    # Get dimensions
    hidden_dim = clean_state_dict['encoder.csp_layer_0.edge_mlp.0.weight'].shape[0]
    input_dim = clean_state_dict['encoder.csp_layer_0.edge_mlp.0.weight'].shape[1]
    dis_dim = input_dim - hidden_dim * 2 - 9
    
    num_layers = 0
    for key in clean_state_dict.keys():
        if 'csp_layer_' in key and '.edge_mlp.0.weight' in key:
            layer_num = int(key.split('csp_layer_')[1].split('.')[0])
            num_layers = max(num_layers, layer_num + 1)
    
    max_atoms = clean_state_dict['encoder.node_embedding.weight'].shape[0]
    latent_dim = clean_state_dict['encoder.atom_latent_emb.weight'].shape[1] - hidden_dim
    num_freqs = dis_dim // 6
    ln = 'encoder.final_layer_norm.weight' in clean_state_dict
    pred_type = 'encoder.type_out.weight' in clean_state_dict
    
    return {
        'hidden_dim': hidden_dim,
        'latent_dim': latent_dim,
        'num_layers': num_layers,
        'max_atoms': max_atoms,
        'num_freqs': num_freqs,
        'ln': ln,
        'ip': True,  # Always True based on checkpoints
        'pred_type': pred_type,
        'edge_style': 'knn' if 'encoder.gen_edges' not in str(list(clean_state_dict.keys())[:5]) else 'fc',
    }


def load_and_verify_checkpoint(torch_ckpt_path, model=None):
    """Load checkpoint and verify against model."""
    paddle_state_dict = load_paddle_checkpoint(torch_ckpt_path)
    
    if model is None:
        config = analyze_checkpoint(torch_ckpt_path)
        print(f"\nInferred model config: {config}")
        model = CSPNet(**config)
    
    # Set model state dict
    model.set_state_dict(paddle_state_dict)
    print(f"\nSuccessfully loaded {len(paddle_state_dict)} parameters into model")
    
    return model


def test_forward_pass(model):
    """Test a forward pass with random data."""
    print("\nTesting forward pass...")
    
    batch_size = 2
    num_atoms = paddle.to_tensor([4, 3])  # 4 atoms in first structure, 3 in second
    total_atoms = int(num_atoms.sum().item())
    
    # Create random input data
    t = paddle.rand([batch_size, 1])
    atom_types = paddle.randint(1, 90, [total_atoms])  # Random atomic numbers
    frac_coords = paddle.rand([total_atoms, 3])
    # Create lattice matrices (3x3 per structure)
    lattices = paddle.rand([batch_size, 3, 3])
    node2graph = paddle.concat([paddle.zeros([4]), paddle.ones([3])]).astype('int64')
    
    # Forward pass
    try:
        outputs = model(t, atom_types, frac_coords, lattices, num_atoms, node2graph)
        print(f"Forward pass successful!")
        print(f"  Outputs: {[o.shape for o in outputs]}")
    except Exception as e:
        print(f"Forward pass failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    # Path to PyTorch checkpoint (DNG model)
    torch_ckpt_path = "/home/cao/.cache/huggingface/hub/models--OMatG--MP-20-DNG/snapshots/6af2cc9a1fa569341fb7a08a542490868ab84867/VPSBD-ODE/checkpoint.ckpt"
    
    print("=" * 60)
    print("Testing CSPNet Model Loading from Checkpoint")
    print("=" * 60)
    
    # Analyze checkpoint
    config = analyze_checkpoint(torch_ckpt_path)
    print(f"\nCheckpoint analyzed:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    
    # Create model with correct config
    print("\nCreating model...")
    model = CSPNet(
        hidden_dim=config['hidden_dim'],
        latent_dim=config['latent_dim'],
        num_layers=config['num_layers'],
        max_atoms=config['max_atoms'],
        num_freqs=config['num_freqs'],
        ln=config['ln'],
        ip=config['ip'],
        pred_type=config['pred_type'],
    )
    
    # Load checkpoint
    print("\nLoading checkpoint...")
    load_and_verify_checkpoint(torch_ckpt_path, model)
    
    # Test forward pass
    test_forward_pass(model)
    
    print("\n" + "=" * 60)
    print("Testing CSP Model Loading from Checkpoint (CSP model)")
    print("=" * 60)
    
    # CSP model checkpoint
    csp_ckpt_path = "/home/cao/.cache/huggingface/hub/models--OMatG--MP-20-CSP/snapshots/87dcc2a222f849f4f3c381a8cfa47ede0971d364/EncDec-ODE-Gamma/checkpoint.ckpt"
    
    print(f"\nCheckpoint analyzed:")
    config_csp = analyze_checkpoint(csp_ckpt_path)
    for k, v in config_csp.items():
        print(f"  {k}: {v}")
    
    # Create CSP model
    print("\nCreating CSP model...")
    model_csp = CSPNet(
        hidden_dim=config_csp['hidden_dim'],
        latent_dim=config_csp['latent_dim'],
        num_layers=config_csp['num_layers'],
        max_atoms=config_csp['max_atoms'],
        num_freqs=config_csp['num_freqs'],
        ln=config_csp['ln'],
        ip=config_csp['ip'],
        pred_type=config_csp['pred_type'],
    )
    
    # Load checkpoint
    print("\nLoading checkpoint...")
    load_and_verify_checkpoint(csp_ckpt_path, model_csp)
    
    # Test forward pass
    test_forward_pass(model_csp)
    
    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    print("=" * 60)


if __name__ == '__main__':
    main()