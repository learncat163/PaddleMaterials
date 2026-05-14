"""
OMG Checkpoint Converter - PyTorch to PaddlePaddle

This script converts OMG model checkpoints from PyTorch to PaddlePaddle format.
Handles proper weight transposition for Linear layers.
"""

import argparse
import os
import paddle
import torch
import numpy as np


def convert_checkpoint(torch_ckpt_path, paddle_output_path, max_atoms=101, time_embed_dim=256):
    """Convert PyTorch checkpoint to PaddlePaddle format with proper weight handling."""
    print(f"Loading PyTorch checkpoint from: {torch_ckpt_path}")
    torch_ckpt = torch.load(torch_ckpt_path, map_location='cpu')
    
    if 'state_dict' in torch_ckpt:
        state_dict = torch_ckpt['state_dict']
    else:
        state_dict = torch_ckpt
    
    # Convert keys and transpose weights where needed
    paddle_state_dict = {}
    for key, value in state_dict.items():
        # Remove "model." prefix and "encoder." prefix
        new_key = key
        if new_key.startswith('model.'):
            new_key = new_key[6:]  # Remove 'model.'
        if new_key.startswith('encoder.'):
            new_key = new_key[8:]  # Remove 'encoder.'
        
        # Convert torch tensor to numpy, then to paddle tensor
        # For Linear layers in PyTorch: weight shape is [out_features, in_features]
        # For Linear layers in Paddle: weight shape is [out_features, in_features]
        # But when loading with set_state_dict, Paddle expects [in_features, out_features]
        # So we need to transpose the weights
        tensor = value.numpy()
        
        # Transpose weights for Linear layers (not for Embedding)
        if 'weight' in key and 'embedding' not in key:
            # This is a Linear layer weight - transpose it
            if len(tensor.shape) == 2:
                tensor = tensor.T
        
        paddle_state_dict[new_key] = paddle.to_tensor(tensor)
        
    print(f"Converted {len(paddle_state_dict)} parameters")
    
    # Show first 10 keys to verify
    print("First 10 converted keys:")
    for i, key in enumerate(list(paddle_state_dict.keys())[:10]):
        print(f"  {key}: {paddle_state_dict[key].shape}")
    
    # Save converted checkpoint
    paddle.save(paddle_state_dict, paddle_output_path)
    print(f"Saved PaddlePaddle checkpoint to: {paddle_output_path}")


def load_paddle_checkpoint_for_inference(paddle_ckpt_path, model):
    """Load PaddlePaddle checkpoint into model for inference."""
    paddle_state_dict = paddle.load(paddle_ckpt_path)
    model.set_state_dict(paddle_state_dict)
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert OMG PyTorch checkpoint to PaddlePaddle')
    parser.add_argument('--torch_ckpt', type=str, required=True,
                        help='Path to PyTorch checkpoint')
    parser.add_argument('--output', type=str, required=True,
                        help='Output path for PaddlePaddle checkpoint')
    parser.add_argument('--max_atoms', type=int, default=101,
                        help='Maximum number of atoms (default: 101)')
    parser.add_argument('--time_embed_dim', type=int, default=256,
                        help='Time embedding dimension (default: 256)')
    args = parser.parse_args()
    
    convert_checkpoint(args.torch_ckpt, args.output, args.max_atoms, args.time_embed_dim)