import numpy as np
from pathlib import Path


def load_and_print_first4(data_dir, output_file):
    data_dir = Path(data_dir)
    output_file = Path(output_file)

    test_files = [
        'vae_encoder_output_full.npz',
        'vae_encoder_quant_conv_full.npz',
        'vae_encoder_latent_mean_full.npz',
        'vae_encoder_latent_logvar_full.npz',
        'vae_decoder_atom_types_full.npz',
        'vae_decoder_frac_coords_full.npz',
        'vae_decoder_lengths_full.npz',
        'vae_decoder_angles_full.npz',
        'ldm_denoiser_output_noise_full.npz',
    ]

    results = []

    for filename in test_files:
        file_path = data_dir / filename
        if not file_path.exists():
            results.append(f"\n{'='*80}\n")
            results.append(f"File: {filename}\n")
            results.append(f"Status: NOT FOUND\n")
            continue

        data = np.load(file_path)
        key = list(data.keys())[0]
        arr = data[key]

        results.append(f"\n{'='*80}\n")
        results.append(f"File: {filename}\n")
        results.append(f"Key: {key}\n")
        results.append(f"Shape: {arr.shape}\n")
        results.append(f"Dtype: {arr.dtype}\n")
        results.append(f"Full array stats: min={arr.min():.6f}, max={arr.max():.6f}, mean={arr.mean():.6f}, std={arr.std():.6f}\n")
        results.append(f"\nFirst 4x4x4 elements:\n")

        def format_slice(data, dims):
            if len(dims) == 0:
                return f"{data:.8f}"
            result = "["
            for i in range(dims[0]):
                if i > 0:
                    result += ", "
                if len(dims) == 1:
                    result += format_slice(data[i], ())
                else:
                    result += format_slice(data[i], dims[1:])
            result += "]"
            return result

        slice_dims = tuple(min(4, d) for d in arr.shape)
        sliced_data = arr[tuple(slice(0, d) for d in slice_dims)]
        array_str = format_slice(sliced_data, slice_dims)

        results.append(f"\nFirst 4x4x4 elements (Python array format):\n")
        results.append(f"{array_str}\n")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(results)

    print(f"Results saved to: {output_file}")
    return output_file


if __name__ == '__main__':
    data_dir = 'chemeleon2/cyy_test/outputs/precision_test'
    output_file = 'test-validate-weight-first4/raw_npz_first4_output.txt'

    load_and_print_first4(data_dir, output_file)
