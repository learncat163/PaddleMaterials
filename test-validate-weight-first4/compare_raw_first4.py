import sys
import numpy as np
from pathlib import Path
import ast

parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))


def parse_reference_file(file_path):
    data = {}

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    blocks = content.split('================================================================================')

    for block in blocks:
        if not block.strip():
            continue

        lines = block.strip().split('\n')
        file_key = None
        array_str = None
        array_line_idx = -1

        for i, line in enumerate(lines):
            if line.startswith('File: '):
                filename = line.replace('File: ', '').strip()
                file_key = filename.replace('_full.npz', '').replace('.npz', '')
            elif 'Python array format:' in line:
                array_line_idx = i + 1

        if file_key and array_line_idx < len(lines):
            array_str = lines[array_line_idx].strip()
            if array_str and array_str.startswith('['):
                try:
                    ref_array = np.array(ast.literal_eval(array_str))
                    data[file_key] = ref_array
                except Exception as e:
                    print(f"Warning: Failed to parse {file_key}: {e}")

    return data


def load_paddle_slice(file_path, max_dims=(4, 4, 4, 4)):
    data = np.load(file_path)
    key = list(data.keys())[0]
    arr = data[key]
    slice_shape = tuple(min(d, s) for d, s in zip(max_dims, arr.shape))
    slices = tuple(slice(0, s) for s in slice_shape)
    return arr[slices]


def compare_results(reference_file, paddle_dir, threshold=1e-4):
    reference_data = parse_reference_file(reference_file)
    paddle_dir = Path(paddle_dir)

    test_files = [
        ('vae_encoder_output', 'VAE Encoder Output'),
        ('vae_encoder_quant_conv', 'VAE Encoder Quant Conv'),
        ('vae_encoder_latent_mean', 'VAE Latent Mean'),
        ('vae_encoder_latent_logvar', 'VAE Latent Logvar'),
        ('vae_decoder_atom_types', 'VAE Decoder Atom Types'),
        ('vae_decoder_frac_coords', 'VAE Decoder Frac Coords'),
        ('vae_decoder_lengths', 'VAE Decoder Lengths'),
        ('vae_decoder_angles', 'VAE Decoder Angles'),
        ('ldm_denoiser_output_noise', 'LDM Denoiser Output'),
    ]

    results = []
    all_passed = True

    print("="*80)
    print("Comparing Paddle vs Reference (First 4x4x4 elements, Threshold: 1e-4)")
    print("="*80)

    for file_key, name in test_files:
        print(f"\n{name}:")

        if file_key not in reference_data:
            print(f"  ⚠️  Reference data not found for {file_key}")
            continue

        ref_array = reference_data[file_key]
        npz_file = paddle_dir / f'{file_key}_full.npz'

        if not npz_file.exists():
            print(f"  ⚠️  Paddle file not found: {npz_file}")
            continue

        paddle_array = load_paddle_slice(npz_file)

        ref_flat = ref_array.flatten()
        paddle_flat = paddle_array.flatten()

        min_len = min(len(ref_flat), len(paddle_flat))
        ref_flat = ref_flat[:min_len]
        paddle_flat = paddle_flat[:min_len]

        diff = np.abs(ref_flat - paddle_flat)
        max_diff = diff.max()
        mean_diff = diff.mean()

        passed = max_diff < threshold

        if not passed:
            all_passed = False

        status = "✅" if passed else "❌"
        print(f"  {status} Max diff: {max_diff:.6e}, Mean diff: {mean_diff:.6e}")

        if not passed:
            max_idx = diff.argmax()
            print(f"     Ref value: {ref_flat[max_idx]:.8f}, Paddle value: {paddle_flat[max_idx]:.8f}")

        results.append({
            'name': name,
            'max_diff': float(max_diff),
            'mean_diff': float(mean_diff),
            'passed': bool(passed)
        })

    print("\n" + "="*80)
    if all_passed:
        print("✅ ALL TESTS PASSED (< 1e-4)")
    else:
        print("❌ SOME TESTS FAILED")
        for r in results:
            if not r['passed']:
                print(f"   - {r['name']}: Max diff = {r['max_diff']:.6e}")
    print("="*80)

    return all_passed, results


if __name__ == '__main__':
    reference_file = 'test-validate-weight-first4/raw_npz_first4_output.txt'
    paddle_dir = 'test-validate-weight/precision_test'

    all_passed, results = compare_results(reference_file, paddle_dir, threshold=1e-4)
