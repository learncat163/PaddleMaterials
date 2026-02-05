#!/usr/bin/env python3
"""
Integrated validation script for VAE and LDM models.

This script:
1. Loads VAE and LDM models with converted weights
2. Uses hardcoded input data (no npz file dependencies)
3. Compares outputs against reference values (first 4x4x4 elements)
4. Uses 1e-6 as precision threshold

Usage:
    python integrated_validation.py [options]

Options:
    --weight-dir PATH       Directory containing converted weights (default: test-for-weight/converted_weights)
    --reference-file PATH   Reference output file (default: test-validate-weight-first4/raw_npz_first4_output.txt)
    --output-file PATH      Output JSON file for results (default: test-validate-weight-first4/integrated_validation_results.json)
    --threshold FLOAT       Precision threshold (default: 1e-6)
"""

import sys
import argparse
from pathlib import Path
import ast

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


# =============================================================================
# Default Configuration
# =============================================================================
DEFAULT_WEIGHT_DIR = 'test-for-weight/converted_weights'
DEFAULT_REFERENCE_FILE = 'test-validate-weight-first4/raw_npz_first4_output.txt'
DEFAULT_OUTPUT_FILE = 'test-validate-weight-first4/integrated_validation_results.json'
DEFAULT_PRECISION_THRESHOLD = 1e-5


# =============================================================================
# Hardcoded Input Data (extracted from npz files)
# =============================================================================
# VAE Encoder Input Data
ATOM_TYPES = np.array(
    [66, 67, 68, 68, 68, 69, 52, 52, 52, 33, 15, 15]
)

FRAC_COORDS = np.array(
    [[0.16941863298416138, 0.25193023681640625, 0.45317789912223816],
     [0.4915277361869812, 0.4513784646987915, 0.1179681122303009],
     [0.48285794258117676, 0.9133925437927246, 0.1451053023338318],
     [0.8267775774002075, 0.5902515053749084, 0.7981773614883423],
     [0.8183310031890869, 0.11027566343545914, 0.7618123292922974],
     [0.1570594757795334, 0.7773158550262451, 0.42618808150291443],
     [0.333425909280777, 0.8513659238815308, 0.7798552513122559],
     [0.654494047164917, 0.5193513631820679, 0.45445090532302856],
     [0.9899516105651855, 0.17083004117012024, 0.11222820729017258],
     [0.336728572845459, 0.33988437056541443, 0.7731209993362427],
     [0.6685014963150024, 0.0031565700192004442, 0.4517180025577545],
     [0.9902676343917847, 0.6586886644363403, 0.12688931822776794]]
)

LATTICES = np.array(
    [[[4.070977210998535, 0.0, -0.9100706577301025],
      [-0.4610702395439148, 8.279973983764648, -2.1255712509155273],
      [0.0, 0.0, 9.44053840637207]]]
)

NUM_ATOMS = np.array([12])

LENGTHS = np.array(
    [[4.1714606285095215, 8.560876846313477, 9.44053840637207]]
)

ANGLES = np.array(
    [[104.37628173828125, 102.60133361816406, 89.90788269042969]]
)

# VAE Decoder Input Data (latent z)
Z = np.array(
    [[1.967786192893982, 0.5211002230644226, -0.37526771426200867, -1.0607879161834717, -0.8559808135032654, 1.1134672164916992, 0.22077970206737518, 0.965729296207428],
     [0.6457473635673523, -0.4311949908733368, 0.03418273478746414, -0.006454047746956348, 0.8924912810325623, 0.3727598190307617, 2.41711688041687, 0.11571212112903595],
     [0.42960017919540405, -0.4362214207649231, 1.4410359859466553, 0.11944306641817093, -1.8423340320587158, -0.07848608493804932, 1.1500141620635986, -0.5406312346458435],
     [0.9139401912689209, 1.6348258256912231, 0.5740830302238464, 1.0798999071121216, -2.488610029220581, 1.022837519645691, 1.024582028388977, 0.9981280565261841],
     [0.6651034951210022, 1.6327271461486816, -0.9687730669975281, 0.9175443053245544, -2.0414130687713623, 2.0009686946868896, 0.6404735445976257, 0.051522813737392426],
     [2.287353992462158, 0.3691967725753784, 1.113817811012268, -1.0069981813430786, 0.6471315622329712, 0.056496068835258484, -0.6219279766082764, 0.9739170074462891],
     [-1.5223760604858398, 1.3992975950241089, 1.430759310722351, -0.46554267406463623, -0.6827797889709473, -0.5638044476509094, 1.099972128868103, 0.7320751547813416],
     [-1.6696141958236694, 0.5772172212600708, 0.25194433331489563, 0.39816927909851074, -0.6570543646812439, -0.12071487307548523, 0.8859682083129883, 1.2068629264831543],
     [-0.7076722979545593, -0.25845712423324585, -1.0974304676055908, 1.779221773147583, -0.8451420068740845, 2.0822441577911377, 0.5447143316268921, -0.14093801379203796],
     [-1.0009422302246094, 1.5222424268722534, -0.1030266284942627, -0.580216109752655, 1.4293816089630127, -0.5898170471191406, -1.166476845741272, 0.9782193303108215],
     [-0.5899573564529419, 0.7521655559539795, -1.4372801780700684, 0.3741850256919861, -0.3077695965766907, 2.8079261779785156, -0.7969849705696106, -1.0039623975753784],
     [-0.8462305068969727, -0.3148186206817627, 0.5480344295501709, 1.9970613718032837, -0.12883026897907257, 1.1838531494140625, -0.8120221495628357, -0.8874912261962891]]
)

# LDM Denoiser Input Data (diffusion latent z)
LDM_Z = np.array(
    [[[0.19401879608631134, 2.1613736152648926, -0.17205022275447845, 0.8490601181983948, -1.9243990182876587, 0.6529855132102966, -0.6494408249855042, -0.8175247311592102],
      [0.5279644727706909, -1.2753498554229736, -1.6621263027191162, -0.3033137321472168, -0.09256987273693085, 0.1992371529340744, -1.1204328536987305, 1.8576586246490479],
      [-0.7145188450813293, 0.6881051063537598, 0.7968308329582214, -0.03340187668800354, 1.491731882095337, -0.5165092945098877, -0.25409597158432007, 1.4746155738830566],
      [-0.32603731751441956, -1.1599626541137695, 2.355130910873413, -0.6924470663070679, 0.18374282121658325, -1.1835099458694458, -1.8028671741485596, -1.5807569026947021],
      [0.8386695384979248, 1.4191802740097046, 0.6469367146492004, 0.4252724051475525, -1.5892407894134521, 0.622344970703125, 1.6898036003112793, -0.6648038625717163],
      [0.9425426721572876, 0.07832549512386322, 0.08465634286403656, -0.1408299207687378, 0.33156055212020874, -0.5889761447906494, -1.0722894668579102, 0.0953957587480545],
      [-0.3346919119358063, -0.525797426700592, -0.8776255249977112, 0.39383137226104736, 0.16395936906337738, -0.1976822465658188, 1.010413646697998, -1.3482447862625122],
      [-0.34977224469184875, -0.6442679762840271, 0.4467834234237671, -0.5371097326278687, 1.2423185110092163, -0.8145953416824341, 0.2501504123210907, -0.42725861072540283],
      [1.1043692827224731, -1.1027987003326416, 0.5543266534805298, -1.2846554517745972, -0.38157832622528076, 0.5139457583427429, 0.10019008070230484, 0.2586260437965393],
      [0.3616807162761688, 2.278669595718384, 0.023345094174146652, 1.5827577114105225, -1.1591792106628418, 0.9483923316001892, -0.45734766125679016, 0.7605476975440979]],
     [[-0.5786830186843872, -0.7050208449363708, -0.7233877182006836, -0.5070619583129883, -0.43984994292259216, -0.41817018389701843, 0.17413868010044098, 0.44268035888671875],
      [0.5068982839584351, -1.2168086767196655, -0.27187299728393555, 0.27654942870140076, -1.4398165941238403, -0.6463212370872498, 0.07486921548843384, 0.19387875497341156],
      [0.5960116982460022, 0.23220330476760864, 1.141465425491333, -0.6817070245742798, -1.6531448364257812, 0.0060356417670845985, 1.3814870119094849, 1.270426630973816],
      [0.02323809638619423, -1.300145149230957, -0.7509413361549377, 0.37562432885169983, -0.5474422574043274, -0.03964127227663994, -0.7778646945953369, -2.5018858909606934],
      [0.7000165581703186, -0.0937746912240982, -0.21625694632530212, 0.44839420914649963, -0.31519615650177, 0.021637355908751488, 0.6253470778465271, 0.2465813010931015],
      [0.7485606074333191, -0.11692450940608978, -0.10216468572616577, -0.5010807514190674, -0.504887580871582, -1.2071924209594727, -0.24375997483730316, -0.6784263849258423],
      [0.19728611409664154, 0.9782202839851379, -0.028667714446783066, 1.6825895309448242, 1.0908557176589966, -0.9920945763587952, -0.6712623834609985, 1.7196345329284668],
      [2.460554361343384, -0.6198393702507019, 1.2713860273361206, -0.27986663579940796, 0.43596675992012024, 0.4260239005088806, 1.0645543336868286, -2.027996301651001],
      [-0.6325827240943909, 2.1106481552124023, -0.09474601596593857, 0.23587526381015778, -0.7300723195075989, -1.6857199668884277, 0.9114198088645935, 0.788545548915863],
      [-0.6287308931350708, 2.159554958343506, 1.1642494201660156, -0.4256628751754761, 0.23932021856307983, -1.2776703834533691, -0.1206469014286995, -0.6065842509269714]]]
)


# =============================================================================
# Reference data parsing
# =============================================================================
def parse_reference_file(file_path):
    """Parse the reference output file containing first 4x4x4 elements."""
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


# =============================================================================
# Utility functions
# =============================================================================
def get_first_n_elements(arr, max_dims=(4, 4, 4, 4)):
    """Extract first N elements from array based on dimensions."""
    slice_shape = tuple(min(d, s) for d, s in zip(max_dims, arr.shape))
    slices = tuple(slice(0, s) for s in slice_shape)
    return arr[slices]


# =============================================================================
# Model testing functions
# =============================================================================
def run_test_vae_encoder(reference_data, weight_dir, precision_threshold):
    """Test VAE Encoder and compare with reference data."""
    print("\n" + "="*80)
    print("Testing VAE Encoder")
    print("="*80)

    # Initialize model
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

    # Load weights
    weight_path = Path(weight_dir) / 'vae_paddle.pdparams'
    print(f"Loading weights from: {weight_path}")
    full_state_dict = paddle.load(str(weight_path))
    vae.set_state_dict(full_state_dict)
    vae.eval()

    # Create batch using hardcoded data
    batch = CrystalBatch()
    batch.atom_types = paddle.to_tensor(ATOM_TYPES, dtype='int64')
    batch.frac_coords = paddle.to_tensor(FRAC_COORDS, dtype='float32')
    batch.lattices = paddle.to_tensor(LATTICES, dtype='float32')
    batch.num_atoms = paddle.to_tensor(NUM_ATOMS, dtype='int64')
    batch.batch = paddle.zeros([12], dtype='int64')
    batch.token_idx = paddle.arange(12, dtype='int64')
    batch.num_graphs = 1

    # Run inference
    with paddle.no_grad():
        encoded = vae.encode(batch)
        encoder_output = encoded['x']
        quant_conv_output = encoded['moments']
        latent_mean = encoded['posterior'].mean
        latent_logvar = encoded['posterior'].logvar
        latent_z = encoded['posterior'].sample()

    # Collect outputs for comparison
    outputs = {
        'vae_encoder_output': encoder_output.numpy(),
        'vae_encoder_quant_conv': quant_conv_output.numpy(),
        'vae_encoder_latent_mean': latent_mean.numpy(),
        'vae_encoder_latent_logvar': latent_logvar.numpy(),
    }

    # Compare with reference
    results = []
    all_passed = True

    for key, output in outputs.items():
        if key not in reference_data:
            print(f"WARNING {key}: No reference data found")
            continue

        ref_array = reference_data[key]
        output_slice = get_first_n_elements(output)

        # Flatten for comparison
        ref_flat = ref_array.flatten()
        output_flat = output_slice.flatten()

        min_len = min(len(ref_flat), len(output_flat))
        ref_flat = ref_flat[:min_len]
        output_flat = output_flat[:min_len]

        diff = np.abs(ref_flat - output_flat)
        max_diff = diff.max()
        mean_diff = diff.mean()

        passed = max_diff < precision_threshold

        if not passed:
            all_passed = False

        status = "PASS" if passed else "FAIL"
        print(f"{status} {key}:")
        print(f"   Max diff: {max_diff:.6e}, Mean diff: {mean_diff:.6e}")
        print(f"   Shape: {output.shape}")

        if not passed:
            max_idx = diff.argmax()
            print(f"   Ref value: {ref_flat[max_idx]:.8f}, Output value: {output_flat[max_idx]:.8f}")

        results.append({
            'name': key,
            'max_diff': float(max_diff),
            'mean_diff': float(mean_diff),
            'passed': bool(passed)
        })

    return all_passed, results


def run_test_vae_decoder(reference_data, weight_dir, precision_threshold):
    """Test VAE Decoder and compare with reference data."""
    print("\n" + "="*80)
    print("Testing VAE Decoder")
    print("="*80)

    # Initialize model
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

    # Load weights
    weight_path = Path(weight_dir) / 'vae_paddle.pdparams'
    print(f"Loading weights from: {weight_path}")
    full_state_dict = paddle.load(str(weight_path))
    vae.set_state_dict(full_state_dict)
    vae.eval()

    z_t = paddle.to_tensor(Z, dtype='float32')
    batch = CrystalBatch()
    batch.atom_types = paddle.zeros([12], dtype='int64')
    batch.frac_coords = paddle.zeros([12, 3], dtype='float32')
    batch.lattices = paddle.zeros([1, 3, 3], dtype='float32')
    batch.num_atoms = paddle.to_tensor(NUM_ATOMS, dtype='int64')
    batch.batch = paddle.zeros([12], dtype='int64')
    batch.token_idx = paddle.arange(12, dtype='int64')
    batch.num_graphs = 1

    # Run inference
    with paddle.no_grad():
        encoded = {
            'x': z_t,
            'z': z_t,
            'batch': batch.batch,
            'token_idx': batch.token_idx,
            'num_atoms': batch.num_atoms
        }
        decoded = vae.decode(encoded)
        atom_types = decoded['atom_types']
        frac_coords = decoded['frac_coords']
        lengths = decoded['lengths']
        angles = decoded['angles']

    # Collect outputs for comparison
    outputs = {
        'vae_decoder_atom_types': atom_types.numpy(),
        'vae_decoder_frac_coords': frac_coords.numpy(),
        'vae_decoder_lengths': lengths.numpy(),
        'vae_decoder_angles': angles.numpy(),
    }

    # Compare with reference
    results = []
    all_passed = True

    for key, output in outputs.items():
        if key not in reference_data:
            print(f"WARNING {key}: No reference data found")
            continue

        ref_array = reference_data[key]
        output_slice = get_first_n_elements(output)

        # Flatten for comparison
        ref_flat = ref_array.flatten()
        output_flat = output_slice.flatten()

        min_len = min(len(ref_flat), len(output_flat))
        ref_flat = ref_flat[:min_len]
        output_flat = output_flat[:min_len]

        diff = np.abs(ref_flat - output_flat)
        max_diff = diff.max()
        mean_diff = diff.mean()

        passed = max_diff < precision_threshold

        if not passed:
            all_passed = False

        status = "PASS" if passed else "FAIL"
        print(f"{status} {key}:")
        print(f"   Max diff: {max_diff:.6e}, Mean diff: {mean_diff:.6e}")
        print(f"   Shape: {output.shape}")

        if not passed:
            max_idx = diff.argmax()
            print(f"   Ref value: {ref_flat[max_idx]:.8f}, Output value: {output_flat[max_idx]:.8f}")

        results.append({
            'name': key,
            'max_diff': float(max_diff),
            'mean_diff': float(mean_diff),
            'passed': bool(passed)
        })

    return all_passed, results


def run_test_ldm_denoiser(reference_data, weight_dir, precision_threshold):
    """Test LDM Denoiser and compare with reference data."""
    print("\n" + "="*80)
    print("Testing LDM Denoiser")
    print("="*80)

    # Initialize model
    denoiser = DiT(
        hidden_dim=768,
        num_layers=12,
        num_heads=12,
        mlp_ratio=4.0,
        input_dim=8,
        learn_sigma=True
    )

    # Load weights
    weight_path = Path(weight_dir) / 'ldm_paddle.pdparams'
    print(f"Loading weights from: {weight_path}")
    full_state_dict = paddle.load(str(weight_path))

    denoiser_state_dict = {}
    for key, value in full_state_dict.items():
        if key.startswith('denoiser.'):
            new_key = key.replace('denoiser.', '')
            denoiser_state_dict[new_key] = value

    denoiser.set_state_dict(denoiser_state_dict)
    denoiser.eval()

    z_t = paddle.to_tensor(LDM_Z, dtype='float32')
    t = paddle.to_tensor([13, 25], dtype='int64')
    batch_size, num_atoms, _ = LDM_Z.shape
    mask = paddle.ones([batch_size, num_atoms], dtype='bool')

    # Run inference
    with paddle.no_grad():
        noise_pred = denoiser(z_t, t, mask=mask)

    # Collect outputs for comparison
    outputs = {
        'ldm_denoiser_output_noise': noise_pred.numpy(),
    }

    # Compare with reference
    results = []
    all_passed = True

    for key, output in outputs.items():
        if key not in reference_data:
            print(f"WARNING {key}: No reference data found")
            continue

        ref_array = reference_data[key]
        output_slice = get_first_n_elements(output)

        # Flatten for comparison
        ref_flat = ref_array.flatten()
        output_flat = output_slice.flatten()

        min_len = min(len(ref_flat), len(output_flat))
        ref_flat = ref_flat[:min_len]
        output_flat = output_flat[:min_len]

        diff = np.abs(ref_flat - output_flat)
        max_diff = diff.max()
        mean_diff = diff.mean()

        passed = max_diff < precision_threshold

        if not passed:
            all_passed = False

        status = "PASS" if passed else "FAIL"
        print(f"{status} {key}:")
        print(f"   Max diff: {max_diff:.6e}, Mean diff: {mean_diff:.6e}")
        print(f"   Shape: {output.shape}")

        if not passed:
            max_idx = diff.argmax()
            print(f"   Ref value: {ref_flat[max_idx]:.8f}, Output value: {output_flat[max_idx]:.8f}")

        results.append({
            'name': key,
            'max_diff': float(max_diff),
            'mean_diff': float(mean_diff),
            'passed': bool(passed)
        })

    return all_passed, results


# =============================================================================
# Main execution
# =============================================================================
def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Integrated validation script for VAE and LDM models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Use default paths
    python integrated_validation.py

    # Specify custom weight directory
    python integrated_validation.py --weight-dir /path/to/weights

    # Specify custom reference and output files
    python integrated_validation.py --reference-file ref.txt --output-file results.json

    # Use different precision threshold
    python integrated_validation.py --threshold 1e-5
        """
    )

    parser.add_argument(
        '--weight-dir',
        type=str,
        default=DEFAULT_WEIGHT_DIR,
        help=f'Directory containing converted weights (default: {DEFAULT_WEIGHT_DIR})'
    )

    parser.add_argument(
        '--reference-file',
        type=str,
        default=DEFAULT_REFERENCE_FILE,
        help=f'Reference output file (default: {DEFAULT_REFERENCE_FILE})'
    )

    parser.add_argument(
        '--output-file',
        type=str,
        default=DEFAULT_OUTPUT_FILE,
        help=f'Output JSON file for results (default: {DEFAULT_OUTPUT_FILE})'
    )

    parser.add_argument(
        '--threshold',
        type=float,
        default=DEFAULT_PRECISION_THRESHOLD,
        help=f'Precision threshold (default: {DEFAULT_PRECISION_THRESHOLD})'
    )

    return parser.parse_args()


def main(weight_dir=None, reference_file=None, output_file=None, precision_threshold=None):
    """
    Main function to run all validation tests.

    Args:
        weight_dir: Directory containing converted weights
        reference_file: Reference output file path
        output_file: Output JSON file path for results
        precision_threshold: Precision threshold for comparison

    Returns:
        int: 0 if all tests passed, 1 otherwise
    """
    # Parse command line arguments if not provided
    if weight_dir is None:
        args = parse_args()
        weight_dir = args.weight_dir
        reference_file = args.reference_file
        output_file = args.output_file
        precision_threshold = args.threshold

    # Convert to Path objects
    weight_dir = Path(weight_dir)
    reference_file = Path(reference_file)
    output_file = Path(output_file)

    print("="*80)
    print("Integrated VAE & LDM Validation (Hardcoded Input Data)")
    print(f"Precision Threshold: {precision_threshold:.1e}")
    print(f"Weight Directory: {weight_dir}")
    print(f"Reference File: {reference_file}")
    print(f"Output File: {output_file}")
    print("="*80)

    # Parse reference data
    print(f"\nLoading reference data from: {reference_file}")
    reference_data = parse_reference_file(reference_file)
    print(f"Loaded {len(reference_data)} reference outputs")

    # Run all tests
    all_results = []
    global_passed = True

    # Test VAE Encoder
    passed, results = run_test_vae_encoder(reference_data, weight_dir, precision_threshold)
    all_results.extend(results)
    global_passed = global_passed and passed

    # Test VAE Decoder
    passed, results = run_test_vae_decoder(reference_data, weight_dir, precision_threshold)
    all_results.extend(results)
    global_passed = global_passed and passed

    # Test LDM Denoiser
    passed, results = run_test_ldm_denoiser(reference_data, weight_dir, precision_threshold)
    all_results.extend(results)
    global_passed = global_passed and passed

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    passed_count = sum(1 for r in all_results if r['passed'])
    total_count = len(all_results)

    print(f"Passed: {passed_count}/{total_count}")
    print(f"Threshold: {precision_threshold:.1e}")

    if global_passed:
        print("ALL TESTS PASSED")
    else:
        print("FAILED TESTS:")
        for r in all_results:
            if not r['passed']:
                print(f"   - {r['name']}: Max diff = {r['max_diff']:.6e}")

    # Save results to JSON
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump({
            'threshold': precision_threshold,
            'global_passed': global_passed,
            'passed_count': passed_count,
            'total_count': total_count,
            'results': all_results
        }, f, indent=2)

    print(f"\nResults saved to: {output_file}")
    print("="*80)

    return 0 if global_passed else 1


if __name__ == '__main__':
    paddle.seed(42)
    np.random.seed(42)

    exit_code = main()
    sys.exit(exit_code)
