# Multi-View Part Recognition

Training and evaluation scripts for multi-view object-part recognition on the MVIP dataset.

## Dataset

Download MVIP from:
https://fordatis.fraunhofer.de/handle/fordatis/358

Expected dataset root (passed via `--path`) is similar to:

```text
<path>/
  <class_name>/
    meta.json
    train_data/<position>/<rotation>/cam_XX/
      XX_rgb.png
      XX_rgb_mask_gen.png
      XX_depth.png
      XX_meta.json
      XX_hha.png (optional)
    valid_data/<sample>/cam_XX/...
    test_data/<sample>/cam_XX/...
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If you need CUDA-specific PyTorch wheels, install the matching `torch`/`torchvision` pair from the official PyTorch index after the command above.

## Common Commands

Train:

```bash
python3 main.py --path ./MVIP/sets --outdir ./results --name run001
```

Evaluate best checkpoint of a run:

```bash
python3 test.py --outdir ./results/run001 --name run001
```

Run predefined experiment sweeps:

```bash
python3 exp.py
```

Batch retest previous runs:

```bash
python3 run_tests.py
```

Plot a run log:

```bash
python3 plot_logs.py --outdir ./results/run001 --name run001
```

## Configuration

A JSON configuration file can be used to set default values for common CLI options:

```bash
cp config.example.json config.json
$EDITOR config.json
```

Supported keys:

| Key | Description | Default |
|-----|-------------|---------|
| `dataset_path` | Dataset root (used when `--path` is not passed) | `./MVIP/sets` |
| `results_path` | Output directory for runs and logs | `./results` |
| `checkpoints_path` | Checkpoint directory | `./checkpoints` |

`config.json` is gitignored since paths are machine-specific. CLI arguments always override config values.

## Notes

- This repository does not currently include a `pytest` suite or CI pipeline; `test*.py` scripts are experiment/evaluation runners.
- Scripts read default values (such as dataset path) from `config.json`; explicit CLI arguments (`--path`, `--outdir`, `--name`, etc.) always take precedence.
- Many boolean CLI flags use `argparse` with `type=bool`, which can be unintuitive when overridden from the command line.
- Sample `run_real` checkpoints are stored as split parts in `checkpoints/run_real/` to comply with GitHub's 100MB file limit. Reconstruct with:
  `bash reconstruct_run_real_checkpoints.sh`
