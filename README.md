## PathSpec

This directory provides scripts for running full-dataset PathSpec evaluation with the `speculative_jacobi` setting and a static grouped tree (`Grouped_Tree_2`).

## Environment

Recommended environment:

- Python 3.10
- CUDA 12.5
- PyTorch 2.5.1+cu124
- Transformers 4.47.1

Create the environment from YAML:

```bash
conda env create -f environment.yaml
```

## Full Benchmark Evaluation Scripts

The main test entry points are the following three scripts:

- `script/mscoco2017_pathspec.sh`
- `script/pp_pathspec.sh`
- `script/t2i_pathspec.sh`

Before running any script, set `output_path` in the script to your result directory.

### 1) MSCOCO2017

Script: `script/mscoco2017_pathspec.sh`

```bash
bash script/mscoco2017_pathspec.sh
```

Default benchmark settings:

- `prompt="MSCOCO2017Val"`
- `num_images=4000`
- `slice='1759-4000'`

### 2) PartiPrompts

Script: `script/pp_pathspec.sh`

```bash
bash script/pp_pathspec.sh
```

Default benchmark settings:

- `prompt="PartiPrompts"`
- `num_images=1600`
- `slice='1370-1600'`

### 3) T2ICompBench

Script: `script/t2i_pathspec.sh`

```bash
bash script/t2i_pathspec.sh
```

Default benchmark settings:

- `prompt="T2ICompBenchVal"`
- `num_images=2400`
- `slice='0-2400'`

## Notes

- `mscoco2017_pathspec.sh` uses `../main.py`.
- `pp_pathspec.sh` and `t2i_pathspec.sh` use an absolute path to `main.py`. Update paths if your local directory layout is different.
- Scripts are configured with:
  - `isp='random'`
  - `method='speculative_jacobi'`
  - `num_init_new_token=16`
  - `benchmark_way='order'`
  - `--static_tree --tree_choices=Grouped_Tree_2`
