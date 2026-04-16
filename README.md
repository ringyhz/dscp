# DSCP Experiments

This folder contains script versions of the DSCP and X-DSCP LightGCN experiments that were previously only available as notebooks.

## Files

- `dscp_lightgcn.py`: standalone DSCP LightGCN training and evaluation entry point.
- `x_dscp_lightgcn.py`: standalone X-DSCP LightGCN training and evaluation entry point.
- `load_data.py`: dataset loading and split generation.
- `evaluation.py`: ranking metric evaluation helpers.

## Requirements

Install the dependencies in `requirements.txt` and place the expected datasets under `data/`.

```bash
pip install -r requirements.txt
```

## Usage

Run the default Grocery DSCP experiment with the same five seeds used in the paper:

```bash
python dscp_lightgcn.py
```

Run the X-DSCP variant:

```bash
python x_dscp_lightgcn.py
```

Example with custom settings:

```bash
python dscp_lightgcn.py --dataset grocery --embedding-dim 16 --random-seeds 111 222 --alpha 0.1 --beta 0.1
```

Results are saved to `DSCP_LightGCN-<dataset>-results.npy` or `X-DSCP_LightGCN-<dataset>-results.npy` by default, depending on the script.

## Citation

If you use this code in your research, please cite the following paper:

Yihong Zhang and Takahiro Hara, 2025. Modeling Social Behavior in Collaborative Filtering. In Proceedings of the 48th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR 2025), Padua, Italy, pp. 1954-1963.
