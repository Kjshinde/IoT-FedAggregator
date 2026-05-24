# IoT-FedAggregator

Flower-based federated learning for screenshot classification. The project trains a small PyTorch CNN across distributed clients while keeping each client's image data under its own local `client_<id>` partition.

## What Is In This Repo

```text
.
├── flower-fl/                  # Runnable Flower application
│   ├── client/                 # Real FL clients and local client data mount point
│   │   ├── client.py           # Basic client
│   │   ├── client_v2.py        # Client with cleaner metrics logging
│   │   ├── data/               # Ignored local client datasets
│   │   └── requirements.txt
│   ├── common/                 # Shared model, data loading, and metrics code
│   ├── server/                 # Flower server, Dockerfile, server requirements
│   └── tools/
│       ├── diagnostics/        # TCP connectivity checks
│       └── simulation/         # Dummy one-client Flower smoke test
├── scripts/                    # Repo maintenance and dataset preparation scripts
├── experiments/                # Recorded experiment logs and generated outputs
├── docs/reports/               # Final project report and supporting docs
├── archive/legacy/             # Older notebook/prototype artifacts
├── LICENSE
└── README.md
```

The active training code lives in `flower-fl/`. Experiment logs, report files, and legacy notebooks are kept outside the runtime app so the main client/server paths stay easier to scan.

## Requirements

- Python 3.9+ recommended
- `pip`
- Client machines need PyTorch and torchvision
- Server machines need Flower, and also PyTorch when using `server.py` because it initializes the global CNN parameters

For a single local environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r flower-fl/server/requirements.txt -r flower-fl/client/requirements.txt
```

On separate machines, install only the requirements needed by that role.

## Prepare Client Data

Client data is expected at:

```text
flower-fl/client/data/client_<id>/
├── train/
│   ├── Food/
│   ├── movie/
│   ├── notes/
│   ├── real_life/
│   └── shopping/
└── test/
    ├── Food/
    ├── movie/
    ├── notes/
    ├── real_life/
    └── shopping/
```

Use the dataset splitter when you have a zipped image dataset arranged either as class folders at the zip root or as one top-level folder containing class folders:

```bash
python3 scripts/prepare_dataset.py path/to/screenshots.zip \
  --client-id 1 \
  --fraction 0.2 \
  --train-ratio 0.8
```

This writes to `flower-fl/client/data/client_1/` by default and does not overwrite existing images. Client data is ignored by git.

## Run Federated Training

Start the server:

```bash
cd flower-fl/server
python3 server.py \
  --port 8080 \
  --num-rounds 10 \
  --strategy FedAvg \
  --local-epochs 1 \
  --learning-rate 0.01 \
  --num-classes 5
```

Start one or more clients in separate terminals or on separate machines:

```bash
cd flower-fl/client
python3 client_v2.py \
  --client-id 1 \
  --server-address 127.0.0.1:8080 \
  --num-classes 5
```

Supported server strategies are `FedAvg`, `FedAdagrad`, `FedAdam`, and `FedYogi`. The server saves the best model parameters as `best_model.npz` in the server working directory; model artifacts are ignored by git.

For networked clients, replace `127.0.0.1:8080` with the server host/IP and make sure TCP port `8080` is reachable through the OS firewall and router/NAT rules.

## Smoke Tests And Diagnostics

Run a dummy Flower setup without image data:

```bash
python3 flower-fl/tools/simulation/server_sim.py --port 8080 --rounds 3
python3 flower-fl/tools/simulation/client_sim.py --server 127.0.0.1:8080
```

Check raw TCP connectivity before debugging Flower itself:

```bash
python3 flower-fl/tools/diagnostics/server_ping.py --port 8080
python3 flower-fl/tools/diagnostics/client_ping.py 127.0.0.1 --port 8080
```

## Scripts

`scripts/prepare_dataset.py` prepares one client's train/test split from a dataset zip.

`scripts/create_structure.py` scaffolds a Flower project layout. Use it with a new target directory to avoid overwriting this app:

```bash
python3 scripts/create_structure.py --full --client-id 2 --base-dir scratch-fl
```

## Experiment Artifacts

Past run logs are stored in `experiments/logs/`. Server-generated text/image outputs that used to live under `flower-fl/server/` are now in `experiments/server/` so the active server folder contains only runtime code and dependencies.
