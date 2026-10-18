# Train on a rented RTX 3090 VPS

## Recommended instance

Choose one NVIDIA RTX 3090 with:

- Ubuntu 22.04 or a PyTorch CUDA image;
- at least 8 vCPU and 32 GB RAM;
- at least 80 GB disk for downloaded ScanNet data and preprocessing;
- SSH access and persistent storage until the checkpoint is downloaded.

The model uses one GPU. Renting multiple 3090s does not help without adding
distributed training.

## Clone and run entirely on the VPS

No local upload is required. On the Linux VPS:

```bash
cd /workspace
git clone https://github.com/nuitfsdev/object-selector.git scanqa_object_selector
cd scanqa_object_selector
bash run_vps_all.sh --accept-scannet-tos
```

The pipeline downloads the official ScanQA annotations from Google Drive,
downloads the required ScanNet files, prepares both caches, verifies them, and
starts training.

## Data layout created automatically

The scripts create this layout:

```text
/workspace/
  data/scanqa/ScanQA_v1.0_train.json
  data/scanqa/ScanQA_v1.0_val.json
  scanqa_object_selector/
```

After personally accepting the ScanNet terms, one script downloads the three
required files per scene, creates both caches, and verifies their coverage:

```bash
cd /workspace/scanqa_object_selector
bash vps_setup.sh
bash prepare_vps_data.sh --accept-scannet-tos
bash run_vps_training.sh
```

This route needs more disk and spends GPU rental time on CPU preprocessing.

To perform setup, preparation, validation, and training in one command:

```bash
bash run_vps_all.sh --accept-scannet-tos
```

## Install and verify CUDA PyTorch

Connect through SSH, then run:

```bash
cd /workspace/scanqa_object_selector
chmod +x vps_setup.sh run_vps_training.sh
bash vps_setup.sh
nvidia-smi
```

The setup must print `CUDA available: True` and `NVIDIA GeForce RTX 3090`.

## Run safely after disconnecting SSH

Use `tmux` so training continues after closing the terminal:

```bash
sudo apt-get update && sudo apt-get install -y tmux
tmux new -s scanqa
cd /workspace/scanqa_object_selector
bash run_vps_training.sh
```

Detach with `Ctrl+B`, then `D`. Reconnect later:

```bash
tmux attach -t scanqa
```

Monitor the GPU in another SSH session:

```bash
watch -n 2 nvidia-smi
```

The included `config.3090.json` uses batch size 64, four DataLoader workers, CUDA,
and AMP. Reduce batch size to 32 if CUDA reports out-of-memory.

## Save results before terminating the VPS

Find the newest run:

```bash
ls -lt runs
```

Verify that the downloaded run contains:

```text
best.pt
last.pt
history.json
config.json
```

Use the VPS provider's file browser or persistent volume to retain the `runs`
and `logs` directories. They can also be uploaded to your model storage account.

Only after downloading these files should the rented instance and its volume be
deleted. Stopping compute may still leave storage billing active, depending on
the provider.
