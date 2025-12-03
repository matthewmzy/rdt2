# RDT2 Copilot Instructions
## Big Picture
- RDT2 couples a Qwen2.5-VL-7B backbone (`models/rdt`, `models/rdt_inferencer.py`) with a Residual VQ action tokenizer (`vqvae/models`) to emit 24×20 bimanual action chunks at 30 Hz.
- `models/normalizer.LinearNormalizer` keeps the action/state statistics consistent between datasets, inference, and flow-matching expert models.
- Flow-matching experts (`models/rdt_runner.py`, configs under `configs/rdt/`) distill the VLA outputs for low-latency deployment.
- Hardware deployment lives in `deploy/` and assumes UMI grippers, stereo wrist cameras, and tracker-to-TCP calibration derived from Vive trackers.
## Repo Landmarks
- `main.py` parses CLI flags and forwards to `train.py`, which wires QLoRA/LoRA/FSDP-friendly HuggingFace training with flash-attn Qwen weights.
- `vla_trainer.py` extends `transformers.Trainer` to rotate across multiple eval datasets and compute action metrics (`utils.compute_action_metrics`).
- `configs/` holds dataset recipes (`configs/datasets/*.yaml`), robot IP/camera geometry (`configs/robots/*.yaml`), and RDT post-training hyperparameters.
- `data/` implements loaders: `umi_video_dataset.py` pulls zarr/webdataset shards via `ReplayBuffer`, `data/utils.py` blends shards with optional instruction overrides.
- `deploy/umi/**` wraps real robots via `BimanualUmiEnv`, shared-memory queues, and collision solvers; `deploy/inference_real_vq.py` is the reference CLI.
- `scripts/finetune_*.sh` show the expected `accelerate launch` invocations, ZeRO configs, and checkpoint locations.
## Data + Normalization
- Training data must be WebDataset shards named `shard-*.tar` containing `.image.jpg`, `.action.npy`, `.action_token.npy`, and `.meta.json` tuples.
- Dataset configs (e.g. `configs/datasets/example.yaml`) declare `shards_dir`, `instruction_path`, and `normalizer_path`; placeholders often reference `{hostname}`.
- `data/utils.get_instructions_and_blended_train_dataset` merges additional instruction dictionaries, so keep JSON keys stable across datasets.
- `LinearNormalizer` checkpoints (default `umi_normalizer_wo_downsample_indentity_rot.pt`) must align with both tokenizer scale and robot execution; store them beside shards if custom.
- Image ordering matters: `preprocess_data_from_umi` reverses camera IDs to left→right before concatenation, so keep `camera0`=right and `camera1`=left in datasets.
- `umi_video_dataset` enforces `shape_meta` from `configs/bimanual_video_data.yaml`; adjusting image resolutions requires updating both `raw_shape` and resize transforms.
## Training Workflow
- Typical command: `accelerate launch main.py --deepspeed scripts/zero1.json --pretrained_model_name_or_path robotics-diffusion-transformer/RDT2-VQ --tokenizer_name Qwen/Qwen2.5-VL-7B-Instruct --vae_name robotics-diffusion-transformer/RVQActionTokenizer --dataset configs/datasets/example.yaml --image_corruption`.
- Toggle `--use_lora`/`--use_qlora` to inject PEFT adapters; `train.py` already targets Qwen proj modules and prepares k-bit weights.
- Always keep `processor.tokenizer.add_special_tokens('<action>')` intact—removing it breaks the action-id remapping that replaces generated `<action>` placeholders.
- When enabling eval (`--eval_strategy steps/epoch`), ensure dataset configs provide `kwargs.normalizer_path`; eval collate swaps between identity/custom via `--use_default_collate_fn_for_eval`.
- `VLATrainer` only evaluates `num_eval_batches` batches per dataset; adjust this if you need statistically stable metrics before gating checkpoints.
## Inference & Deployment
- `utils.batch_predict_action` runs the canonical prompt → generate → RVQ decode → unnormalize loop; it expects JPEG-compressed images for best parity (`apply_jpeg_compression=True`).
- `models/RDTInferencer` wraps the flow-matching expert: pass `pretrained_vision_language_model_name_or_path='robotics-diffusion-transformer/RDT2-VQ'` and matching `normalizer_path` when loading `configs/rdt/post_train.yaml`.
- Real-robot CLI (`deploy/inference_real_vq.py`) requires `--data_config configs/bimanual_video_data.yaml` plus robot configs under `configs/robots/`; it caches processor/LLM/VAE per process to avoid repeated loads.
- Calibrate tracker-to-TCP transforms via `deploy/calibration/*` and paste matrices into `configs/robots/eval_bimanual_*_config.yaml` before deployment, otherwise collision checks use the wrong frames.
- `deploy/umi/.../real_inference_util.py` converts model actions to robot TCP space; ensure gripper widths are rescaled from `[0,0.088]` to `[0,0.1]` when bypassing helpers.
## Project Conventions
- Language prompts should follow “Verb Object.” (capitalized, trailing period) to match the supervised data and maximize instruction grounding.
- Two-arm assumptions are hard-coded (action_dim=20 split into 10-d blocks); gate any single-arm experiments by slicing outputs before downstream controllers.
- Default precision is `bfloat16` for both Qwen and flow-matching models; keep VAE decodes in `float32` to avoid discretization drift.
- Do not reorder camera names—the model maps `camera0_rgb` to the right arm and `camera1_rgb` to the left; deployment scripts use regexes that depend on this naming.
- When adding new metrics or datasets, reuse `utils.compute_action_metrics` so dashboards keep `action_valid_rate`, `action_mse_error_*`, and geodesic rotation errors in the same format.
