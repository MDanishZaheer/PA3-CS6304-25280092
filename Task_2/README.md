# Task 2: SigLIP–Qwen visual grounding

`main_task_2.ipynb` implements the prescribed 196-visual-token Task 2 run in short, ordered cells. The completed Colab training and evaluation outputs are saved under local `Task_2/results/`. The notebook can still run unfinished work on Colab Pro with an L4 GPU and High RAM; locally, it replays the saved results without model computation.

## Upload and run

1. Upload this complete `Task_2` folder to `MyDrive/ATML_PA3/Task_2/`. Include `assets/times.ttf` from your locally licensed Times New Roman font. It is ignored by Git, but the figure code requires it. Keep the Drive path exact or edit `DRIVE_PROJECT` in the notebook.
2. Open `MyDrive/ATML_PA3/Task_2/main_task_2.ipynb` in Colab. Select **L4 GPU** and **High RAM**. Use **Run all**. The notebook mounts Drive, copies code to `/content/Task_2`, installs the pinned Python packages, downloads the official COCO/VQAv2 metadata and selected COCO images, and downloads pinned SigLIP/Qwen snapshots.
3. All raw downloads and FP16 feature chunks are in `/content/Data/Task_2/`. Feature chunks are copied to `MyDrive/ATML_PA3/Data/Task_2/features/` as they finish. No raw data needs to be uploaded to Drive first. The selected IDs and pinned model revisions are written into `Task_2/results/`.
4. Runtime results are written to `/content/Task_2/results/` and copied to `MyDrive/ATML_PA3/Task_2/results/`. Every 250-update recovery checkpoint is copied with a SHA-256 sidecar immediately after saving. Histories, development metrics, and other small files are synced at each checkpoint and evaluation. If the notebook stops near its time reserve, start a fresh L4/High RAM runtime and use **Run all** again. It restores the latest verified checkpoint, feature chunks, and saved outputs and continues at the next update.
   If the COCO host reports a TLS hostname mismatch, use the current `data/download.py` from this folder. It confines the certificate workaround to the COCO host and checks the 2014 annotation archive against its published SHA-256. Do not add a notebook-wide `requests.Session.request` patch; that also affects VQAv2 and model downloads and can recurse when the cell is rerun.
5. Training completes at update **4,500**. The final evaluation has `grounding_test` and `natural_test` in `results/metrics/final_evaluation.json` and matching JSONL prediction files. The notebook then builds RQ1–RQ4 tables and 300 dpi PNG figures. The optional 49-token comparison is not part of this main run.

The fixed batch is microbatch 8 with two accumulation steps. High RAM holds cached FP16 patch features; the training batch and final-generation batch remain at the assignment's prescribed sizes. SigLIP feature extraction uses batch 128, then the encoder is removed before Qwen training. Model memory occupancy is not expected to approach 22.5 GB throughout the run.

## Saved layout and local evidence rerun

```text
Task_2/
  main_task_2.ipynb
  configs/base.yaml
  data/  models/  training/  evaluation/  cloud/  common/
  assets/times.ttf                    # supplied locally; ignored by Git
  results/
    checkpoints/                     # every 250 updates and each stage export; Git-ignored
    histories/training.csv
    metrics/                         # revision pins, diagnostics, development and test summaries, CSV tables
    predictions/                     # per-example development NLL and final generations
    splits/                          # selected IDs and fixed mismatched-image maps
    examples/                        # deterministic qualitative selection
    figures/                         # Times New Roman 10 pt, 300 dpi PNGs
Data/Task_2/                        # downloads, snapshots, images, feature cache; Git-ignored
```

To inspect results locally, use the saved `histories/`, `metrics/`, `predictions/`, `splits/`, `examples/`, and `figures/` folders already in `Task_2/results/`. No other Drive files are needed for this replay. Open `main_task_2.ipynb` with the local `ATML` Conda kernel and run all cells. It verifies the saved 4,500-update run, displays each step's recorded Colab metrics and stage resources, then rebuilds the research-question tables and figures. It does not download data, load models, rerun evaluations, or train. Keep the generated qualitative PNGs so the local replay does not need COCO images.

The compact machine-readable `histories/`, `metrics/`, `predictions/`, `splits/`, `examples/`, and `figures/` folders can be committed after a completed run. Raw data, model snapshots, feature chunks, and recovery checkpoints should stay out of Git. A fresh rerun needs the selected-ID file and checkpoint set in Drive; a local evidence-only rerun does not need checkpoint weights.

## Protocol checks

- `data.prepare.verify_selection` checks dataset counts, pair construction, 8/10 VQA consensus, question-type quotas, and image isolation. `models.vlm.GroundedQwen` inserts 196 post-layer-normalized SigLIP patch tokens into Qwen's non-thinking chat prompt. `training.train.TrainingSession.check_interface` checks feature shape/dtype, frozen original Qwen weights, label masking, and a finite nonzero connector gradient.
- `data.inspection.build_grounding_label_audit` fixes a seeded six-pair train/development/test contact sheet before model predictions are viewed. Inspect it manually for annotation omissions or ambiguous counts; the code cannot certify visual label correctness.
- Stage 1 trains the connector for two passes over 18,000 captions. Stages 2 and 3 train the same connector and rank-16 attention LoRA with the prescribed pair-level sampling probabilities and caption replay. The FP16 backbone is frozen; connector, LoRA, and AdamW states are FP32. A dynamic scaler retries a batch if its optimizer step is skipped.
- Development checks occur before training, every 500 updates, and at stage endpoints 2,250, 3,750, and 4,500. The final test uses all four prescribed image conditions, greedy batch-8/16-token generation, local VQA soft scoring, pair diagnostics, and 4,000 bootstrap resamples of complete matched pairs. Saved partner-answer and unsupported-answer metrics help interpret complementary swaps without treating every changed answer as correct visual grounding.
- The notebook never uses held-out test results to choose a checkpoint or sampling setting. `configs/base.yaml` and `results/metrics/model_revisions.json` record the fixed recipe and source revisions.

## Sources and attribution

The data code downloads official [COCO 2014 annotations and images](https://cocodataset.org/#download) and [VQAv2 questions, annotations, and complementary pairs](https://visualqa.org/download.html). The answer-normalization rules in `data/answers.py` follow the official [VQA evaluator](https://github.com/GT-Vision-Lab/VQA/blob/master/PythonEvaluationTools/vqaEvaluation/vqaEval.py); the leave-one-out ten-reference score follows the assignment formula. The model code loads [SigLIP base patch16-224](https://huggingface.co/google/siglip-base-patch16-224) and [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) through Hugging Face Transformers, with attention adapters through PEFT. The frozen feature cache uses `SiglipVisionModel.last_hidden_state`, which already has SigLIP's final layer normalization applied. LLM assistance was used to implement and review Task 2 code; the student remains responsible for verifying it and writing the report independently.
