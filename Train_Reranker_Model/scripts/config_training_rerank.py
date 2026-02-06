# Configuration for reranker model fine-tuning
# Get a model to fine tune: a HuggingFace/SentenceTransformers-style model folder 
# with model weights, config, tokenizer, etc.) or a Transformers model.

# Multi-epoch training configuration
# Set CURRENT_EPOCH manually when running or resuming training
CURRENT_EPOCH = 1  # Which epoch to run (1=easy, 2=medium, 3=hard)
RUN_EPOCHS = [1, 2, 3]  # List of all epochs to run in sequence

# Directory configuration
BASE_CWD = Path("C:/GIT/Z_Master_Rag/Data/framemaker/mif_jsx") # Base domain working directory. May contain 1-N docs.
LOG_FILES = BASE_CWD

# Shared subdirectory names so every script references the same structure.
RERANK_TRAINING_SUBDIR = "reranker_training_data"
TOKENIZED_SUBDIR = "tokenized_HFTrainer"
RERANK_OUTPUT_SUBDIR = "rerank_model_adapter"

# Model configuration
BASE_MODEL = "C:\\GIT\\Z_Master_Rag\\Data\\framemaker\\mif_jsx\\Qwen3-Reranker-0.6B"
TRAINING_DATA_DIR = BASE_CWD / RERANK_TRAINING_SUBDIR # Training data base path: difficulty subdirectory added by pipeline
CONFIG_MODEL_NAME = BASE_MODEL # Model to use for tokenization and training set by pipeline based on CURRENT_EPOCH (so epock 2 uses epoch 1 output, etc.)
RERANKER_OUTPUT_PATH = BASE_CWD / RERANK_OUTPUT_SUBDIR # PEFT adapter weights and other trainer artifacts
TOKENIZED_DATA_DIR = TRAINING_DATA_DIR / TOKENIZED_SUBDIR # Tokenized dataset location (created by scripts/tokenize_triplets.py)

# Reranker training data configuration #####################################################
RERANK_INSTRUCTION = "Given a FrameMaker MIF or JSX scripting question, retrieve information about MIF object structure and JSX script behavior."
MIN_CONTENT_LENGTH = 100  # Minimum characters for content to be useful for training
MAX_CONTENT_LENGTH = 4000  # Maximum characters to avoid overwhelming the model
MIN_TRIPLETS_PER_CATEGORY = 200  # Minimum triplets per category/domain
MAX_TRIPLETS_PER_CATEGORY = 400  # Maximum triplets per category to avoid bias
TRAIN_TEST_SPLIT = 0.8  # 80% training, 20% evaluation
NEGATIVE_SAMPLING_RATIO = 2 # Number of negative examples to attempt per anchor when constructing auxiliary pairs
GENERATE_DIFFICULTY_LEVELS = True  # Set to False to skip easy/medium/hard separation
ANCHOR_CATEGORY_BALANCE = {"jsx": 0.5, "mif": 0.5}  # Target share of anchors per source doc
SAME_SOURCE_POSITIVE_RATIO = 0.7  # 70% of positives should come from the same source doc as the anchor
CROSS_SOURCE_HARD_NEGATIVE_RATIO = 0.3  # 30% of hard negatives should come from the opposite source doc
EASY_TRIPLET_RATIO = 0.20
MEDIUM_TRIPLET_RATIO = 0.35
HARD_TRIPLET_RATIO = 0.45

# Training parameters
# PERFORMANCE IMPACT GUIDE:
# ========================
# Model choice: PyTorch training requires the model in HuggingFace format, so 
# CONFIG_MODEL_NAME must point to a valid model directory.
# These settings directly affect training time, memory usage, and model quality.
#
# CONTINUING TRAINING (Multi-Epoch Strategy):
# To continue training from a previous epoch:
# 1. After epoch 1 completes, the model is saved to OUTPUT_MODEL_PATH (configured in config)
# 2. Update CONFIG_MODEL_NAME in config to point to the saved model from epoch 1
# 3. Run training again for epoch 2 - it will load from your fine-tuned model
# 4. Repeat: Update CONFIG_MODEL_NAME to epoch 2 output, run epoch 3
# 5. This allows progressive refinement with harder negatives or additional data
#
# LEARNING_RATE: How aggressively model weights are updated
# - 1e-6 (very low): Stable, slow learning, may underfit
# - 5e-6 (current): Conservative, prevents NaN, longer training
# - 2e-5 (normal): Faster learning, risk of instability/NaN
# - 1e-4+ (high): Fast but likely to cause NaN or poor quality
# - Training time impact: Minimal (same computation)
# - Quality impact: Too low = underfit, too high = NaN/overfit
#
# BATCH_SIZE: How many triplets processed simultaneously
# - Memory formula: ~2GB × batch_size for Qwen3-4B
# - Per 1000 triplets: Time scales roughly inversely with batch size
# - Effective batch size = batch_size * gradient_accumulation_steps
#
# FP16: Half-precision training (16-bit vs 32-bit floats)
# - False (current): Stable, full memory usage, prevents NaN
# - True: 2x faster, 50% less memory, but may cause NaN with large models
# - Memory savings: ~30-50% reduction
# - Time savings: ~50% faster training
# - Risk: Higher chance of NaN
#
# GRADIENT_ACCUMULATION_STEPS: Simulate larger batches without memory cost
# - 1: No accumulation, memory efficient
# - 4 (current): Simulates 4x batch size (8×4=32 effective batch)
# - 8+: Very large effective batch, may slow convergence
# - Memory impact: Minimal increase
# - Training impact: More stable gradients, slightly slower per step
#
# max_sequence_length impact (practical):
# - Attention memory can grow O(seq_len^2) for standard full attention (biggest memory pressure).
# - Token buffers and activations scale ~O(seq_len); doubling seq_len roughly doubles per-step compute.
# - Longer seqs increase tokenization memory and slow each training step; pick the shortest seq that preserves signal.
# - Mitigations: truncate/chunk long docs, use sliding-window or memory-efficient attention, 
# enable gradient_checkpointing/fp16 when safe. 
#
# Difficulty levels for curriculum learning
# EASY: Random chunks from completely different domains/topics
# MEDIUM: Chunks from similar topics but different contexts
# HARD: Chunks from same topic that look similar but aren't the answer

TRAINING_CONFIG = {
    "epochs": 1,  # Number of passes through the training data.
    "learning_rate": 2e-5,  # Higher LR for faster convergence on smaller model
    "warmup_steps": 50,  # Shorter warmup
    "evaluation_steps": 100,  # Number of steps between evaluations: no vram impact.
    "save_steps": 500, # Number of steps between model saves: no vram impact.
    "fp16": True,  # Enable fp16 to reduce memory usage by ~50%
    "gradient_accumulation_steps": 4,  # Interacts with batch size. Allows simulating larger batches with no memory buildup
    "batch_size": 4,  # Raises mem usage, but larger batches imporve training quality by averaging gradients over more examples.
    "max_sequence_length": 1024,  # Maximum token length for inputs; adjust based on your data
    # NOTE: This should never exceed the model's `config.max_position_embeddings`.
    #       Verify the local model's config (model.config.max_position_embeddings) is >= this value.
    #       E.g., for Qwen-3 variants this can be very large (tens of thousands) but always check.
    # Fail-fast and timeout controls
    # If True, any attempt to fall back to CPU (for optimizer state/allocation or retry) will abort the process.
    "FAIL_ON_CPU_FALLBACK": False,
    # Max allowed time (seconds) for a single training step. If exceeded, the script will abort.
    "MAX_STEP_SECONDS": 120,
    # Max total time (seconds) for a limited run. If exceeded, the script will abort.
    "MAX_TOTAL_SECONDS": 900,
    # Optional advanced performance flags
    # If True, attempt to use torch.compile() on the model (PyTorch 2.x). Test first. May causes issues with other code.
    "USE_TORCH_COMPILE": False,
    # If True, attempt to load the underlying HF model in 8-bit with device_map='auto'. Requires bitsandbytes & accelerate.
    # Disabled by default to prefer FP32 behavior and avoid allocator/fragmentation changes.
    "USE_LOAD_IN_8BIT": False,
    # Number of prefetch worker processes/threads used by the limited-run prefetcher. Default 1 (thread), set higher to use more CPU.
    "PREFETCH_WORKERS": 0,
    "DATALOADER_PIN_MEMORY": True,
    # How many steps to run for smoke tests (limited-step runs)
    "SMOKE_STEPS": 6,
    # Diagnostics/profiling flags
    "USE_BOTTLENECK": False,  # If True, run the smoke test under `python -m torch.utils.bottleneck`
    # Disable the heavy profiler for normal FP32 smoke runs to avoid runtime overhead and CUPTI issues on Windows
    "USE_TORCH_PROFILER": False,
    "TORCH_PROFILER_TRACE": None,
    "USE_NSIGHT": False,  # If True, attempt to invoke NVIDIA Nsight (nsys) to profile the smoke run (external tool)
    # How often (every N steps) to emit detailed DIAG logs (isfinite/mean/std).
    # Set to >1 (e.g., 5) to reduce CPU logging overhead during long runs.
    "DIAG_EVERY_N_STEPS": 5,
    # LoRA (PEFT) defaults - opt-in via config or CLI flag
    "use_lora": True,
    "lora_r": 16, # (rank) — # of trainable dimensions added to each weight matrix. < 16 = faster, less VRAM > = more VRAM better quality. 
    "lora_alpha": 32, # scaling factor: controls how strongly LoRA modifies the base weights. Typical default = alpha = 2 * r
    "lora_dropout": 0.1,  # Dropout probability for LoRA layers
    "lora_target_modules": ["q_proj", "v_proj"],  # Which layers to apply LoRA to (attention query/value projections)
    # Triplet loss configuration
    "triplet_margin": 1.2,  # Margin for triplet loss (distance between positive and negative). Higher = stronger separation
    # Model loading configuration
    "trust_remote_code": True,  # Whether to trust remote code when loading models (required for some models)
}

RERANKER_TRAINING_CONFIG = {
    "epochs": 3,
    "learning_rate": 2e-5,
    "batch_size": 16,  # Cross-encoders use more memory per batch
    "warmup_steps": 100,
    "max_length": 512,  # Query + document together
    "use_amp": True,  # Automatic mixed precision for speed
}