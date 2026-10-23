# HALMES: Knowing When to Intervene in LVLM Hallucination Mitigation

## Environment Setup

### Requirements and Compatibility

Running this project requires a GPU with sufficient VRAM: While training HALMES itself is lightweight, it operates as a plug-in for Large Vision-Language Model (LVLM) inference. Therefore, generating training its data and evaluating its performance demands adequate computational resources to perform inference on these large models.

We conducted our experiments primarily on Ubuntu using NVIDIA A800. However, we have avoided platform-specific features or shell-dependent scripts, so the project should run smoothly on any major platform, including Windows. If you encounter any issues on any major platform, please feel free to open an issue.

### Environment Management

As the project is divided into sub-packages to resolve historical dependency conflicts, managing it with standard pip is challenging. Therefore, we use uv, a modern Python project manager, for environment management. For installation instructions, please refer to [the official guide](https://docs.astral.sh/uv/getting-started/installation/). If you encounter any issues using uv with this project, we will do our best to provide support.

With uv, you don't need to worry about package installation: simply use `uv run` to execute programs, and dependencies will be handled automatically. The only potential complication is PyTorch. We recommend updating your CUDA version to the latest release, which allows you to use the standard version from PyPI without additional configuration. If you need to install PyTorch from a specific source, please refer to [the official guide](https://docs.astral.sh/uv/guides/integration/pytorch/#using-uv-with-pytorch).

### VS Code Support

We have configured a VS Code multi-root workspace. You can get started immediately by opening the `halmes.code-workspace` file. For tips on optimizing your development experience in VS Code and discussions of known issues, please see [the "VS Code" section](#vs-code).

## Directory Structure

```text
.
├── halmes                                 // This directory contains all executable scripts.
│   ├── li-halmes-configurations           // Global configurations. Contains presets to avoid redundant setup across multiple scripts.
│   ├── li-halmes-generate-data            // Runs the LVLM on training datasets and records the attention features.
│   ├── li-halmes-label-data               // Annotates the previously extracted features with hallucination labels.
│   ├── li-halmes-split-datasets           // Handles dataset splitting and balancing.
│   ├── li-halmes-train-halmes             // Trains the proposed HALMES model.
│   ├── li-halmes-validate-checkpoints     // Evaluates checkpoints on the validation set (corresponding to Table 1 in the paper) to identify the optimal model.
│   ├── li-halmes-inference                // Runs various model variants on test datasets.
│   └── li-halmes-score-inference-outputs  // Scores the inference outputs (corresponding to Tables 2 and 3 in the paper).
├── lib                                    // Contains utility packages and shared libraries called by the executable scripts.
│   ├── halmes/li-halmes-model             // Includes the full definition of the HALMES model. If you wish to migrate HALMES to another project, you can use this package directly.
│   └── ...                                // Other general utility tools (not listed exhaustively).
├── managing                               // Used for global project management and organization.
│   ├── links                              // A dedicated directory for symbolic links. This allows you to keep code, models, inputs, and outputs on different file systems while maintaining a unified access path within the project.
│   └── ...
├── halmes.code-workspace                  // The VS Code workspace file.
└── README.md                              // This documentation file.
```

## Data Preparation

To fully run our code, the following datasets are needed. We recommend storing the datasets directly at `managing/links` or linking them here via symbolic links, forming the following structure:

```text
links
├── datasets
│   ├── AMBER
│   │   └── ...
│   ├── Bingo
│   │   └── ...
│   ├── MMHal-Bench
│   │   └── ...
│   └── MS-COCO
│       └── ...
└── ...
```

Note: Scoring on certain datasets requires calling external LLMs and accessing tools such as nltk and spacy. While these tools are not listed here, they may require additional downloading or configuration. Please refer to [the "Configuration" section](#configuration) for details.

### COCO

The COCO dataset can be downloaded [here](https://cocodataset.org/). Specifically, its `train2014` will be used for training and validation, while `val2014` will be used for testing. We recommend extracting the archives and organizing the files into the following directory structure:

```text
MS-COCO
├── annotations
│   ├── captions_train2014.json
│   ├── captions_val2014.json
│   ├── instances_train2014.json
│   ├── instances_val2014.json
│   ├── person_keypoints_train2014.json
│   └── person_keypoints_val2014.json
├── train2014
│   ├── COCO_train2014_000000000009.jpg
│   ├── COCO_train2014_000000000025.jpg
│   └── ...
└── val2014
    ├── COCO_val2014_000000000042.jpg
    ├── COCO_val2014_000000000073.jpg
    └── ...
```

### AMBER

Please refer to [their GitHub repository](https://github.com/junyangwang0410/AMBER). You will need to clone the repository and download the images shared via their Google Drive link. We recommend organizing the files into the following directory structure:

```text
AMBER
├── data         // From GitHub
│   ├── query
│   │   ├── query_all.json
│   │   ├── query_discriminative-attribute.json
│   │   ├── query_discriminative-existence.json
│   │   ├── query_discriminative.json
│   │   ├── query_discriminative-relation.json
│   │   └── query_generative.json
│   ├── annotations.json
│   ├── metrics.txt
│   ├── relation.json
│   └── safe_words.txt
└── image        // From Google Drive
    ├── AMBER_1000.jpg
    ├── AMBER_1001.jpg
    └── ...
```

### Bingo

The Bingo dataset can be downloaded [here](https://huggingface.co/datasets/PahaII/Bingo). We recommend organizing it into the following directory structure:

```text
Bingo
├── images
│   ├── factual_factual_0.png
│   ├── factual_factual_10.png
│   └── ...
├── Factual.json
├── I2I.json
├── OCR.json
├── Region.json
└── T2I.json
```

### MMHal-Bench

The MMHal-Bench dataset can be downloaded [here](https://huggingface.co/datasets/Shengcao1006/MMHal-Bench). You may clone the Hugging Face repository or simply download `test_data.zip` (the contents are identical). We recommend the following directory structure:

```text
MMHal-Bench
├── images
│   ├── 10172500456_1f40b6bd38_o.jpg
│   ├── 11715451803_24861529ab_o.jpg
│   └── ...
└── response_template.json
```

### How to Add Other Datasets?

All dataset inputs are handled uniformly via the `DatasetInputProvider` interface. For implementation details, please refer to the specific dataset examples located in `lib/data-loading`.

- **For Training**: To use a new dataset for training, you should implement the `SupportedJudge` interface found in `halmes/li-halmes-label-data`. You can use our existing implementation for the COCO dataset as a reference.
- **For Testing**: To use a dataset for testing, you should implement the `SupportedJudge` interface located in `halmes/li-halmes-score-inference-outputs`. You can refer to our current implementations for AMBER, Bingo, COCO, and MMHal-Bench.

## LVLM Preparation

Similar to the datasets, we recommend storing the models directly at `managing/links/llms` or linking them here, forming the following structure:

```text
links
├── datasets
│   └── ...
├── llms
│   ├── InternVL3_5-8B-HF
│   │   ├── added_tokens.json
│   │   ├── chat_template.jinja
│   │   └── ...
│   └── Qwen3-VL-8B-Instruct
│       ├── chat_template.json
│       ├── config.json
│       └── ...
└── ...
```

InternVL3.5 8B can be downloaded [here](https://huggingface.co/OpenGVLab/InternVL3_5-8B-HF), and Qwen3 VL 8B [here](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct).

### How to Add Other Models?

- **For Training**: To generate training data specific to a new model, you should to implement the `SupportedLlm` interface in `halmes/li-halmes-generate-data`. Additionally, to ensure precise token positioning, you should implement the `SupportedTokenizer` interface in `halmes/li-halmes-label-data`.
- **For Testing**: To evaluate various contrastive decoding methods on a new model, you should to implement the `SupportedLlm` interface in `halmes/li-halmes-inference`.

You can refer to the existing implementations for InternVL3.5 8B and Qwen3 VL 8B as templates.

### How to Add Other Contrastive Decoding Methods?

Because contrastive decoding methods must be embedded within the model's inference loop, they often require specialized handling based on the model's specific architecture. Consequently, these methods are abstracted within the `SupportedLlm` implementation located in `halmes/li-halmes-inference`.

## Configuration

Each executable script requires individual configuration. However, we recommend first setting up global presets in `halmes/li-halmes-configurations`, which can then be directly called by subsequent scripts.

Note that presets from this package will be invoked explicitly. They are not "hidden" or automatically injected into other scripts. This means you do not need to complete all configurations at once; you can safely leave sections unfinished without worrying about scripts using incorrect or unknown default values in the background.

### Common Configuration

In `halmes_common_configuration.py`, you can configure the specific GPU device to be used.

### Output Path Configuration

In `halmes_output_path_configuration.py`, you can define where output files are stored.

By default, we recommend creating a directory (or a symbolic link) at `managing/links/outputs`, which will serve as the root for all generated stuffs.

### External LLM Configuration

In `halmes_openai_llm_configuration.py`, you can configure access to external LLMs (e.g., GPT-4o) used for dataset scoring.

By default, API keys are read from `managing/links/secrets/openai`. You can modify this logic as needed, but be sure to ensure the security.

### Tool Configuration

In `halmes_tool_configuration.py`, you can configure access to external tools like nltk and en_core_web_lg, which are primarily used for scoring.

We recommend using the default settings and placing the tools in `managing/links/tools` using the following structure:

```text
tools
├── nltk
│   └── nltk_data
│       ├── collections
│       │   └── ...
│       ├── packages
│       │   └── ...
│       └── ...
└── spacy
    └── en_core_web_lg
        ├── en_core_web_lg-3.8.0
        │   └── ...
        ├── __init__.py
        └── meta.json
```

These tools can be downloaded from [the nltk_data repository](https://github.com/nltk/nltk_data) and [the spacy-models repository](https://github.com/explosion/spacy-models/releases/tag/en_core_web_lg-3.8.0).

### Target LVLM Configuration

In `halmes_target_llm_configuration.py`, you can configure the target LVLMs.

- The `id` parameter determines the output directory name (ensure it contains no special characters).
- The `model_type` parameter is used for assertion checks to prevent logic errors; you can extend or ignore this as needed.

If you follow the recommended method of placing models in `managing/links/llms`, no adjustments to this file are necessary.

### Input Feature Configuration

In `halmes_feature_selector_configuration.py`, you can customize the features input into the HALMES model. This allows you to remove specific inputs for ablation studies, as demonstrated in Table 1 of our paper.

- The `id` parameter of `HalmesFeatureSelector` determines the output path.

Under normal circumstances, this file does not require modification.

### Dataset Configuration

In `halmes_dataset_configuration.py`, you can configure the datasets.

If you have placed your datasets in `managing/links/datasets` as recommended, no changes are required.

## Running the Project

### Step 1: Change the Working Directory

First, navigate to the package corresponding to the task you wish to perform. For example, to generate data, change your current working directory to `halmes/li-halmes-generate-data`.

For a full list of available packages and their purposes, please refer to [the "Directory Structure" section](#directory-structure).

### Step 2: Choose a Configuration

Using data generation as an example, we provide two default scripts: `halmes_generate_data_qwen3_vl_8b.py` and `halmes_generate_data_intern_vl_3d5_8b.py`. The former generates data for Qwen3 VL 8B, while the latter is for InternVL3.5 8B.

The only difference between these scripts is the content of their `Config` class. You can easily duplicate these scripts and modify the configuration to suit your specific needs. (Naming them `*.gitignored.py` to prevent your machine-specific configuration from being tracked by git.)

### Step 3: Execute the Script

Assuming you have chosen Qwen3 VL 8B, you can run the script directly using the following command (ensure you are still inside the `halmes/li-halmes-generate-data` directory):

```bash
uv run ./src/halmes_generate_data_qwen3_vl_8b.py
```

uv will automatically detect the dependencies listed in `pyproject.toml`, install them, and begin execution.

### Full Example

Below is a complete sample for generating data, training HALMES, and evaluating the performance on Qwen3 VL 8B.

```bash
# assume that it's now at the root directory of the repository

# to generate the datasets for training HALMES
cd halmes/li-halmes-generate-data
uv run src/halmes_generate_data_qwen3_vl_8b.py
cd ../..

cd halmes/li-halmes-label-data
uv run src/halmes_label_data_qwen3_vl_8b.py
cd ../..

cd halmes/li-halmes-split-datasets
uv run src/halmes_split_datasets_qwen3_vl_8b.py
cd ../..

# to train HALMES
cd halmes/li-halmes-train-halmes
uv run src/halmes_train_halmes_qwen3_vl_8b.py
cd ../..

cd halmes/li-halmes-validate-checkpoints
uv run src/halmes_validate_checkpoints_qwen3_vl_8b.py
cd ../..

# to inference with certain contrastive decoding methods
cd halmes/li-halmes-inference
uv run src/halmes_inference_qwen3_vl_8b.py
cd ../..

cd halmes/li-halmes-score-inference-outputs
uv run src/halmes_score_inference_outputs_qwen3_vl_8b.py
cd ../..
```

### Multi-threading

Some scripts in this project support multi-threading. To maintain a clean interface and avoid log interleaving, we do not print detailed execution progress to the console. Instead, logs are directed to a dedicated directory. The console will display the location of the log file at the start and print `Done.` upon completion.

Note that a "Done" message signifies the program has finished its execution cycle, but not necessarily that every task was successful. You should inspect the log files to verify the status of individual tasks. For any tasks that encounter exceptions, you can locate specific task logs to diagnose the issue.

Some scripts also include a "Skip Existing" feature. If a run fails or is interrupted, you can simply restart the script; it will automatically identify completed tasks and attempt to retry only the failed or missing ones. For efficiency and reliability, we use SQLite databases for outputs instead of standard CSV or Excel files. You can use some SQLite management tools (such as the SQLite Viewer extension in Visual Studio Code) to browse these files as easily as a regular spreadsheet.

## VS Code

### Recommended Extensions
- **Python**: For writing and debugging Python scripts.
- **Even Better TOML**: For editing and viewing TOML files (including `pyproject.toml` and our log files).
- **SQLite Viewer**: For browsing and managing SQLite database files.
- **vscode-pydata-viewer**: For viewing data saved with `torch.save`.
- **Flake8**: For certain linting support.

### Installing Dependencies

To enable proper type hints and autocompletion in VS Code, you must first install the dependency packages (for each package):

- Navigate into the package folder.
- Run `uv sync` to install the dependencies.

You don't need to manually `cd` into each package directory. When creating a new terminal in VS Code, you will be asked to select one package from a list, and the terminal will open in that directory by default.

After installation, we recommend restarting VS Code (or using the `Developer: Reload Window` command) to ensure the Python environment is correctly loaded. Additionally, if you add an editable dependency, you may need to restart the Python language server for the changes to be detected.

If you notice significant lag in VS Code's type hints or IntelliSense, you can comment out packages you are not currently editing in the `halmes.code-workspace`. This prevents Pylance from analyzing them unnecessarily. Note that packages depending on `transformers` are known to cause Pylance performance issues (see [this issue](https://github.com/microsoft/pylance-release/issues/7615) for details).

### Running the Project

In VS Code, you can run scripts directly by opening the file and pressing `F5`. You do not need to manually use `uv run` in the terminal; VS Code will automatically navigate to the correct working directory and select the appropriate Python environment for you.
