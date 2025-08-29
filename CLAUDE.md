# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

The Frame Representation Hypothesis is a research project focused on multi-token LLM interpretability and concept-guided text generation. It uses WordNet to generate concepts that can guide model text generation and expose biases or vulnerabilities in language models.

## Development Commands

### Environment Setup
```bash
# Install packages
pip install -U pip
pip install uv
uv sync
```

### Running Notebooks
```bash
# Run a specific notebook with output tracking
make run_notebook N=path/to/notebook.ipynb

# Example for running a specific experiment
uv run make run_notebook N=14_multilang_figstep_analysis.ipynb
```

### Model Download
Run `01_START_HERE.ipynb` to download all required models and set up NLTK data.

## Architecture & Structure

### Core Components

1. **frames/** - Main package directory containing:
   - `abstract/` - Base classes and abstract models
   - `data/` - Dataset loaders (FigStep, SafeBench multilingual data)
   - `experiments/` - Experiment implementations
   - `linalg/` - Linear algebra operations (frame operations, orthogonalization)
   - `models/` - Model wrappers and implementations
   - `nlp/` - NLP utilities (datasets, synsets, WordNet integration)
   - `representations/` - Core frame representation logic (concepts, frames, unembedding)
   - `utils/` - Utility functions (memory, plotting, settings, translation)

2. **Notebooks** - Experimental workflows numbered 00-14, each focusing on different aspects:
   - Tokenization analysis
   - Concept-word frame relationships
   - Model family comparisons
   - Guided generation experiments
   - Multilingual analysis
   - Vision-language integration

### Key Classes

- `FrameUnembeddingRepresentation` (frames/representations/frame.py) - Central class for frame-based unembedding operations
- `Concept` (frames/representations/concept.py) - Concept representation management
- `MultiLingualWordNetSynsets` (frames/nlp/) - WordNet synset handling across languages
- `BaseHuggingFaceModel` (frames/models/) - HuggingFace model wrapper

### Data Processing

The project works with:
- Vision datasets (VHD11K with 10,000+ images)
- Multilingual SafeBench data (parquet format)
- FigStep visual reasoning dataset
- Translation capabilities via googletrans

### Model Configuration

Models are configured in `models.yaml` and loaded via `frames.utils.settings.load_models()`. The project supports:
- Llama 3.1/3.2 models
- Gemma 2
- Phi 3
- Vision-language models

### Dependencies

Key dependencies include:
- PyTorch ecosystem (torch, transformers, accelerate)
- HuggingFace libraries
- NLTK for linguistic processing
- Scientific computing (numpy<2.1, pandas, numba)
- Vision processing (Pillow)
- Model optimization (bitsandbytes, llmcompressor, unsloth)