# Vision Language Model Interpretability with Concept Guided Decoding

**ICIP 2025** • *Anchorage, Alaska*

[![Paper](https://img.shields.io/badge/Paper-PDF-blue?logo=adobeacrobatreader)](https://cmsworkshops.com/ICIP2025/view_paper.php?PaperNum=1443)
[![Code](https://img.shields.io/badge/Github-Code-black?logo=github)](https://github.com/phvv-me/icip2025)
[![Dataset](https://img.shields.io/badge/Dataset-Excel-green?logo=microsoftexcel)](https://github.com/phvv-me/icip2025/blob/main/Translated%20SafeBench%20verified%202025-01-12.xlsx)
[![IEEE Xplore](https://img.shields.io/badge/IEEE_Xplore-Publication-purple?logo=ieee)](https://ieeexplore.ieee.org/document/11084299)

## Installation

```shell
# Clone repository
git clone https://github.com/phvv-me/icip2025
cd icip2025

# Install dependencies
pip install -U pip
pip install uv
uv sync
```

## Reproducing Results

1. **Start here**: Run `01_START_HERE.ipynb` to download models and set up NLTK data
2. **View results**: Check `00_results.ipynb` for analysis overview  
3. **Run experiments**: Execute notebooks 02-14 for specific analyses:
   - **02-04**: Frame analysis and theory
   - **05-09**: Guided generation experiments  
   - **10-12**: Vision-language integration
   - **13-14**: Multilingual vulnerability analysis

Use the makefile for convenient notebook execution:

```shell
make run_notebook N=path/to/notebook.ipynb
```

## Citation

```bibtex
@inproceedings{valois2025vision,
  title={Vision Language Model Interpretability with Concept Guided Decoding},
  author={Valois, Pedro H. V. and Satav, Dipesh and de Campos, Rodrigo A. P. and 
          Pratamasunu, Gulpi Q. O. and Fukui, Kazuhiro},
  booktitle={2025 IEEE International Conference on Image Processing (ICIP)},
  pages={397--402},
  year={2025},
  organization={IEEE},
  doi={10.1109/ICIP55913.2025.11084299}
}
```

## License

MIT License
