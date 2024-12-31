run_notebook:
	@if [ -z "$(N)" ]; then \
		echo "Error: N variable is not set. Usage: make run_notebook N=path/to/notebook.ipynb"; \
		exit 1; \
	fi
	@N_PATH=$$(dirname "$(N)"); \
	N_NAME=$$(basename "$(N)"); \
	OUTPUT_DIR="$$N_PATH/outputs"; \
	mkdir -p "$$OUTPUT_DIR"; \
	echo "Running notebook: $(N)"; \
	HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 papermill \
		$(N) \
		"$$OUTPUT_DIR/$$N_NAME" \
		--log-output \
		--progress-bar