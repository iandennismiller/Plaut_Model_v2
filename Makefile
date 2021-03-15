# PMSP Makefile
# Brian Lam; Ian Dennis Miller

requirements:
	pip3 install -r requirements.txt

simulation:
	python3 main.py single
	@echo "Results are in ./results/BASE*"

hidden-layer-viz:
	python3 analysis.py hidden_similarity \
		-c BASE-S1D2O1-20210315_700 \
		-d plaut_fig18_data.csv
