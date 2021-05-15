# PMSP Makefile
# Brian Lam; Ian Dennis Miller

requirements:
	pip3 install -r requirements.txt

simulation:
	python3 main.py single
	@echo "Results are in ./results/BASE*"

hidden-layer-fig18:
	python3 analysis.py hidden_similarity_lens \
		-c LENS-20210314-1 \
		-d plaut_fig18_data.csv
