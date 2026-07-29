# Weekly report build. `make figures` needs network (pulls run artifacts from HF, cached);
# `make` alone rebuilds the PDF offline from the committed figures/ + generated/.
all: report.pdf

figures:
	python3 make_figures.py

report.pdf: main.tex ../template/weeklyreport.sty generated/numbers.tex generated/runs_table.tex
	TEXINPUTS=../template: latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex
	cp build/main.pdf report.pdf

clean:
	rm -rf build

.PHONY: all figures clean
