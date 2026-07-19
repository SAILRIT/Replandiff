# Build the paper.  `make` -> main.pdf ;  `make clean` -> remove build artifacts
main.pdf: main.tex refs.bib tab_main.tex tab_family.tex neuripsstyle.sty
	latexmk -pdf -interaction=nonstopmode main.tex

clean:
	latexmk -C
	rm -f *.aux *.bbl *.blg *.log *.out *.fls *.fdb_latexmk

.PHONY: clean
