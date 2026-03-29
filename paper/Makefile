TECTONIC ?= $(HOME)/.local/bin/tectonic

.PHONY: all clean

all: main.pdf

main.pdf: main.tex references.bib
	$(TECTONIC) main.tex

clean:
	rm -f main.pdf main.aux main.log main.out main.bbl main.blg
