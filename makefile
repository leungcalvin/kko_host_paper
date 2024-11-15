# Use this makefile to 1) download the right sheet, 2) run the authorship wrangler.
# Then put \include{authors.tex} into your main.tex and call it a day :)
# Usage: make authors
# Author: Calvin Leung

PATH_TO_PAPER='/arc/home/calvin/kko_host_paper/'
PATH_TO_AUTHORSHIP_REPO='/arc/home/calvin/authorship'
authors:
	curl -L "https://docs.google.com/spreadsheets/d/1rJuFHVxd1Qcnvw5yLEXEtYH5M8SWB46X0Pmr0QiXLBs/export?gid=0&format=tsv" > $(PATH_TO_PAPER)/authors.tsv
	curl -L "https://docs.google.com/spreadsheets/d/1rJuFHVxd1Qcnvw5yLEXEtYH5M8SWB46X0Pmr0QiXLBs/export?gid=2085366068&format=tsv" > $(PATH_TO_PAPER)/affils.tsv
	python $(PATH_TO_AUTHORSHIP_REPO)/authors.py $(PATH_TO_PAPER)/authors.tsv $(PATH_TO_PAPER)/affils.tsv --discard 1 > $(PATH_TO_PAPER)/authors.tex
