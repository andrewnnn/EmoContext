.PHONY: clean  fetch_data fetch_models doc show-help create_environment remove_environment jupyter doc

#################################################################################
# GLOBALS                                                                       #
#################################################################################
SHELL:=/bin/bash
PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PROFILE = default
PYTHON_INTERPRETER_VERSION = 3
UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S),Linux)
kernel_path = $(HOME)/.local/share/jupyter/kernels/$(PROJECT_NAME)
endif
ifeq ($(UNAME_S),Darwin)
kernel_path = $(HOME)/Library/Jupyter/kernels/$(PROJECT_NAME)
endif

include config.mk

ifeq (,$(shell which conda))
HAS_CONDA=False
else
HAS_CONDA=True
endif

PACKAGE_DIR = $(PROJECT_DIR)/src/$(subst .,/,$(PACKAGE))

INITILALISED :=$(shell if [ -f $(PACKAGE_DIR)/__init__.py ]; then echo 1; else echo 0; fi;)

PYTHON_PATH = $(shell which python$(PYTHON_INTERPRETER_VERSION))


ifeq (True,$(HAS_CONDA))
PYTHON_ENV=source activate $(PROJECT_NAME)
else
PYTHON_ENV=pyenv activate $(PROJECT_NAME)
endif

ifeq (True,$(HAS_CONDA))
E:=$(shell conda env list | grep $(PROJECT_NAME) |wc -l)
ifeq ($(strip $(E)),1)
HAS_ENV=True
else
HAS_ENV=False
endif
else
HAS_ENV=$(shell if [ -d .env ]; then echo True; else echo False; fi;)
endif

#################################################################################
# COMMANDS                                                                      #
#################################################################################

check_initialised:
	@if [ "$(INITILALISED)" -eq "1" ]; then echo Project already initialised; False; fi;

## Initialise the projet for the fist time
init_project: check_initialised
	mkdir -p $(PACKAGE_DIR)
	touch $(PACKAGE_DIR)/__init__.py

models := $(patsubst %.txt,%.d,$(shell find . -name "models.txt"))
data := $(patsubst %.txt,%.d, $(shell find ./data -name "datasets.txt"))

%.d:%.txt
	@echo Processing model file: $<	
	@while read -r line;	do \
		echo "Processing $$line"; \
		if [[ $${line:0:1} == "#" ]]; then \
			echo "File ignored"; \
			continue; \
		fi; \
		if [[ $$(basename $$line) = '*' ]]; then \
			echo "All files: Downloading from $$line"; \
			aws s3 cp $$(dirname $$line)/ $$(dirname $<)/ --recursive; \
		else \
			line=$$(echo $$line | tr -s [:blank:]); \
			file=$$(echo $$line| cut -d' ' -f2); \
			src=$$(echo $$line| cut -d' ' -f1); \
			if [[ $$file == s3* ]]; then \
				file=$$(basename $$file); \
			fi; \
			if [ -e ./$$(dirname $<)/$$file ]; then \
				echo "$$(basename $$file): File exists"; \
			else \
				echo "$$(basename $$file): Downloading from $$src"; \
				aws s3 cp $$src $$(dirname $<)/$$file ; \
			fi; \
		fi; \
	done< ./$< 
	@echo ""

## Download required models
fetch_models: $(models)
	@echo "Done"

## Download data
fetch_data: $(data)
	@echo "Done"

## Clean python files
clean:
	find  . -name "__pycache__" -not -path "*.env*" -prune -exec rm -rf {} \;
	$(eval out!=$(MAKE) -C docs clean 2>&1)
	$(eval has_sphinx:=$(shell echo $(out) | grep "No module named sphinx" | wc -l))
ifeq (True,$(HAS_ENV))
	$(if $(findstring 1,$(has_sphinx)), bash -c "$(PYTHON_ENV); $(MAKE) -C docs clean")
else
	@printf "Run make create_environment to create the envirnment. \n"
endif

	
## Remove all the data
clean_all: clean
	@find $(DATA_DIR) -depth 1 -not -name "datasets.txt" -delete
	@find $(MODEL_DIR) -depth 1 -not -name "models.txt" -delete


## Create html documentation
doc:
	@if [ -e html ]; then rm -rf html; fi
	$(eval out!=$(MAKE) -C docs html 2>&1)
	$(eval has_sphinx:=$(shell echo $(out) | grep "No module named sphinx" | wc -l))
	$(if $(findstring 0,$(has_sphinx)), cp -r docs/build/html ./; open html/index.html)
ifeq (True,$(HAS_ENV))
	$(if $(findstring 1,$(has_sphinx)), bash -c "$(PYTHON_ENV); $(MAKE) -C docs html")
	cp -r docs/build/html ./
	open html/index.html
else
	@printf "Run make create_environment to create the envirnment. \n"
endif



	
## Set up python interpreter environment
create_environment:
ifeq (True,$(HAS_CONDA))
	@echo ">>> Detected conda, creating conda environment."
	conda create --name $(PROJECT_NAME) python=$(PYTHON_INTERPRETER_VERSION)
	@printf ">>> New conda env created. Activate with:\nsource activate $(PROJECT_NAME)"
	@echo ">>> Install requirements"
	sort requirements.txt > requirements_srt.txt
	bash -c "$(PYTHON_ENV); conda install --dry-run  --file requirements.txt --json | jq -a -r ".packages[]" | sort > not_in_conda.txt"
	$(eval IN_CONDA:=$(shell comm -3 requirements_srt.txt not_in_conda.txt))
	bash -c "$(PYTHON_ENV); conda install $(IN_CONDA) sphinx numpydoc jupyter"
	bash -c "$(PYTHON_ENV); pip install -r not_in_conda.txt"
	rm -f not_in_conda.txt requirements_srt.txt
	@echo ">>> Install Jupyter kernal"
	bash -c "$(PYTHON_ENV); python -m ipykernel install --user --name=$(PROJECT_NAME) --display-name='Python ($(PROJECT_NAME))'"	
else
	@if [ $(PYTHON_PATH) ] ; \
	then  printf "Has python $(PYTHON_PATH)\n"; \
	zsh -c "pyenv virtualenv $(PROJECT_NAME)"; \
	printf ">>> New virtualenv created. Activate with:\n$(PYTHON_ENV)\n"; \
	echo ">>> Install requirements" ; \
	zsh -c "$(PYTHON_ENV); pip install -r requirements.txt; pip install sphinx numpydoc jupyter" ; \
	echo ">>> Install Jupyter kernal" ; \
	zsh -c "$(PYTHON_ENV); python -m ipykernel install --user --name=$(PROJECT_NAME) --display-name='Python ($(PROJECT_NAME))'" ; \
	else printf "python$(PYTHON_INTERPRETER_VERSION) not found" ; \
	fi; 
endif

## Remove python interpreter environment
remove_environment:
	@printf ">>> Remove Jupyter kernal at: \n $(kernel_path)\n"
	rm -fr $(kernel_path)
ifeq (True,$(HAS_CONDA))
	conda env remove --name $(PROJECT_NAME)
else
	rm -fr .env
endif
	@echo ">>> Removed the envirnment and jupyter kernal."


## Run jupyter notebook using the environment
jupyter:
	zsh -c "$(PYTHON_ENV); jupyter notebook"

## Run jupyter notebook using the environment
bash:
	bash -c "bash ;$(PYTHON_ENV)"

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################



#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := show-help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>

show-help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
