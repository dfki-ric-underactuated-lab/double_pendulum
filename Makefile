PYTHON_ROOT = "src/python"
CPP_ROOT = "src/cpp/python"
DOC_ROOT = "docs/src"

default: install

install: python cpp

python:
	make -C $(PYTHON_ROOT)

cpp:
	make -C $(CPP_ROOT)

doc: 
	make -C $(DOC_ROOT) clean
	make -C $(DOC_ROOT) generate_rst
	make -C $(DOC_ROOT) html
