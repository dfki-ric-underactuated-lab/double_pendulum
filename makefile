PYTHON_ROOT = "src/python"
CPP_ROOT = "src/cpp/python"

default: cpp python

python:
	make -C $(PYTHON_ROOT)

cpp:
	make -C $(CPP_ROOT)
