PYTHON_ROOT = "src/python"
CPP_ROOT = "src/cpp/python"

default: python cpp

python:
	make -C $(PYTHON_ROOT)

cpp:
	make -C $(CPP_ROOT)
