PYTHON_ROOT = "src/python"
CPP_ROOT = "src/cpp/python"

default: dp_python dp_cpp

dp_python:
	make -C $(PYTHON_ROOT)

dp_cpp:
	make -C $(CPP_ROOT)
