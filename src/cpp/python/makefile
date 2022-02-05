LIB_DIR = "../controllers/ilqr/obj"
LIB_DIR_ROOT = "../controllers/ilqr"

default: cppilqr

cppilqr : setup.py cppilqr.pyx $(LIB_DIR)/libilqr.a
	#python3 setup.py build_ext --inplace && rm -f cppilqr.cpp && rm -Rf build && mv cppilqr.cpython-*.so $(python -c 'import site; print(site.getsitepackages()[0]+"/")')
	#CC=clang++-9 python3 setup.py build_ext --inplace  && rm -f cppilqr.cpp && rm -Rf build && mv cppilqr.cpython-*.so $(python -c 'import site; print(site.getsitepackages()[0]+"/")')
	CC=g++ python3 setup.py build_ext --inplace  && rm -f cppilqr.cpp && rm -Rf build && bash move_pylib.sh
	#CC=g++ python3 setup.py build_ext --inplace  && rm -f cppilqr.cpp && rm -Rf build && mv cppilqr.cpython-*.so $(PYTHON_SITE_PACKAGE_DIR)

$(LIB_DIR)/libilqr.a:
	make -C $(LIB_DIR_ROOT) obj/libilqr.a

clean:
	rm *.so
	#rm cilqr.cpp
