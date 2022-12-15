CXX       := g++
#CXX       := clang++-9
CXX_FLAGS := -Wall -Wextra -std=c++17 -g -O3

BIN     := bin
OBJ := obj
SRC_PLANT       := ../../model/src
SRC_SIMULATOR   := ../../simulator/src
SRC_ILQR        := src
SRC_UTILS       := ../../utils/src
INCLUDE := -I/usr/include/eigen3 -I/usr/local/include
LIB     := lib
LIBRARIES   := -lyaml-cpp 

EXECUTABLE1 := compute_ilqr
EXECUTABLE3 := mpc_ilqr
SRCS1 := $(SRC_ILQR)/main_ilqr.cpp $(SRC_PLANT)/dp_plant.cpp $(SRC_SIMULATOR)/simulator.cpp $(SRC_ILQR)/ilqr.cpp
SRCS2 := $(SRC_PLANT)/dp_plant.cpp $(SRC_SIMULATOR)/simulator.cpp $(SRC_ILQR)/ilqr.cpp $(SRC_ILQR)/ilqr_mpc.cpp
SRCS3 := $(SRC_ILQR)/main_ilqr_mpc.cpp $(SRC_PLANT)/dp_plant.cpp $(SRC_SIMULATOR)/simulator.cpp $(SRC_ILQR)/ilqr.cpp $(SRC_ILQR)/ilqr_mpc.cpp $(SRC_UTILS)/csv_reader.cpp

all: $(BIN)/$(EXECUTABLE1) $(OBJ)/libilqr.a $(BIN)/$(EXECUTABLE3)

$(BIN)/$(EXECUTABLE1): $(SRCS1)
	@echo " Building ilqr ..."
	$(CXX) $(CXX_FLAGS) -I$(INCLUDE) -L$(LIB) $^ -o $@ $(LIBRARIES)

$(OBJ)/libilqr.a: $(OBJ)/libilqr.o
	ar rcs $@ $^

$(OBJ)/libilqr.o: $(SRCS2)
	$(CXX) $(CXX_FLAGS) -I$(INCLUDE) -L$(LIB) -c $< -o $@ $(LIBRARIES)

$(BIN)/$(EXECUTABLE3): $(SRCS3)
	@echo " Building ilqr ..."
	$(CXX) $(CXX_FLAGS) -I$(INCLUDE) -L$(LIB) $^ -o $@ $(LIBRARIES)

clean:
	@echo " Clearing..."
	-rm $(BIN)/*
	-rm $(OBJ)/*

