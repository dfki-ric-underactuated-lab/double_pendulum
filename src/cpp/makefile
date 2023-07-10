CXX		  := g++
CXX_FLAGS := -Wall -Wextra -std=c++17 -g -O3

BIN		:= bin
SRC_PLANT		:= model/src
SRC_SIMULATOR	:= simulator/src
SRC_UTILS		:= utils/src
INCLUDE := -I/usr/local/include-I/usr/local/include/eigen3
LIB		:= lib
LIBRARIES	:= -lyaml-cpp

EXECUTABLE	:= test_sim
SRCS := $(SRC_SIMULATOR)/test_sim.cpp $(SRC_PLANT)/dp_plant.cpp $(SRC_SIMULATOR)/simulator.cpp $(SRC_UTILS)/csv_reader.cpp

all: $(BIN)/$(EXECUTABLE)

$(BIN)/$(EXECUTABLE): $(SRCS)
	@echo " Building simulation test ..."
	$(CXX) $(CXX_FLAGS) -I$(INCLUDE) -L$(LIB) $^ -o $@ $(LIBRARIES)

clean:
	@echo " Clearing..."
	-rm $(BIN)/*
