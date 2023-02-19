CXX = icpx

objects = mnist_pre_process.o node.o random_forest.o sample.o tree.o main.o

LIBRARY_DIRS := /usr/local/lib

LDFLAGS := $(foreach librarydir, $(LIBRARY_DIRS),-L$(librarydir)) $(PKG_CONFIG) \
$(foreach library,$(LIBRARIES),-l$(library))
CXXFLAGS := -g -std=c++17 -fsycl -pthread -O2

random_forests : $(objects)
	icpx -Wall -o random_forest $(objects) $(CXXFLAGS) $(LDFLAGS)

.PHONY : clean
clean :
	-rm random_forest $(objects)
