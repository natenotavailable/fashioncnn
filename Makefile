CC = g++
CFLAGS = -Wall -std=c++23
INCLUDE = include
SRC = src
BUILD = build

TARGET  = $(BUILD)/main

SRCS = \
	$(SRC)/main.cpp \
	$(SRC)/loading.cpp \
	$(SRC)/tensor4f.cpp

OBJS = $(SRCS:$(SRC)/%.cpp=$(BUILD)/%.o)

all: $(TARGET)


$(TARGET): $(OBJS) | $(BUILD)
	$(CC) $(CFLAGS) -o $@ $(OBJS)

$(BUILD)/%.o: $(SRC)/%.cpp | $(BUILD)
	$(CC) $(CFLAGS) -I$(INCLUDE) -c $< -o $@

$(BUILD):
	mkdir -p $(BUILD)


clean:
	rm -rf $(BUILD)

test: $(TARGET)
	./$(TARGET)
