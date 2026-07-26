# ============================================================
# 配置
# ============================================================
NVCC      = nvcc
ARCH      = sm_89          # RTX 4090；其他卡改这里
NVCCFLAGS = -O3 -arch=$(ARCH) -std=c++17

TARGET    = test_reduce
TEST_SRC  = test_reduce.cpp

# ============================================================
# 自动检测最新版本：找 reduce_vN.cu 中 N 最大的那个
# ============================================================
LATEST_SRC := $(lastword $(sort $(wildcard reduce_v*.cu)))
ifeq ($(LATEST_SRC),)
  $(error "没有找到任何 reduce_vN.cu 文件")
endif
LATEST_VER := $(patsubst reduce_v%.cu,%,$(LATEST_SRC))

# ============================================================
# 编译规则
# ============================================================
$(TARGET): $(LATEST_SRC) $(TEST_SRC) reduce.h
	@echo ">>> 使用版本: $(LATEST_SRC)"
	$(NVCC) $(NVCCFLAGS) $(LATEST_SRC) $(TEST_SRC) -o $(TARGET)

# ============================================================
# 快捷命令
# ============================================================
run: $(TARGET)
	./$(TARGET)

bench: $(TARGET)
	./$(TARGET)

no-bench: $(TARGET)
	./$(TARGET) --no-bench

# 编译并运行指定版本，例如: make ver V=1
ver: reduce_v$(V).cu $(TEST_SRC) reduce.h
	@echo ">>> 指定版本: reduce_v$(V).cu"
	$(NVCC) $(NVCCFLAGS) reduce_v$(V).cu $(TEST_SRC) -o $(TARGET)_v$(V)
	./$(TARGET)_v$(V) --no-bench

# 编译所有版本的二进制（不运行）
all-versions: $(TEST_SRC) reduce.h
	@for f in $(sort $(wildcard reduce_v*.cu)); do \
	    v=$${f#reduce_v}; v=$${v%.cu}; \
	    echo ">>> 编译 $$f -> $(TARGET)_v$$v"; \
	    $(NVCC) $(NVCCFLAGS) $$f $(TEST_SRC) -o $(TARGET)_v$$v; \
	done

# 对所有版本跑 bench 并输出对比
compare: all-versions
	@for f in $(sort $(wildcard reduce_v*.cu)); do \
	    v=$${f#reduce_v}; v=$${v%.cu}; \
	    echo ""; echo "========== v$$v =========="; \
	    ./$(TARGET)_v$$v; \
	done

clean:
	rm -f $(TARGET) $(TARGET)_v*

.PHONY: run bench no-bench ver all-versions compare clean
