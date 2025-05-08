# CUDA Reduce Operators (Warp, Block, Grid Level)

本项目实现了三种层级的 GPU 并行规约（Reduce）算子，并支持通用操作（如加法、乘法、最大值等）的模板扩展。

---

## 📌 项目功能

通过 CUDA 编写并优化了以下三类规约算子：

| 阶段 | 名称               | 特点说明 |
|------|--------------------|----------|
| ✅ 第一阶段 | Warp-Level Reduce        | 使用 `__shfl_xor_sync` 实现 32 线程 warp 内规约。|
| ✅ 第二阶段 | Block-Level Reduce       | 使用共享内存支持任意线程数（<=1024）的 block 内规约。|
| ✅ 第三阶段 | Grid-Level Reduce        | 使用两轮 block 级规约，实现多 block 的全局规约。|

---

## 📁 项目结构

```bash
cuda-reduce/
├── include/
│   └── reduce_kernels.h       # 所有模板 reduce 函数的声明与实现（必须 header-only）
├── kernels/
│   ├── warp_reduce.cu         # 第一阶段：warp 级规约
│   ├── block_reduce.cu        # 第二阶段：block 级规约
│   └── grid_reduce.cu         # 第三阶段：grid 级规约（支持模板操作）
├── main.cu                    # 示例入口，运行所有 reduce 方法并打印结果
├── Makefile                   # 一键构建项目
└── reduce_test                # 编译生成的可执行文件
