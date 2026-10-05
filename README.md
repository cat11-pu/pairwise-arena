# arena

在一个固定长度的字节缓冲区上模拟堆的小型分配器：用空闲链表管理可分配区间，
支持首次适配与最佳适配、按 align 对齐、分割与相邻空闲区间合并、碎片统计，
以及带边界检查的读写。

## 目录

- arena/core.py 分配器实现
- tests/test_core.py 行为测试

## 运行测试

在项目根目录执行：

    python3 -m unittest discover -s tests -v

只使用 Python 标准库，不需要安装任何依赖。
