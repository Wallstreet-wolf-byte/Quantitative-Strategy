"""
CSV 编码转换脚本: UTF-8 → GBK
用于 VeighNa Studio 数据管理导入 CSV

使用方式:
    1. 把这个脚本放到任意位置
    2. 修改下面的 INPUT_PATH 为你的 CSV 文件路径
    3. 运行: python convert_csv_to_gbk.py
    4. 会在同目录生成一个 _gbk 后缀的副本，原文件不动
"""

import pandas as pd
from pathlib import Path

# ============ 配置 ============
# 你的原始 CSV 文件路径（UTF-8 编码）
INPUT_PATH = r"D:\量化交易员\数据\HSI_2021-01-01_2026-08-29\HSI.csv"

# 输出文件路径（默认在原文件同目录，加 _gbk 后缀）
# 留空的话自动生成: 原文件名_gbk.csv
OUTPUT_PATH = ""
# ==============================


def convert_csv_to_gbk(input_path: str, output_path: str = ""):
    """将 UTF-8 CSV 转换为 GBK CSV"""
    input_file = Path(input_path)

    if not input_file.exists():
        print(f"错误: 找不到文件 {input_path}")
        return False

    # 自动生成输出路径
    if not output_path:
        output_path = str(input_file.with_name(f"{input_file.stem}_gbk.csv"))

    output_file = Path(output_path)
    print(f"正在读取: {input_file}")

    # 尝试读取 CSV（优先 UTF-8，失败再试 utf-8-sig）
    try:
        df = pd.read_csv(input_file, encoding="utf-8")
    except UnicodeDecodeError:
        print("UTF-8 读取失败，尝试 utf-8-sig...")
        df = pd.read_csv(input_file, encoding="utf-8-sig")

    print(f"  行数: {len(df):,}")
    print(f"  列名: {list(df.columns)}")

    # 保存为 GBK 编码
    print(f"\n正在保存: {output_file}")
    df.to_csv(output_file, encoding="gbk", index=False)

    # 验证一下
    print("\n验证输出文件...")
    df_check = pd.read_csv(output_file, encoding="gbk", nrows=5)
    print(f"  前5行读取成功, 列名: {list(df_check.columns)}")

    print(f"\n转换完成! ✓")
    print(f"  原文件: {input_file}")
    print(f"  GBK文件: {output_file}")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("CSV 编码转换: UTF-8 → GBK")
    print("=" * 60)
    print()

    success = convert_csv_to_gbk(INPUT_PATH, OUTPUT_PATH)

    if success:
        print("\n下一步: 在 VeighNa Trader 数据管理中导入 _gbk.csv 文件")
    else:
        print("\n转换失败，请检查文件路径是否正确")
