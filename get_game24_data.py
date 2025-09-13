import pandas as pd


def extract_csv_ranges(input_file, output_file, ranges):
    """
    从CSV文件中提取指定范围的数据并重新编写索引

    Args:
        input_file: 输入CSV文件路径
        output_file: 输出CSV文件路径
        ranges: 要提取的范围列表，每个范围是(start, end)的元组
    """
    # 读取CSV文件
    df = pd.read_csv(input_file)

    # 存储提取的数据
    extracted_data = []

    # 提取指定范围的数据
    for start, end in ranges:
        # 过滤出指定范围的数据（包含start和end）
        range_data = df[(df["Rank"] >= start) & (df["Rank"] <= end)]
        extracted_data.append(range_data)

    # 合并所有提取的数据
    combined_data = pd.concat(extracted_data, ignore_index=True)

    # 重新编写索引（从1开始）
    combined_data["Rank"] = range(1, len(combined_data) + 1)

    # 保存到新的CSV文件
    combined_data.to_csv(output_file, index=False)

    print(f"提取完成！")
    print(f"原始数据行数: {len(df)}")
    print(f"提取数据行数: {len(combined_data)}")
    print(f"输出文件: {output_file}")

    return combined_data


# 使用脚本
if __name__ == "__main__":
    # 定义要提取的范围
    ranges_to_extract = [(1311, 1362)]  # 100-199  # 500-599  # 900-999

    # 输入和输出文件路径
    input_file = "src/tot/data/24/24.csv"
    output_file = "src/tot/data/24/24_hardest.csv"

    # 执行提取
    extracted_df = extract_csv_ranges(input_file, output_file, ranges_to_extract)

    # 显示提取结果的统计信息
    print(f"\n提取的数据范围:")
    for start, end in ranges_to_extract:
        count = len(
            extracted_df[
                extracted_df.index
                < sum(end - start + 1 for s, e in ranges_to_extract[: ranges_to_extract.index((start, end)) + 1])
                - (end - start)
            ]
        )
        print(f"  原始范围 {start}-{end}: 提取了 {end-start+1} 行数据")

    # 显示前几行和后几行数据预览
    print(f"\n提取数据预览:")
    print("前5行:")
    print(extracted_df.head())
    print(f"\n后5行:")
    print(extracted_df.tail())
