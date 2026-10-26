"""合并公寓用电量和天气数据，按分钟对齐并保存为统一的数据集。"""

from pathlib import Path

import pandas as pd


def merge_power_weather(
    power_dir="data/2016",
    weather_file="data/apartment2016.csv",
    output_path="data/power_weather.csv",
    cut_end="2016-12-15 19:59:00",
):
    """合并 114 个公寓的数据，保留时间交集并将缺失值填为 0。

    天气 Unix 时间戳按秒解析，并向前填充到分钟频率。
    重复的用电时间戳取均值；最终仅保留不晚于 cut_end 的记录。
    """
    power_dir = Path(power_dir)
    apt_series = {}
    for i in range(1, 115):
        file_path = power_dir / f"Apt{i}_2016.csv"
        if not file_path.exists():
            print(f"⚠️ 文件不存在: {file_path}")
            continue

        df = pd.read_csv(
            file_path,
            names=["timestamp", "value"],
            header=None,
            parse_dates=["timestamp"],
            dtype={"value": "float32"},
        )
        series = df.groupby("timestamp")["value"].mean().sort_index()
        if series.empty:
            print(f"⚠️ 文件没有有效数据: {file_path}")
            continue
        apt_series[f"Apt_{i}"] = series

    if not apt_series:
        raise ValueError(f"未找到有效的公寓用电数据: {power_dir}")

    global_start = min(series.index.min() for series in apt_series.values())
    global_end = max(series.index.max() for series in apt_series.values())
    print(f"✅ 公寓数据读取完成，共 {len(apt_series)} 个")
    print(f"全局时间范围：{global_start}  ~  {global_end}")

    weather_df = pd.read_csv(weather_file)
    weather_df["time"] = pd.to_datetime(weather_df["time"], unit="s")
    weather_df = weather_df.set_index("time").sort_index()
    if weather_df.empty:
        raise ValueError(f"天气数据为空: {weather_file}")
    weather_df = weather_df.drop(columns=["cloudCover"], errors="ignore")

    weather_start = weather_df.index.min()
    weather_end = weather_df.index.max()
    print(f"🌦 天气时间范围：{weather_start}  ~  {weather_end}")

    final_start = max(global_start, weather_start)
    final_end = min(global_end, weather_end)
    print(f"交集时间范围：{final_start}  ~  {final_end}")
    time_index = pd.date_range(start=final_start, end=final_end, freq="1min")

    # 只在最终交集上对齐，避免对整个用电时间范围重复计算。
    power_df = pd.DataFrame(apt_series, index=time_index)
    weather_df = weather_df.resample("1min").ffill().reindex(time_index).ffill()
    final_df = pd.concat([power_df, weather_df], axis=1)
    final_df = final_df.loc[final_df.index <= pd.Timestamp(cut_end)]

    print("\n🔍 缺失值统计（处理前）:")
    print(final_df.isna().sum())
    final_df = final_df.fillna(0)
    final_df = final_df.reset_index().rename(columns={"index": "Datetime"})
    final_df.to_csv(output_path, index=False)

    print(f"✅ 已生成合并文件: {output_path}")
    print(f"最终数据维度: {final_df.shape}")
    print(f"最终时间范围: {final_df['Datetime'].min()}  ~  {final_df['Datetime'].max()}")
    return final_df


if __name__ == "__main__":
    merge_power_weather()
