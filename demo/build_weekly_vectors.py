"""
构建周级别量价向量样本（含归一化）
- 7个量价指标：open, high, low, close, vol, amount, pct_chg
- 只保留周内恰好有5个交易日的样本
- 每个样本形状：(5, 7)，即 5天 × 7指标

归一化策略（逐样本内部处理，确保跨股票/跨时期可比）：
  OHLC     : (price_t / close_day1) - 1，相对第一天收盘价的涨跌幅
  vol/amount: log1p 后做周内 z-score
  pct_chg  : 直接使用（本身已是日涨跌幅%，无需再处理）
"""
from datetime import datetime, timedelta
import chinadata.ca_data as ts
import pandas as pd
import numpy as np

TOKEN = 'ma122c996ccaac10157d709313800051983'
ts.set_token(TOKEN)
pro = ts.pro_api(TOKEN)

TS_CODE = '000001.SZ'
OHLC_COLS    = ['open', 'high', 'low', 'close']
VOL_COLS     = ['vol', 'amount']
KEEP_COLS    = ['pct_chg']
INDICATORS   = OHLC_COLS + VOL_COLS + KEEP_COLS  # 列顺序固定

end_date   = datetime.today().strftime('%Y%m%d')
start_date = (datetime.today() - timedelta(days=365)).strftime('%Y%m%d')

print(f"获取 {TS_CODE} 日线数据 {start_date} ~ {end_date}...")
df = pro.daily(ts_code=TS_CODE, start_date=start_date, end_date=end_date)
df = df.sort_values('trade_date').reset_index(drop=True)
df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')

# 按 ISO 自然周分组
iso = df['trade_date'].dt.isocalendar()
df['year_week'] = iso.year.astype(str) + '-W' + iso.week.astype(str).str.zfill(2)

# 只保留恰好5个交易日的周
week_sizes   = df.groupby('year_week').size()
valid_weeks  = week_sizes[week_sizes == 5].index
df_filtered  = df[df['year_week'].isin(valid_weeks)].copy()

total_weeks  = len(week_sizes)
kept_weeks   = len(valid_weeks)
print(f"\n总周数: {total_weeks}，保留: {kept_weeks}，丢弃（不足5天）: {total_weeks - kept_weeks}")


def normalize_week(group: pd.DataFrame) -> np.ndarray:
    """对单周 (5, 7) 数据做归一化，返回归一化后的 numpy 数组"""
    g = group[INDICATORS].values.astype(float).copy()  # (5, 7)

    col = {c: i for i, c in enumerate(INDICATORS)}

    # OHLC → 相对第一天收盘价的涨跌幅
    close0 = g[0, col['close']]
    for c in OHLC_COLS:
        g[:, col[c]] = g[:, col[c]] / close0 - 1.0

    # vol / amount → log1p 后周内 z-score
    for c in VOL_COLS:
        vals = np.log1p(g[:, col[c]])
        mu, sigma = vals.mean(), vals.std()
        g[:, col[c]] = (vals - mu) / (sigma + 1e-8)

    # pct_chg 保持原值（已是%，无需额外处理）

    return g  # (5, 7)


# 构建样本
samples     = []
week_labels = []

for week, group in df_filtered.groupby('year_week'):
    group = group.sort_values('trade_date')
    vec   = normalize_week(group)
    samples.append(vec)
    week_labels.append(week)

samples = np.array(samples)  # (N, 5, 7)

print(f"\n样本矩阵 shape: {samples.shape}  → (样本数, 交易日, 指标数)")
print(f"指标顺序: {INDICATORS}")

# 统计归一化后的值域，验证合理性
print(f"\n{'指标':<10} {'均值':>8} {'标准差':>8} {'最小值':>8} {'最大值':>8}")
print('-' * 46)
flat = samples.reshape(-1, len(INDICATORS))
for i, name in enumerate(INDICATORS):
    col = flat[:, i]
    print(f"{name:<10} {col.mean():>8.4f} {col.std():>8.4f} {col.min():>8.4f} {col.max():>8.4f}")

print(f"\n--- 第一个样本 ({week_labels[0]}) ---")
print(pd.DataFrame(samples[0], columns=INDICATORS,
                   index=[f"Day{i+1}" for i in range(5)]).to_string(float_format='%.4f'))

print(f"\n--- 最后一个样本 ({week_labels[-1]}) ---")
print(pd.DataFrame(samples[-1], columns=INDICATORS,
                   index=[f"Day{i+1}" for i in range(5)]).to_string(float_format='%.4f'))

# flatten: (N, 5, 7) → (N, 35)，这才是喂给 RQ-VAE 的 embedding
samples_flat = samples.reshape(len(samples), -1)

# 保存
np.save('weekly_vectors.npy', samples_flat)
np.save('week_labels.npy', np.array(week_labels))
print(f"\n已保存 weekly_vectors.npy  shape={samples_flat.shape}  (每个样本是 35 维向量)")
