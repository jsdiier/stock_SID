"""
统计近一年交易日按自然周分组后，每周交易日数量的分布
"""
from datetime import datetime, timedelta
import chinadata.ca_data as ts
import pandas as pd

TOKEN = 'ma122c996ccaac10157d709313800051983'
ts.set_token(TOKEN)
pro = ts.pro_api(TOKEN)

end_date = datetime.today().strftime('%Y%m%d')
start_date = (datetime.today() - timedelta(days=365)).strftime('%Y%m%d')

print(f"获取交易日历 {start_date} ~ {end_date}...")
cal = pro.trade_cal(exchange='', start_date=start_date, end_date=end_date)
trade_days = cal[cal['is_open'] == 1][['cal_date']].copy()
trade_days['cal_date'] = pd.to_datetime(trade_days['cal_date'], format='%Y%m%d')

# 按自然周（ISO week）分组，统计每周交易日数
trade_days['year_week'] = trade_days['cal_date'].dt.isocalendar().year.astype(str) \
    + '-W' + trade_days['cal_date'].dt.isocalendar().week.astype(str).str.zfill(2)

week_counts = trade_days.groupby('year_week').size().reset_index(name='days')

dist = week_counts['days'].value_counts().sort_index()
total_weeks = len(week_counts)

print(f"\n共 {total_weeks} 个自然周，总交易日 {len(trade_days)} 天\n")
print(f"{'每周交易日数':>10}  {'周数':>6}  {'占比':>8}")
print('-' * 30)
for days, count in dist.items():
    print(f"{days:>10}  {count:>6}  {count/total_weeks*100:>7.1f}%")

print(f"\n各周交易日数明细（days < 5 的周）：")
short_weeks = week_counts[week_counts['days'] < 5].sort_values('year_week')
print(short_weeks.to_string(index=False))
