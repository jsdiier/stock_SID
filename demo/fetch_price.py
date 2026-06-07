"""
示例：用 tushare (chinadata) 获取某只股票近一年的日线价格数据
"""
import sys
import os
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'stock'))

import chinadata.ca_data as ts

TOKEN = 'ma122c996ccaac10157d709313800051983'
ts.set_token(TOKEN)
pro = ts.pro_api(TOKEN)

# 目标股票：平安银行
TS_CODE = '000001.SZ'

end_date = datetime.today().strftime('%Y%m%d')
start_date = (datetime.today() - timedelta(days=365)).strftime('%Y%m%d')

print(f"获取 {TS_CODE} 从 {start_date} 到 {end_date} 的日线数据...")

df = pro.daily(ts_code=TS_CODE, start_date=start_date, end_date=end_date)

if df is None or df.empty:
    print("未获取到数据")
else:
    df = df.sort_values('trade_date').reset_index(drop=True)
    print(f"\n共 {len(df)} 个交易日\n")
    print(df[['trade_date', 'open', 'high', 'low', 'close', 'vol', 'pct_chg']].to_string(index=False))
