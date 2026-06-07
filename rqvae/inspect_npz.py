import numpy as np
import pandas as pd

path = r'D:\stock\stock_sid\data\raw\000001_SZ.npz'
data = np.load(path, allow_pickle=True)

print('keys      :', list(data.keys()))
print('ts_code   :', str(data['ts_code']))
print('vectors   :', data['vectors'].shape, data['vectors'].dtype)
print('labels    :', len(data['labels']), 'weeks')
print('label range:', data['labels'][0], '~', data['labels'][-1])

INDICATORS = ['open', 'high', 'low', 'close', 'vol', 'amount', 'pct_chg']

print('\n--- 前 3 个样本 ---')
for i in range(min(3, len(data['vectors']))):
    label = data['labels'][i]
    vec = data['vectors'][i].reshape(5, 7)
    df = pd.DataFrame(vec, columns=INDICATORS, index=[f'Day{j+1}' for j in range(5)])
    print(f'\n{label}:')
    print(df.to_string(float_format='%.4f'))
