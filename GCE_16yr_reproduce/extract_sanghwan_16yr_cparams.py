# extract_sanghwan_16yr_cparams.py
# Sanghwan 16yr 노트북의 셀 59 stream 출력에서 14 model × 17 bin × 5 param 추출
# 파라미터 순서 (Sanghwan 코드 likelihood_constrained 함수 기준):
#   idx 0=c_gas(pion_bremss), 1=c_ics, 2=c_gce, 3=c_bub, 4=c_iso

import json, re, os
import numpy as np

NOTEBOOK = './GCE_16yr_data/GC_analysis-60x60-models_front_back_16yr.ipynb'
CELL_INDEX = 59
MODEL_LIST = ['X', 'XLIX', 'I', 'IV', 'V', 'VI', 'VII', 'IX',
              'XV', 'XLI', 'XLVII', 'XLVIII', 'L', 'LII']
N_BINS = 17

with open(NOTEBOOK) as f:
    nb = json.load(f)

# 셀 59의 모든 stream 출력을 시간 순으로 합침
stream = []
for o in nb['cells'][CELL_INDEX].get('outputs', []):
    if o.get('output_type') == 'stream':
        t = o.get('text', '')
        if isinstance(t, list):
            t = ''.join(t)
        stream.append(t)
text = ''.join(stream)

# 한 줄에 걸쳐 print되거나 두 줄에 나눠 print된 5-벡터를 robust하게 파싱
# (4번째 bin 이후 scientific notation으로 줄바꿈 되는 경우 있음)
text_join = re.sub(r'\n\s+', ' ', text)  # 줄바꿈된 array를 한 줄로
lines = text_join.split('\n')

# 'Max position: [...]'  →  MAP 5-vec
map_pat = re.compile(
    r'Max position:\s*\[\s*([-\deE\.\+\s]+?)\s*\]'
)
# 그 다음 줄의 'std [...]'
std_pat = re.compile(r'std\s*\[\s*([-\deE\.\+\s]+?)\s*\]')

records = []  # (model_name, bin_idx, [c_gas, c_ics, c_gce, c_bub, c_iso], [stds])
bin_counter = 0
model_counter = 0

for i, ln in enumerate(lines):
    m = map_pat.search(ln)
    if m:
        vals = np.fromstring(m.group(1), sep=' ')
        if len(vals) != 5:
            continue
        # 같은 fit 블록 안의 std는 보통 1~3줄 뒤
        stds = None
        for k in range(1, 5):
            if i + k < len(lines):
                ms = std_pat.search(lines[i+k])
                if ms:
                    s = np.fromstring(ms.group(1), sep=' ')
                    if len(s) == 5:
                        stds = s
                        break
        records.append({
            'model': MODEL_LIST[model_counter] if model_counter < len(MODEL_LIST) else '?',
            'bin': bin_counter,
            'c_gas':  vals[0], 'c_ics': vals[1], 'c_gce':  vals[2],
            'c_bub':  vals[3], 'c_iso': vals[4],
            'std':    stds.tolist() if stds is not None else None,
        })
        bin_counter += 1
        if bin_counter >= N_BINS:
            bin_counter = 0
            model_counter += 1

print(f'Total fits parsed: {len(records)} (expected {len(MODEL_LIST)*N_BINS})')

# 2D 배열로 저장
n_models = len(MODEL_LIST)
arr = np.full((n_models, N_BINS, 5), np.nan)
arr_std = np.full((n_models, N_BINS, 5), np.nan)
for r in records:
    if r['model'] not in MODEL_LIST:
        continue
    mi = MODEL_LIST.index(r['model'])
    bi = r['bin']
    arr[mi, bi]  = [r['c_gas'], r['c_ics'], r['c_gce'], r['c_bub'], r['c_iso']]
    if r['std'] is not None:
        arr_std[mi, bi] = r['std']

np.savez('./sanghwan_16yr_cparams_extracted.npz',
         models=MODEL_LIST,
         params=arr,           # shape (14, 17, 5)
         stds=arr_std,         # shape (14, 17, 5)
         param_names=['c_gas','c_ics','c_gce','c_bub','c_iso'])
print('Saved: ./sanghwan_16yr_cparams_extracted.npz')

# 빠른 sanity print: Model X의 c_iso 17 bin
print('\nModel X c_iso per bin (Sanghwan 16yr, FRONT+BACK):')
mi_X = MODEL_LIST.index('X')
for bi in range(N_BINS):
    v = arr[mi_X, bi, 4]
    print(f'  bin {bi:2d}: c_iso = {v:.4e}')
