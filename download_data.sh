#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")" && mkdir -p data
curl -L -o data/raw.zip "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
(cd data && unzip -o raw.zip)
python -c "import pandas as pd; d = pd.concat(pd.read_excel('data/online_retail_II.xlsx', sheet_name=None, dtype={'Invoice': str, 'StockCode': str, 'Description': str}).values()); d['Description'] = d['Description'].astype('string'); d.to_parquet('data/online_retail_II.parquet', index=False)"
