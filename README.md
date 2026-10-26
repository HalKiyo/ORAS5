# ORAS5

ORAS5 海洋再解析の月平均 SST (`sosstsst`) を 1° 格子へリグリッドし、CNN 用の入力 (tos) を作成するスクリプト群です。

## 構成

| ファイル | 役割 |
| --- | --- |
| `cdo_regrid.py` | CDO (`remapbil r360x180`) で高解像度 ORAS5 を 1° 格子に変換する本体。期間ごとの設定は `PERIODS` |
| `preprocess.py` | リグリッド済み SST から raw / clim / variance(標準偏差) / anom / std と 5° 粗視化版を計算し pickle と npy を保存する本体。期間ごとの設定は `PERIODS` |
| `cdo_regrid_1958_2014.py`, `cdo_regrid_2015_2023.py` | 従来のエントリポイント (薄いラッパー) |
| `preprocess_1958_2014.py`, `preprocess_2015_2023.py` | 従来のエントリポイント (薄いラッパー) |

期間ごとの違い

| 期間 | 年 | 月 | 入力ストリーム |
| --- | --- | --- | --- |
| `1958_2014` | 1958 to 2014 (57年) | Apr, Jul | `CONS` (consolidated) |
| `2015_2023` | 2015 to 2023 (9年) | Apr | `OPER` (operational) |

## パス

* 生データ入力 `/work/kajiyama/data/ORAS5/sosstsst_control_monthly_highres_2D_{YYYYMM}_{CONS|OPER}_v0.1.nc`
* リグリッド出力 / 前処理入力 `/work/kajiyama/cdo/ORAS5/sst_{year}_{Mon}.nc`
* pickle 出力 `/work/kajiyama/preprocessed/ORAS5` + `tos_ORAS5_{Mon}.pickle`
* npy 出力 `/work/kajiyama/cnn/transfer_input/tos/tos_{raw,clim,variance,anom,std,coarse,...}_{Mon}.npy`

## 実行方法

```bash
# 1. リグリッド (従来どおり save_flag=False で確認のみ)
python cdo_regrid_1958_2014.py        # = python cdo_regrid.py 1958_2014
python cdo_regrid.py 1958_2014 --save # 実際に保存

# 2. 前処理
python preprocess_1958_2014.py        # = python preprocess.py 1958_2014
python preprocess.py 1958_2014 --save # 実際に保存
```

ラッパーは従来どおり保存しません (`save_flag=False`)。保存する場合は本体スクリプトに `--save` を付けてください。

## 注意

* `PICKLEDIR` の末尾に `/` が無いため、pickle は `/work/kajiyama/preprocessed/ORAS5tos_ORAS5_{Mon}.pickle` に保存されます (従来の挙動を維持)。
* `1958_2014` と `2015_2023` の前処理は出力ファイル名が同じ (`tos_*_Apr.npy` など) なので、後に実行した方で上書きされます。
