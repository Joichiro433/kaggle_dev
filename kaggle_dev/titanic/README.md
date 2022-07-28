# タイタニック号の生存者予測

Kaggle: Titanic - Machine Learning from Disaster の予測モデル作成リポジトリ

https://www.kaggle.com/competitions/titanic/leaderboard

## Installation

インストールにはcondaforgeを用いている。動作はpython=3.9.x で確認済み

```shell
./pip_install.sh
```

## Usage

* `data_analysis.ipynb` でデータの前処理を行う

* 予測モデル作成には下記のコマンドを実行する。モデルは`outputs/` に作成される
  ```shell
  python src/train.py
  ```

* 推論には下記のコマンドを実行する
  ```shell
  python src/predict.py
  ```

* デモアプリの立ち上げには下記のコマンドを実行する
  ```shell
  streamlit run src/front.py
  ```

  
