# procon_W&B – 重み付きA*探索実験用フレームワーク

本リポジトリでは、高速なC++実装の重み付きA*探索アルゴリズムをPythonベースのオーケストレーション層に接続し、バッチ処理による評価と（オプションで）Weights & Biasesへのログ記録を可能にしています。実装構造は`AI.md`で説明されている仕様に準拠しています。

## クイックスタート

- サイズ4の小規模バッチを実行し、初回使用時にC++ソルバーをコンパイルします：
- 実行ツールは解を`artifacts/results/{size}x{size}/`ディレクトリに保存し、結果のJSON要約を出力します。

```
python3 src/run_search.py --size 4 --max_problems 3 --w 2.0 --time_limit_s 5
```

## スイープ対応トレーナー（Weights & Biasesオプション）

- 同じ評価を実行しつつ、問題ごとのメトリクスと概要をログ記録します。
- `wandb`がインストールされていない場合、エラーなく無効化されたロガーにフォールバックします。

```
python3 src/trainer.py
```

カスタマイズを行うには、`src/trainer.py:Config`内のデータクラスデフォルト値を変更するか、`train({...})`関数にカスタム設定を指定して呼び出してください。

## Weights & Biases設定（リモート対応版）

- インストールとログイン：
  - `pip install wandb`
  - `wandb login`  # または環境変数`WANDB_API_KEY=...`を設定
- `WANDB_PROJECT`でプロジェクトを設定します（デフォルトは`procon-wastar`）。オプションで`WANDB_ENTITY`も設定可能です。
- 実行結果はW&Bアカウントとプロジェクトの下に表示され、アーティファクトには解ディレクトリと実際に使用した問題リストが含まれます。

### スイープ機能

本リポジトリには`sweep.yaml`ファイルが含まれています。スイープを作成して実行するには：

```
wandb sweep sweep.yaml   # `entity/project/abc123`のようなスイープIDが返されます
wandb agent <that-sweep-id>
```

複数のエージェント（リモートマシン上でも可）を同時に起動して並列実行可能です。すべての結果はW&Bサイト上で同じスイープに集約されます。

## パラメータ

- `--w`: 重み付きA*アルゴリズムの係数（環境変数`WASTAR_W`として設定）。値は1.0以上である必要があります。
- 高速ソルバーの調整パラメータ（環境変数として渡されます）：
  - `--fast_max_small_n` → `FAST_MAX_SMALL_N`
  - `--fast_cand_cap` → `FAST_CAND_CAP`
  - `--fast_topk` → `FAST_TOPK`
  - `--k_top_moves` → `K_TOP_MOVES`（同義語; 後者が優先されます）
- その他のパラメータ：
  - `--alpha` → `WASTAR_ALPHA`（0..1の範囲）。複合ヒューリスティック：h_eff=(1-alpha)*h_sum + alpha*h_count
  - `--tie_break` → `TIE_BREAK` （`h_min|h_max|g_min|g_max`）。f値が同点の場合の優先順位決定
  - `--max_depth` → `MAX_DEPTH` （g値の上限値）
  - `--time_limit_s` → `WASTAR_TIME_LIMIT_S`/`TIME_LIMIT_S` （ソルバーが途中最適解を返却する時間制限）

## 注意事項

- CLIソルバーは`core/wastar.cpp`から`core/wastar`に自動的にコンパイルされます（`g++`または`clang++`が必要です）。
- 問題ごとに計算されるメトリクス：
  - `pairs_rate`: 最終グリッド上でマンハッタン距離が1の値ペアの割合
  - `ops_count`: 解法における回転操作の回数
  - `time_s`: ウォールクロック時間（ソルバー計測値）
  - `nodes_expanded`, `nodes_generated`, `open_max`, `peak_rss_mb`
  - ソルバーは`solved`/`partial`の状態と`h_sum`/`h_count`/`h_eff`も出力します
- タイムアウトが発生した場合は操作数は0となり、`pairs_rate`は初期グリッド上で計算されます。
