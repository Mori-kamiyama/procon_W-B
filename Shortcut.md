#!/usr/bin/env markdown

# ショートカット探索 知見まとめ

本ドキュメントは、盤面状態の再現・保存とショートカット探索を高速かつ安定に行うために実施した実装・計測・チューニングの知見をまとめたものです。実装は主に Rust 側（`rust/replay_shortcuts`）に集約し、Python は補助ツール（集計・ウィンドウ探索ドライバ）として利用しています。

## 概要
- 目的: 既知の遷移列（ops）から生成した中間状態集合に対して、より短手数で後段の既知状態へ到達できる「ショートカット」を見つける。
- 成果:
  - 深さ2で既知の+4改善を安定検出（start 398–400 付近 → target 404–405）。
  - 深さ3〜4の targeted 探索で同等の改善を高速再現。
  - 深さ6〜8でも、小さな候補・範囲ウィンドウ・限定乖離探索（LDS）により時間を抑えつつ探索。
- 実装上の要点:
  - 安定・高速な盤面指紋（fingerprint）。
  - 候補評価の差分ヒューリスティック化。
  - 候補集合の焦点化（focus）、深さ依存の幅・サイズ縮小、LDS による分岐多様化。
  - 開始インデックス範囲指定とスライディングウィンドウ走査。

## 指紋（fingerprint）
- 当初の単純 XOR は「位置情報が相殺されやすく大量衝突」を招き、visited/known 判定が崩壊 ⇒ 検出不能に。
- 対策として、非線形ミックス（SplitMix64 風）+ ローテーションで 64bit 指紋を実装し、Python/Rust で同一ロジックに揃えました。
  - Python: `src/replay_shortcuts.py:22`
  - Rust: `rust/replay_shortcuts/src/main.rs:89`
- 効果: 低衝突・安定な状態識別。探索の正当性が復帰。

## 候補評価の高速化（差分ヒューリスティック）
- 旧実装: 各候補ごとに盤面全体のヒューリスティック（マンハッタン距離）を再計算 ⇒ 高コスト。
- 新実装: サブグリッド回転で動く値の寄与のみ差し替え、合計値を差分更新する `HeuState` を導入。
  - 構造体 `HeuState` が「値→座標（row-major 2箇所）」と「合計ヒューリスティック」を保持し、回転差分を適用・復元。
  - 実装: `rust/replay_shortcuts/src/main.rs:181` 以降。
- 効果: d2 で ~6.7s → ~1.3s、d3/d4 も数秒で探索可能に（Release）。

## 探索戦略（Rust バイナリの主なフラグ）
- 範囲指定:
  - `--start-begin <i>` / `--start-end <j>`: 探索する開始ステップの半開区間 [i, j) を指定。
- 候補集合の焦点化（focus）:
  - デフォルトは「距離>1のペアを含む回転」のみ候補化して分枝を削減。
  - `--focus-off`: 焦点化を無効化し、入口の多様性を確保。
- 深さ依存の縮小:
  - 深くなるほど `top-k` と `n_max` を段階的に縮小し、爆発を抑制。
- 限定乖離探索（LDS）:
  - `--lds-limit <k>`: 各層で“最良以外”の選択を合計 k 回まで許容。貪欲なマンハッタン盆地からの離脱に有効。
- リソース・幅:
  - `--search-depth d`, `--n-max N`, `--top-k K`, `--max-nodes M` で探索範囲を制御。

### ストキャスティック・ハイブリッド（実装追加）
- ランダムウォーク前置き + 貪欲降下で深層（d10〜d12）を実用化。
  - `--stochastic`: 有効化スイッチ。
  - `--prefix-depth <p>`: ランダムに“良さげな手”から p 手だけ歩く（デフォルト 3）。
  - `--walks <w>`: 前置きランダムウォークの本数（デフォルト 256）。
  - 以降はヒューリスティック値が最良の手を貪欲に選択（深さ依存で `top-k`/`n-max` を縮小）。
  - `--no-h-gate` を併用すると前置きで“少し登る”ことも許容でき、盆地越えに効く場合あり。

## パラメータの目安（例）
- d2（高速検証）:
  - `--search-depth 2 --n-max 5 --top-k 12 --max-nodes 2000`
  - 結果: +4 改善を 3 件（Release で数秒）。
- d3/d4（targeted で素早く）:
  - d3: `--search-depth 3 --n-max 6 --top-k 20 --max-nodes 15000 --start-begin 380 --start-end 520`
  - d4: `--search-depth 4 --n-max 6 --top-k 16 --max-nodes 20000 --start-begin 380 --start-end 500`
  - `--focus-off` も有効。+4 改善 3 件を数秒で再現。
- d6/d8（深く・小さく・LDS 併用）:
  - d6: `--search-depth 6 --n-max 4 --top-k 12 --max-nodes 40000 --focus-off --lds-limit 2..3`
  - d8: `--search-depth 8 --n-max 4 --top-k 12 --max-nodes 60000..90000 --focus-off --lds-limit 3..4`
  - 40 ステップ幅のウィンドウをスライドして探索。

- d10/d12（ストキャスティック・ハイブリッド）:
  - 例: d12 を現実的時間で試す
    ```bash
    rust/replay_shortcuts/target/release/replay_shortcuts \
      --problem test.json --ops test.ops.json \
      --out-dir artifacts/replay_bench_d12_stoch \
      --search-depth 12 --n-max 4 --top-k 12 --max-nodes 200000 \
      --start-begin 360 --start-end 460 \
      --stochastic --prefix-depth 3 --walks 512 \
      --focus-off --stride 2
    ```
  - 速度と多様性のトレードオフ: `--walks` を 256→1024 に増やすと当たりやすくなるが時間増。
  - “登り”を許す場合は `--no-h-gate` を追加（前置き局所脱出）。

## 実験結果サマリ（代表）
- d2: +4 改善 3 件（start 398–400 → target 404–405）。Debug では ~70s だが Release で数秒に短縮。
- d3/d4（targeted）: 同 +4 改善を高速再現（~3〜6秒）。
- d6/d8（小さめ候補 + focus-off + LDS）: 特定レンジ（380–420 付近）で +4 改善が継続的に観測。他レンジでは未発見が多い。

## スライディングウィンドウ探索の自動化
- Python ドライバ `artifacts/scan_windows.py` を追加。
  - 例: d6 を 40 ステップ幅で 4 区間走査
    ```bash
    python3 artifacts/scan_windows.py \
      --depth 6 \
      --windows 300:340 340:380 380:420 420:460 \
      --focus_off --lds_limit 3 --n_max 4 --top_k 12 --max_nodes 50000
    ```
  - 実行ごとに `artifacts/scan_d{depth}_summary.csv` を出力し、各ウィンドウの `count/total/best` を集約。

## さらなる改善アイデア
- Novelty タイブレーク: ヒューリスティック同値時に「新規性（隣接ペア構成の変化など）」が大きい候補を優先。
- ビームサーチ: 深さごとに上位 B 状態のみ保持し、冗長分枝を抑制。
- マクロ手（2手/短連鎖）: d2 で有効な2手組をマクロ化し、深層での到達性を底上げ。
- 逆探索ランドマーク: 後方から浅く逆回転して到達集合（指紋→後方位置）を作り、前方探索と合流。
- GPU バッチ評価: 候補の回転+ヒューリスティック計算を GPU に一括投げ。差分ヒューリスティック整備後に検討がコスパ良。

## 注意点・落とし穴
- 指紋の安定性: 線形 XOR は位置相殺で大量衝突。非線形ミックス+回転で回避。
- パラメータ爆発: `top-k`/`n-max` を深さ依存で縮小、`max-nodes` で頭打ちを設定。
- フォーカスの使い分け: 初手の多様性確保には `--focus-off` が有効だが、全層 OFF は不要な分枝増を招く場合も。今後は「初手のみ OFF→以降 ON」などの段階適用も有望。
- ビルドモード: Release ビルドを使用（Debug は桁違いに遅い）。

## 参考コマンド（単発実行例）
```bash
# d4, targeted, focus-off, LDS=2, 数秒想定
rust/replay_shortcuts/target/release/replay_shortcuts \
  --problem test.json --ops test.ops.json \
  --out-dir artifacts/replay_bench_d4_rel_focusoff \
  --search-depth 4 --n-max 6 --top-k 16 --max-nodes 20000 \
  --start-begin 380 --start-end 500 \
  --focus-off --lds-limit 2
```

---
以上。探索は「入口の多様性（focus-off / LDS / ウィンドウ）」と「中層以降の収束（focus / 深さ依存縮小 / 差分ヒューリスティック）」を両立させることが肝要です。時間予算に応じてウィンドウをスライドさせ、良さそうな谷を重点的に掘るのが実務上のおすすめ手順です。
