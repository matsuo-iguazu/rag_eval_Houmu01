# 法務お助けAgent評価ツール

AIエージェント（法務お助けAgent1号）の出力結果をベーステキストと比較して評価するツールです。

---

## 📁 データ構成

- `INPUT_CSV = "data/rag_eval_input.csv"`  
  **列名称**：`question`, `toc`, `reference`, `rag_output`, `quote`

- `OUTPUT_CSV = "data/rag_eval_output.csv"`

---

## 🚀 実行方法

```bash
pip install -r requirements.txt  # openai は既に >=1.0.0 でOK
export OPENAI_API_KEY="sk-..."   # 未設定なら環境変数を指定
python evaluator.py