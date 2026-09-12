# Adult Income Classification

An applied machine learning project that predicts whether a U.S. Census respondent earns more than **$50,000 per year** using demographic and employment attributes from the UCI Adult (Census Income) dataset.

The project emphasizes an interpretable, leakage-safe workflow and responsible reporting—not production deployment or maximizing a single metric.

## Project contents

```text
.
├── data/
│   └── adult.csv                         # Prepared UCI Adult dataset (48,842 records)
├── modeling.ipynb                        # End-to-end exploration, training, and evaluation
├── Machine_Learning_Analysis_Report.pdf  # PDF version of the analysis report
└── module_summary.pdf                    # Project/module summary
```

## Dataset

The data originate from the [UCI Adult dataset](https://archive.ics.uci.edu/dataset/2/adult), derived from 1994 U.S. Census Bureau data. The target, `income`, has two classes:

- `<=50K`
- `>50K`

The local dataset contains 48,842 records and 15 columns. Approximately 23.9% of records are in the `>50K` class. Missing values occur in `workclass`, `occupation`, and `native-country`.

## Methodology

The workflow in [`modeling.ipynb`](modeling.ipynb) performs the following:

1. Loads and inspects the dataset.
2. Drops `fnlwgt` (a sampling weight) and `education` (duplicative of `education-num`).
3. Uses a stratified 80/20 train/test split with `random_state=42`.
4. Fits preprocessing only within scikit-learn pipelines to prevent test-data leakage:
   - Numeric variables: median imputation and standard scaling.
   - Categorical variables: most-frequent imputation and one-hot encoding.
5. Trains a class-balanced logistic regression as the primary model.
6. Trains a class-balanced random forest as a nonlinear comparison.
7. Evaluates accuracy, precision, recall, F1, ROC-AUC, confusion matrices, ROC curves, subgroup metrics, and logistic-regression coefficients.

## Results

Metrics below are from the held-out test set (9,769 records; 2,338 `>50K` cases).

| Model | Accuracy | Precision (`>50K`) | Recall (`>50K`) | F1 (`>50K`) | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic regression | 0.8069 | 0.5653 | 0.8366 | 0.6747 | 0.9037 |
| Random forest | 0.8238 | 0.5924 | 0.8456 | 0.6967 | 0.9192 |

The random forest modestly outperforms logistic regression across the reported metrics. Both models favor recall for the minority `>50K` class, which means they identify many high-income cases but generate meaningful false positives. Accuracy should therefore not be interpreted alone.

## Run the analysis

Open the notebook from the repository root in a Python environment with Jupyter and the notebook's imported data-science libraries available:

```bash
jupyter notebook modeling.ipynb
```

Run all cells in order. The notebook reads the included `data/adult.csv` file using a path relative to the repository root.

## Fairness and limitations

`sex` and `race` are included as model inputs, and the notebook reports subgroup recall and F1 to make differing error rates visible. These results are descriptive, not causal, and historical census data can encode societal inequities.

This work should not be used for hiring, lending, eligibility, or other high-impact decisions. The data describe a 1994 context, and model outputs do not represent a person's inherent earning potential.

## Further reading

- [Analysis report](reports/Machine_Learning_Analysis_Report.md)
- [UCI Adult dataset](https://doi.org/10.24432/C5XW20)

## License

No license has been specified for this repository. Consult the UCI dataset's terms before redistributing or using the data.