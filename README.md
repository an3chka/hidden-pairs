# Hidden Pairs: do two observations belong to the same entity?

Solution for a Kaggle competition from the *Introduction to Machine Learning* course
(3rd year, September 2026), team phase. Each row describes a pair of hidden observations
with 512 anonymous numerical features; the task is to predict the probability that both
observations belong to the same hidden entity. Metric: ROC-AUC.

**Result:** validation ROC-AUC **0.9548**, public leaderboard **0.9437**.

*Русская версия — [ниже](#русская-версия).*

## Approach

1. **Data.** Every feature is only weakly related to the target (median |corr| ≈ 0.21,
   best single feature ROC-AUC 0.70), and features are nearly uncorrelated with each other
   (mean |corr| ≈ 0.16; PCA needs 323 components for 90% of variance).
   The signal is spread across all 512 features.
2. **Trees first.** A single decision tree peaks at 0.79 on validation (depth 6) and overfits
   beyond that (train 1.00 / validation 0.69 unconstrained). Random forest and gradient
   boosting reach 0.90 and 0.92.
3. **Distance-based models.** Trees split on one feature at a time, while an RBF kernel compares
   objects across all features at once — so SVM and kNN outperform all tree ensembles.
4. **Final pipeline (solution 1):**
   `log1p → StandardScaler → PCA (464 of 512) → component weighting λ^0.1875 → RBF-SVM (C=5, γ=1/464)`.
   Weighting multiplies each principal component by its variance to the power 0.1875,
   amplifying the main directions relative to the noisy tail.
5. **Weighted ensemble (solution 2)** of 7 models with Optuna-tuned weights. Effectively
   4 models carry the weight: two SVMs (76%) plus kNN and MLP (22%), which are weaker alone
   but least correlated with the SVMs. Redundant models (e.g. Nyström, an approximation of the
   same RBF kernel, ρ = 0.98 with SVM) get ≈ 0.

### Contribution of each preprocessing step (SVM, C = 5)

| Step | CV ROC-AUC | Validation ROC-AUC |
|---|---|---|
| raw features | 0.9210 | 0.9447 |
| + log1p | 0.9242 | 0.9489 |
| + PCA 464 | 0.9238 | 0.9494 |
| + component weighting (power 0.1875) | 0.9253 | **0.9548** |

### Model comparison

| Model | CV ROC-AUC | Validation ROC-AUC |
|---|---|---|
| **SVM on weighted PCA** (final) | 0.9253 | **0.9548** |
| Weighted ensemble | **0.9285** | 0.9531 |
| SVM on PCA 400 | 0.9255 | 0.9470 |
| MLP (PyTorch) | 0.9162 | 0.9359 |
| Nyström + Ridge | 0.9212 | 0.9344 |
| kNN | 0.9077 | 0.9266 |
| MLP (scikit-learn) | 0.9103 | 0.9227 |
| Gradient boosting (HGB) | 0.9051 | 0.9166 |
| Random forest | 0.8916 | 0.8978 |
| Decision tree | 0.7443 | 0.7856 |

CV — 5-fold out-of-fold ROC-AUC on train; validation — the supplied validation set with
entities unseen in train. Hyperparameters were selected by the mean of the two.

**Final choice — SVM on weighted PCA.** It is on par with the ensemble (better on validation
and public leaderboard, slightly worse on CV), but it is a single model, about 5× faster and
easier to explain.

## Repository structure

| File | Contents |
|---|---|
| `hidden_pairs_clean.ipynb` | Full pipeline: data, models, Optuna tuning, ensemble, submissions |
| `week-03-hidden-pairs/` | Competition data: `train.csv`, `validation.csv`, `test.csv` (unlabeled), `sample_submission.csv` |
| `results/experiments.csv` | Experiment table: model, features, CV / validation ROC-AUC, runtime, seed |
| `hidden_pairs_presentation.pdf` | Slides for the project defense |
| `requirements.txt` | Library versions |

## How to run

The competition data are included in `week-03-hidden-pairs/` (published with the
instructor's permission). To reproduce the results:

1. Clone the repository: the notebook reads the data from `week-03-hidden-pairs/`.
2. Install dependencies: `pip install -r requirements.txt` (Python 3.12+).
3. Restart the kernel and run all cells (~3 minutes on a 12-core CPU).

The notebook writes `final_submission.csv`, `submission_1_*.csv`, `submission_2_*.csv`
and `results/experiments.csv`. All models are trained on `train.csv` only; seeds are fixed
and PyTorch runs on CPU, so reruns produce identical predictions.

---

## Русская версия

Решение учебного Kaggle-соревнования курса «Введение в машинное обучение»
(3 курс, сентябрь 2026), командный этап. По 512 анонимным признакам пары наблюдений нужно
предсказать вероятность того, что оба наблюдения относятся к одной скрытой сущности.
Метрика — ROC-AUC.

**Результат:** validation ROC-AUC **0.9548**, public leaderboard **0.9437**.

**Ход решения:**
- в данных сигнал слабый в каждом признаке, но есть во всех, а признаки почти не коррелируют
  между собой;
- деревья решений переобучаются и упираются в 0.79, лес и бустинг — в 0.90–0.92;
- модели на расстояниях (SVM, kNN) работают лучше: RBF-ядро сравнивает объекты сразу
  по всем признакам;
- итоговый пайплайн: `log1p → StandardScaler → PCA 464 → вес компонент λ^0.1875 → RBF-SVM`;
  перевзвешивание усиливает главные компоненты и ослабляет шумные;
- второе решение — ансамбль с подобранными весами; по качеству он равен одной SVM, поэтому
  выбрана более простая модель.

**Данные** соревнования лежат в папке `week-03-hidden-pairs/` (выложены с разрешения
преподавателя). Для запуска установите зависимости (`pip install -r requirements.txt`)
и выполните Restart → Run All.
