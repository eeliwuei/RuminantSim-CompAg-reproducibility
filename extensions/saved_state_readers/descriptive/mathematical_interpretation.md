# R7.2 mathematical interpretation

All statements condition on one frozen training replicate, its finite candidate menu Q, and the specified evaluation distribution. Let L(q,U) be the unchanged binary loss, including the original violation-or-unknown convention. Y is exactly the information available to the implemented rule, including the stated observation noise and excluding actual test DM. The full-information sigma-field contains Y and U. Any feasible rule π(Y) must select from the same Q.

## Three risks and the identifiable decomposition

Define

\[
 R(\pi)=\mathbb E[L(\pi(Y),U)],\quad
 R_F^*=\mathbb E\big[\min_{q\in Q}L(q,U)\big],\quad
 R_Y^*=\mathbb E\big[\min_{q\in Q}\mathbb E\{L(q,U)\mid Y\}\big].
\]

For a finite menu with a fixed tie-breaking rule, the conditional minimum is attainable by a measurable rule. Randomising over actions cannot improve on the conditional minimum because expected binary loss is linear in the randomisation weights. For every Y-measurable rule,

\[
 R_F^*\le R_Y^*\le R(\pi),\qquad
 R(\pi)=R_F^*+(R_Y^*-R_F^*)+(R(\pi)-R_Y^*).
\]

The first inequality follows pointwise from `min_q L(q,U) <= L(π(Y),U)` and more sharply from conditional Jensen/minimum: `E[min_q L | Y] <= min_q E[L | Y]`. The second follows by minimising conditional expected loss. The three nonnegative terms are (i) failure even with complete state information within Q; (ii) the risk of the stated information restriction within Q; and (iii) excess risk relative to the best implementable rule for that Q and Y. The information term includes hidden actual DM and, where present, measurement uncertainty.

The saved candidate flags identify an empirical full-information all-fail frequency and an empirical oracle-excess frequency. They do **not** identify R_Y* or split that oracle excess into (ii) and (iii). Nor is a finite-sample count itself the exact population risk. The existing `selection_mistake_count` field is preserved as an archival name; in current prose it is `oracle-excess failures`, not an estimate of avoidable scoring error.

Illustration: two equally likely hidden states give identical Y. Action a succeeds only in state 1 and action b succeeds only in state 2. The full-information risk is 0, the best implementable risk is 1/2, and either constant rule already achieves 1/2. The entire oracle gap is attributable to missing information, with no algorithmic excess.

For Q3 contained in Q2, the full-information all-fail indicator cannot decrease under compression. The empirical excess `selected loss minus all-fail` can increase or decrease because both the selected rule and reference floor change. In the six saved A/C TAB attribution cells the floor increases in all six, whereas oracle excess decreases in two and increases in four. This does not quantify an information-versus-scoring allocation.

## Marginal sum, joint failure and ranking

For exact marginal failure probabilities, linearity of expectation gives

\[
 \sum_j P(L_j=1)=\mathbb E\Big[\sum_j L_j\Big],\qquad
 P(\cup_j\{L_j=1\})\le\sum_j P(L_j=1).
\]

The sum is the expected number of failed constraints. The union probability counts a state once. In the deterministic 100-state example, action a fails both constraints in the same four states (sum 0.08; union 0.04). Action b fails one constraint in states 0–2 and the other in states 3–5 (sum 0.06; union 0.06). Minimising the sum selects b; minimising joint risk selects a. Exact marginals are sufficient for this reversal. The actual score uses approximated marginal probabilities; its numerical sum is not automatically a bound on the true joint risk. The experiment's exact binomial qualification is based on original evaluated losses, not the score.

## Effect of variance at a fixed predicted margin

Let m_raw be the predicted signed raw margin (supply minus lower bound, or upper bound minus supply, using the implemented row orientation), and let t be the original numerical tolerance. Define the **effective margin** m = m_raw + t, so that violation is raw margin < -t. For positive total variance v and fixed effective margin m, the normal-tail score is

\[
 p(m,v)=\Phi(-m/\sqrt v),\qquad
 \frac{\partial p}{\partial v}=
 \frac{m}{2v^{3/2}}\,\phi(m/\sqrt v).
\]

Thus increasing v raises the approximate failure probability when m_raw + t > 0, lowers it when m_raw + t < 0, and leaves it at 1/2 when m_raw + t = 0. The derivative sign is the effective-margin sign, not the raw-margin sign arbitrarily close to the tolerance boundary. Equivalently, for standard deviation s>0, `∂p/∂s = m φ(m/s)/s²`. Zero variance is a boundary handled by the implemented point rule, not this derivative. The derivative holds m fixed: posterior means and per-action margins also change in actual comparisons. The derivation does not establish that this mechanism caused the observed NEL tradeoff; identifying that would require candidate-wise score diagnostics. No new scoring weights or optimisation rule are proposed here.

## Top-ceiling-5% and conditional severity

For n defined nonnegative deficits d_i and k=ceil(0.05n), if at most k deficits are positive,

\[
 T_{5\%}=k^{-1}\sum_{i=n-k+1}^{n}d_{(i)}
 =\frac n k\overline d.
\]

The identity follows because every positive deficit is in those k positions; the remaining positions contain zeros. It is exact with n/k, not exactly 20 if n is not divisible by 20. This condition holds for all 864 latest rows; their all-defined empirical P95 (NumPy linear quantile) is zero. Consequently the reported top-5% statistic is a rescaled mean in these saved data and contributes no separate tail-shape ranking.

For positive-deficit frequency p_+ over defined states, `mean = p_+ × mean(deficit | deficit>0)`. The archived violation classification instead uses deficit greater than its original, unnormalised numerical tolerance. These two subsets need not be equal in general. They coincide in all 864 checked rows (zero positive deficits at or below tolerance), so the analogous occurrence-times-violation-conditional-mean identity applies here. Unknown states are excluded from these defined-state moments and remain included in the original joint-failure event where applicable; occurrence rates over all 20,000 states and over defined states are labelled separately.

All descriptions of lower conditional mean or P95 compare different strategies' violation subsets. They are not paired claims that every state's deficit decreases, and they are not formal tail significance or animal-harm findings.
