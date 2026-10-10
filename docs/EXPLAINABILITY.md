# Model explanations and adverse-action reasons

All examples and measurements in this project use **synthetic data**. These explanations support a credit officer's review; they are not a credit decision.

## What the waterfall means

The logistic champion assigns each feature a local contribution equal to its fitted coefficient multiplied by the feature's standardised deviation from the training mean. Adding all contributions to the fitted intercept reproduces the model's **raw default log-odds**. The calibrated probability of default is shown separately because isotonic calibration is nonlinear; waterfall bars must not be presented as additive calibrated-PD points.

For the histogram boosting challenger, the module attempts SHAP TreeExplainer when available. The Windows environment used for this run did not have SHAP, so it uses deterministic path-permutation attributions from the training reference instead. The average path contributions telescope to the challenger's raw log-odds difference. A chart receives every feature's value, training reference, signed contribution, English/Urdu label and plain-language text, plus the baseline and prediction log-odds. Top five increasing and top five reducing features are also returned. Global importance is absolute standardised coefficient for logistic and validation-set permutation ROC-AUC loss for the challenger; those scales should not be compared across models.

Tiny-noise stability tests apply eight fixed perturbations of one-millionth of feature/reference scale to continuous inputs. A reason set is stable when its materially sized top-five members remain the same; harmless rank swaps and contributions below 0.01 raw log-odds are ignored.

## Notice and persistence

A rule-based notice is generated only when an existing decision says **rejected** or **referred**. It lists up to three principal risk-increasing factors whose value also points in the registry's expected risk direction. An imputed bureau score is not cited as an observed score. A human officer must verify the evidence and retains the final decision. The notice generator uses no LLM.

Application scoring and `GET /api/v1/applications/{id}/explanation` create a new feature snapshot and `ScoreResult` with bilingual reasons. The API returns the scorecard tier A–E. The current database enum stores the corresponding coarse tier (A/B → low, C → medium, D/E → high); the A–E tier remains in the response. Repeated GET calls currently create additional historical snapshots and score results.

Some fitted logistic coefficients can point against a feature's expected economic direction because several cash-flow and channel features are correlated. The waterfall reports the model's actual arithmetic, including those contributions, while notices prefer aligned directional reasons. These are associations relative to this synthetic training set, not causal explanations. Refit and governance review are required before real lending use.
