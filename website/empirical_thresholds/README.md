# Empirical household-size thresholds for Weebly

`foodnorm-empirical.js` is a complete replacement for the current results-page
script. It keeps the existing Weebly placeholders and calorie-based FoodNorm
calculation, but replaces regression ZL/ZU with the empirical household-size
lookup produced by `analysis.empirical.generate_charts` on 2026-09-28.

Paste its contents inside one `<script>...</script>` block on the Weebly results
page. Do not load it together with the regression version.

The empirical method has no sedentary ZL observation for household sizes 5, 7,
and 9. Those fields display `לא זמין`. Household sizes outside 1–9 raise an
explicit error because the analysis does not define an extrapolation rule.
