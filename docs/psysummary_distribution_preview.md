# PsySummary Distribution Preview

`Distribution Preview` compares the original observations with the rows retained by the active PsySummary filters.
It is available from the main Summary window and from the Define Filter window. The preview uses the same ordered
filter list and the same Summary Rows/Columns grouping as analysis and filtered-data export.

The preview directly uses the variables already configured in the Summary window: numeric variables in `Data` are
plotted, while `Rows` and `Columns` define the groups. There are no duplicate variable or grouping selectors in the
preview. Plot types are independently selectable, with `Violin + Box + Raw Points` enabled by default:

- Violin, boxplot, and jittered raw observations
- Histogram and kernel-density estimate
- Empirical cumulative distribution function (ECDF)

Retained observations are blue and excluded observations are orange. A list on the left contains every combination of
the active Data variables, Rows, and Columns and reports finite-variable counts: `N before`, `N excluded`, and
`N retained`. Clicking a list item updates the plots on the right, where Before filtering and After filtering are kept
in two fixed columns. To change a variable or grouping, update the main Summary window and reopen the preview.

## Condition-wise filtering warning

When `Run` is clicked with active filters and Rows and Columns that define more than one observed data cell, PsySummary
writes a warning to the output log. Criteria estimated separately inside experimental conditions can exaggerate condition
differences. The warning recommends condition-blind exclusion criteria and cites André (2022), *Outlier exclusion
procedures must be blind to the researcher's hypothesis*, https://doi.org/10.1037/xge0001069.
