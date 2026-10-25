# PsySummary

PsySummary is a graphical application for transparent, reproducible reaction-time (RT) preprocessing, descriptive summaries, and optional distributional and cognitive-model fitting. It records supported data-processing and analysis operations as readable Python code.

Choose exclusion criteria, grouping variables, and models from the study design and analysis goals. The interface helps implement and inspect those decisions; it does not determine which analysis is appropriate.

## Software access

- Source code and documentation are available in this [public repository](https://github.com/yangzhangpsy/PsySummary).
- See the [PsyBuilder website](https://www.psybuilder.com) for desktop download and installation information.
- PsySummary can run as a standalone application or inside PsyBuilder through **Toolboxes > Results Data Aggregation**.

Choose an installer matching your operating system and processor architecture. GitHub's **Code > Download ZIP** provides source code, not a prebuilt application.

On Windows, extract the installer archive before launching it. Use **Extract All**; if extraction fails, try a current version of [7-Zip](https://www.7-zip.org). On macOS, open the supplied disk image and install the application.

## Repository contents

| Location | Contents |
| --- | --- |
| [psyData/](psyData/) | Application source and build files |
| [Supporting_materials_20261025.zip](Supporting_materials_20261025.zip) | Tutorial data and scripts, preprocessing validation, cognitive-model references, and GUI import/replay comparisons |
| [figs/](figs/) | Manuscript illustrations, numbered `f1.png` to `f47.png` in document order |
| [LICENSE](LICENSE) | MIT license |

## Tutorial
# Reaction time outlier handling in behavioral research. Part II: PsySummary software, validation, and tutorial

<p align="center"> Zhicheng Lin<sup>1,2</sup>, lian-zi Xing<sup>1</sup>, Xin Chen<sup>1</sup>, Yue-Jie Chen<sup>1</sup>, Yang Zhang<sup>1</sup> </p>

<sup>1</sup> _Department of Psychology, Soochow University, Suzhou, Jiangsu, China 215000_
<br>
<sup>2</sup> _Department of Psychology, Yonsei University, Seoul, Republic of Korea, 03722_


The workflow has three stages:

1. **Raw data management:** import, inspect, rename, and transform variables.
2. **Data filtering:** define eligibility and exclusion criteria, inspect their effects, and export retained trials.
3. **Data summary and RT modeling:** calculate descriptive statistics or fit models to retained trial-level observations.

![PsySummary three-stage workflow](https://raw.githubusercontent.com/yangzhangpsy/PsySummary/main/figs/f1.png)

<details>
<summary>Interface overview</summary>

![PsySummary interface and RT fit diagnostics](https://raw.githubusercontent.com/yangzhangpsy/PsySummary/main/figs/f2.png)

</details>

- [Step 1: Raw Data Management](#1)
- [Step 2: Data Filtering](#2)
- [Step 3: Data Summary and RT Modeling](#3)
- [Validation and Scope](#validation-and-scope)

<a id="1"></a>
## Step 1: Raw Data Management

### 1.1 Load and inspect data

PsySummary accepts MATLAB (`.mat`), delimited text (`.csv`, `.txt`, `.dat`), and typed `.psydata` files.

1. Choose **File > Load Data**, select the input files, and click **Open**.
2. For delimited text, configure **Text Encoding**, **Contains Header**, **Text Delimiter**, and **Merge Delimiters** in the **Data Format** dialog. Check the preview, then click **OK**.
3. Choose **File > View Data** to open **Data Viewer**. Use **Data View** to inspect values and **Variable View** to inspect names, storage types, and display precision.

Choose a predefined delimiter or enter a custom separator. **WhiteSpace** includes spaces and tabs; enter a tab for tab-only separation. **Merge Delimiters** optionally treats consecutive separators as one. For quoted CSV, use a comma without Merge Delimiters. Multi-character separators use regular-expression parsing and do not promise support for quoted fields containing the separator.

In **Variable View**, double-click **Name** to rename a variable, **Type** to choose Numeric, Text, or Boolean, or **Decimals** to change numeric display precision. A column header in **Data View** can also be double-clicked to rename it. Type conversion validates the entire source column and preserves missing values; changing Decimals does not alter stored or exported values. Renaming updates current references, not previously calculated results.

The manuscript tutorial uses the following renamings:

| Imported name | Tutorial name | Meaning |
| --- | --- | --- |
| `target_1.rt` | `target.rt` | RT in seconds |
| `target_1.resp` | `target.resp` | Observed response |
| `target_1.acc` | `target.acc` | Accuracy: 0 incorrect, 1 correct |
| `trialsLoop_1_SOA` | `SOA` | Stimulus onset asynchrony in milliseconds |

In that example, `num` and `name` identify the same participants. Use the corresponding identifiers and units for your own data.

Choose **File > Save Data** to preserve the loaded source data as `.psydata`. **File > Open Recent** reopens previously imported files.

<details>
<summary>Load, inspect, rename, and save data</summary>

![Load input files](https://raw.githubusercontent.com/yangzhangpsy/PsySummary/main/figs/f3.png)

![Configure delimited-text import](https://raw.githubusercontent.com/yangzhangpsy/PsySummary/main/figs/f4.png)

![Reopen previously imported files with Open Recent](https://raw.githubusercontent.com/yangzhangpsy/PsySummary/main/figs/f5.png)

![Inspect and rename variables in Data View and Variable View](https://raw.githubusercontent.com/yangzhangpsy/PsySummary/main/figs/f6.png)

![Save source data](https://raw.githubusercontent.com/yangzhangpsy/PsySummary/main/figs/f7.png)

</details>

### 1.2 Arrange the workspace

Use **View** to show **Output**, **Script**, and **Aggregation Results**. Output reports progress, warnings, and errors; Script displays the recorded analysis code.

<details>
<summary>Workspace panels</summary>

![Show Output, Script, and Aggregation Results](https://raw.githubusercontent.com/yangzhangpsy/PsySummary/main/figs/f8.png)

</details>

### 1.3 Transform variables

1. Choose **Toolbox > Transform Variable**.
2. Enter a target name, such as `log_RT`.
3. Click **log**, then drag `target.rt` into the parentheses to construct its natural logarithm.
4. Click **OK** to calculate and close the dialog, or **Run** to calculate while keeping it open.

The new variable appears in **Variables**. Cancel does not undo variables created by earlier Runs. Transformations should serve a justified analysis; the RT models below use the original RT variable in its recorded time unit, not log-transformed RTs.

<details>
<summary>Transform a variable</summary>

![Construct the natural-log transformation](https://raw.githubusercontent.com/yangzhangpsy/PsySummary/main/figs/f9.png)

![Use the transformed variable](https://raw.githubusercontent.com/yangzhangpsy/PsySummary/main/figs/f10.png)

</details>

<a id="2"></a>
## Step 2: Data Filtering

### 2.1 Define the analysis groups

Drag participant identifiers into **Rows**, condition variables into **Columns**, and the measure into **Data**. For the manuscript example, use `num` and `name` in Rows, `SOA` in Columns, and `target.rt@Mean` in Data.

Rows and Columns define the groups used for filtering and aggregation. Do not assign the same grouping variable to both. Data can contain different analyses of the same measure.

<details>
<summary>Assign grouping and analysis variables</summary>

![Assign Rows, Columns, and Data](https://raw.githubusercontent.com/yangzhangpsy/PsySummary/main/figs/f11.png)

</details>

### 2.2 Choose an analysis

Click a Data item once to open the analysis menu. Select a descriptive statistic or RT distribution, or select **Ratcliff Diffusion Model**, **Linear Ballistic Accumulator**, or **Racing Diffusion Model** to open its settings.

The same menu is available through **right-click > Change Analysis...**. Double-click a configured cognitive-model item, or choose **right-click > Model Settings...**, to reopen its settings. Conventional RT distributions do not have a separate cognitive-model settings dialog.

<details>
<summary>Select an analysis</summary>

![Choose a descriptive statistic, RT distribution, or cognitive model](https://raw.githubusercontent.com/yangzhangpsy/PsySummary/main/figs/f12.png)

</details>

### 2.3 Define filters

Click **Define Filters**, select the variable, choose **Check List**, **Range**, or **Pooling CDF**, and click **Add Filter** to add a rule. Click **OK** to retain the Filter List.

For the manuscript's correct-response RT summaries:

1. Select `target.acc`, choose **Check List**, and retain `1` only.
2. Select `target.rt`, choose **Range**, and define **Raw value** bounds of `> 0.1` and `< 1.5`.
3. Add the criteria to the Filter List. These bounds are in seconds and are an illustration, not a universal recommendation.

The demo includes a missing RT/response and very short RTs. The RT bounds exclude missing or out-of-range values without replacing them. Inspect the resulting counts rather than assuming that every input row is eligible.

For categorical variables, **Search values...**, **Select All**, and **Invert Selection** help select long lists of codes. To retain all eligible identifiers except one, select all and clear the unwanted value; also exclude blank or missing identifiers.

Available range methods are:

| Method | Criterion |
| --- | --- |
| Raw value | An absolute bound in the variable's recorded unit |
| SDs | A bound based on the group's mean and sample standard deviation |
| Shifting Z | An SD cutoff around the mean adjusted to the group's sample size, without iterative recalculation |
| MAD | A bound based on the group's median and scaled median absolute deviation |

Filters are applied in listed order. Drag rules to reorder them; **and** and **or** combine the two comparisons within a range rule.

When group-estimated SD, MAD, Shifting Z, or Pooling CDF criteria are used across multiple Rows/Columns cells, Output warns about possible condition-wise filtering. A common fixed RT window or checklist alone does not trigger that warning. Choose grouping and thresholds independently of the condition effect being investigated.

**For correct/error cognitive modeling, remove the correct-only filter on `target.acc` while retaining the accuracy variable.**

<details>
<summary>Define accuracy and RT filters</summary>

![Open Define Filters](https://raw.githubusercontent.com/yangzhangpsy/PsySummary/main/figs/f13.png)

![Select correct responses with a checklist](https://raw.githubusercontent.com/yangzhangpsy/PsySummary/main/figs/f14.png)

![Select a range filter](https://raw.githubusercontent.com/yangzhangpsy/PsySummary/main/figs/f15.png)

![Choose a range method and bounds](https://raw.githubusercontent.com/yangzhangpsy/PsySummary/main/figs/f16.png)

![Review the filter criteria](https://raw.githubusercontent.com/yangzhangpsy/PsySummary/main/figs/f17.png)

</details>

### 2.4 Optional Pooling CDF

Pooling cumulative distribution functions (CDFs) fits RT distributions within the configured groups and examines their pooled CDF values. It requires finite positive RTs and at least 50 eligible trials per group. This minimum is a numerical safeguard, not a guarantee of reliable estimation.

For the tutorial demonstration, group by participant using `num` and `name` in Rows and leave Columns empty, rather than fitting the smaller participant-by-SOA cells. Apply eligibility rules before Pooling CDF.

1. Add a **Pooling CDF** rule for `target.rt`.
2. Click **Apply Filters & Run** to inspect the pooled CDF histogram, fitted mixture density, and estimated cutoff.
3. Click the plot to adjust omega, or **Estimated ω** to restore the fitted estimate.
4. Click **OK** to retain CDF values at or below omega and exclude values above it. The cutoff must be finite and strictly between 0 and 1.
5. Choose **Abort CDF Pooling** to skip the rule without changing the data.

A smaller cutoff excludes more trials. The fitted estimate is not automatically an appropriate exclusion rule; inspect the mixture and justify its use. Remove the optional CDF rule and restore SOA to Columns before returning to the main fixed-window summary example.

<details>
<summary>Pooling CDF controls and diagnostic plot</summary>

![Add a Pooling CDF rule](https://raw.githubusercontent.com/yangzhangpsy/PsySummary/main/figs/f18.png)

![Inspect the pooled CDF histogram, fitted mixture, and cutoff](https://raw.githubusercontent.com/yangzhangpsy/PsySummary/main/figs/f27.png)

</details>

### 2.5 Preview and export retained trials

Click **Preview Filter Effects** in Data Summary, or **Distribution Preview** in Define Filter. Select a variable/group from the list and inspect the before/after counts and plots:

- Violin, boxplot, and raw points.
- Histogram with kernel density estimate (KDE).
- Empirical cumulative distribution function (ECDF).

Before/after panels share axis scales; retained and excluded observations have distinct colors.

<details>
<summary>Preview filter effects</summary>

![Open the distribution preview](https://raw.githubusercontent.com/yangzhangpsy/PsySummary/main/figs/f19.png)

![Compare retained and excluded observations before and after filtering](https://raw.githubusercontent.com/yangzhangpsy/PsySummary/main/figs/f20.png)

</details>

Use **Export Filtered Data** or **File > Save Filtered Data** to save retained trial-level observations as CSV or `.psydata`, independently of the final summary. This does not replace the loaded source data.

To filter within participants but summarize by condition, leave Columns empty during participant-wise filtering and export the retained trials. Reload that export, add SOA to Columns, and summarize with an empty Filter List to avoid filtering the same trials again.

<details>
<summary>Export retained trials and change the aggregation groups</summary>

![Save filtered trial-level data](https://raw.githubusercontent.com/yangzhangpsy/PsySummary/main/figs/f22.png)

![Choose the filtered-data export destination](https://raw.githubusercontent.com/yangzhangpsy/PsySummary/main/figs/f23.png)

![Summarize the retained data with the required grouping](https://raw.githubusercontent.com/yangzhangpsy/PsySummary/main/figs/f24.png)

</details>

### 2.6 Save and reopen a setup

**Save Setup** writes a `.psysum` containing Rows, Columns, Data, filters, and cognitive-model settings. **Load Setup** restores those settings.

A setup is not a data file and does not recreate import, renaming, or computed-variable operations. Load matching data and prepare the required variables before reopening it. Incomplete model drafts can be saved, but must be valid before fitting.

<details>
<summary>Save and reopen analysis settings</summary>

![Save Setup and Load Setup](https://raw.githubusercontent.com/yangzhangpsy/PsySummary/main/figs/f21.png)

</details>

<a id="3"></a>
## Step 3: Data Summary and RT Modeling

### 3.1 Calculate and save results

Click **Apply Filters & Run** to apply the current rules and display **Aggregation Results**. Models use retained trial-level observations, not previously aggregated RT means.

- **Clipboard** copies the displayed table.
- **Export > Export Table (.txt)** saves the displayed table as text.
- **Export > Save Results (.psyresult)** preserves completed results and available fit diagnostics.
- **Load Results...** reopens a `.psyresult` without rerunning the analysis.

A `.psysum` saves an analysis setup; a `.psyresult` saves completed results.

<details>
<summary>Summary results, filtering warnings, and result export</summary>

![Calculate participant-by-SOA mean RTs](https://raw.githubusercontent.com/yangzhangpsy/PsySummary/main/figs/f25.png)

![Inspect the warning for group-estimated filtering criteria](https://raw.githubusercontent.com/yangzhangpsy/PsySummary/main/figs/f26.png)

![Save, export, and reopen completed results](https://raw.githubusercontent.com/yangzhangpsy/PsySummary/main/figs/f28.png)

</details>

### 3.2 Fit RT distributions

Click the RT item in Data and select a distribution. Options include unshifted and shifted gamma, Weibull, and lognormal distributions, alongside Wald, ex-Wald, ex-Gaussian, and inverse-Gaussian variants.

Click **Apply Filters & Run**, then inspect the fitted parameters, N valid, Converged, Parameter boundary, Shift warning, log-likelihood, Akaike information criterion (AIC), and Bayesian information criterion (BIC). Optimizer convergence does not, by itself, establish an adequate fit.

Click **View Fit** in the Converged section to open **RT Fit Diagnostics**. Select a fit from the list to inspect the observed histogram and fitted density, and the empirical and fitted CDFs. Compare information criteria only when the likelihoods describe the same observations with compatible response coding.

<details>
<summary>RT distribution fitting and diagnostics</summary>

![Inspect RT distribution fit results](https://raw.githubusercontent.com/yangzhangpsy/PsySummary/main/figs/f29.png)

![Locate View Fit in the Converged section](https://raw.githubusercontent.com/yangzhangpsy/PsySummary/main/figs/f30.png)

![Inspect observed and fitted RT densities and cumulative distributions](https://raw.githubusercontent.com/yangzhangpsy/PsySummary/main/figs/f31.png)

</details>

### 3.3 Fit cognitive RT models

Retain both correct and error trials for these examples: remove the correct-only accuracy filter, keep the accuracy column, and use eligible RTs in their recorded unit. In the manuscript example, group by participant across SOAs and use raw RT bounds of 0.1-1.5 s.

1. Select a cognitive model from the Data item's analysis menu.
2. Set **RT Unit**, choose **Response Variable**, and configure the response mappings.
3. Set parameters to Fixed or Free, specify values or optimizer starts and bounds, and review **Fit Options**. Use an explicit random seed when reproducible optimizer starts are required.
4. Click **Save Settings**, then **Apply Filters & Run**.

| Model | Mapping and identification requirements |
| --- | --- |
| Ratcliff Diffusion Model | Two physical response categories and a binary **Accuracy Variable** are required. Choose Accuracy Coding or Response Coding. The evidence starting position is relative `zr = z/a`; keep `s` fixed to identify scale. |
| Linear Ballistic Accumulator | Map response categories to accumulators; accuracy is optional. The implementation uses positive-truncated normal drift. Keep at least one `sd_v` fixed and ensure `b > A >= 0`. |
| Racing Diffusion Model | Map response categories to accumulators; accuracy is optional. Use non-negative drift rates, `b > A >= 0`, and fixed `s`. |

Each Rows/Columns group is fitted independently. Inspect observation and response counts, convergence, parameter estimates, and warnings, especially for sparse groups or groups with few error responses.

In **RT Fit Diagnostics**, **Response** and, when accuracy is available, **Correct / Error** provide views of the same fitted parameters. Switching views does not refit the model. Density and CDF panels compare observed data with model predictions.

<details>
<summary>Ratcliff diffusion model</summary>

![Select the Ratcliff diffusion model](https://raw.githubusercontent.com/yangzhangpsy/PsySummary/main/figs/f32.png)

![Choose Ratcliff boundary coding](https://raw.githubusercontent.com/yangzhangpsy/PsySummary/main/figs/f33.png)

![Configure Ratcliff data mappings, parameters, and fitting options](https://raw.githubusercontent.com/yangzhangpsy/PsySummary/main/figs/f34.png)

![Inspect Ratcliff parameter estimates](https://raw.githubusercontent.com/yangzhangpsy/PsySummary/main/figs/f35.png)

![Open the Ratcliff fit diagnostics](https://raw.githubusercontent.com/yangzhangpsy/PsySummary/main/figs/f36.png)

![Inspect Ratcliff correct and error distributions](https://raw.githubusercontent.com/yangzhangpsy/PsySummary/main/figs/f37.png)

![Inspect Ratcliff physical-response distributions](https://raw.githubusercontent.com/yangzhangpsy/PsySummary/main/figs/f38.png)

</details>

<details>
<summary>Linear ballistic accumulator</summary>

![Select the linear ballistic accumulator](https://raw.githubusercontent.com/yangzhangpsy/PsySummary/main/figs/f39.png)

![Configure LBA mappings, parameters, and fitting options](https://raw.githubusercontent.com/yangzhangpsy/PsySummary/main/figs/f40.png)

![Inspect LBA parameter estimates](https://raw.githubusercontent.com/yangzhangpsy/PsySummary/main/figs/f41.png)

![Inspect LBA response-specific fit diagnostics](https://raw.githubusercontent.com/yangzhangpsy/PsySummary/main/figs/f42.png)

</details>

<details>
<summary>Racing diffusion model</summary>

![Select the racing diffusion model](https://raw.githubusercontent.com/yangzhangpsy/PsySummary/main/figs/f43.png)

![Configure racing diffusion mappings, parameters, and fitting options](https://raw.githubusercontent.com/yangzhangpsy/PsySummary/main/figs/f44.png)

![Inspect racing diffusion parameter estimates](https://raw.githubusercontent.com/yangzhangpsy/PsySummary/main/figs/f45.png)

![Inspect racing diffusion response-specific fit diagnostics](https://raw.githubusercontent.com/yangzhangpsy/PsySummary/main/figs/f46.png)

</details>

### 3.4 Export reproducible analysis code

Show **Script** through View, then **right-click inside Script > Export** to save the Python analysis script and its seven dependencies:

`aggregateData.py`, `rtDist.py`, `cognitiveModels.py`, `cognitiveModelSpec.py`, `expression.py`, `dataPreparation.py`, and `poolingCDF.py`.

Keep these files together with the input data. Check paths and Python dependencies before running the exported script. Share the data, script, modules, and analysis settings needed to reproduce the calculation.

Analysis calls are recorded after the selected analysis completes successfully. Generated code covers supported data-processing and analysis operations, not every window layout, plot selection, or result-file action.

<details>
<summary>Export the recorded analysis code</summary>

![Export the Python analysis script and its matching numerical modules](https://raw.githubusercontent.com/yangzhangpsy/PsySummary/main/figs/f47.png)

</details>

## Validation and scope

The [supporting materials archive](Supporting_materials_20261025.zip) provides the tutorial inputs, saved settings, filtered data and summaries, GUI-exported scripts and matching numerical modules, five preprocessing comparisons, cognitive-model references and fitting benchmarks, and GUI import/replay comparisons. Extract the archive and follow its README to reproduce the supplied analyses. The portable replay and saved-reference checks do not require a GUI source checkout; regenerating the independent model references and fitting benchmarks requires R/rtdists.

The manuscript and Supplementary Validation S1 describe the validation procedures and scope. Agreement for specified calculations does not establish a universally appropriate filter or general parameter recovery.

PsySummary supports descriptive summaries, selected RT distributions, and independently grouped cognitive-model fits. It does not provide hierarchical cognitive-model estimation, trial-level parameter regression, automatic model selection, or general regression-based influence diagnostics such as Cook's distance. Export retained trial-level data for analyses outside this scope.

## Related manuscript

**Reaction time outlier handling in behavioral research. Part II: PsySummary software, validation, and tutorial**

Zhicheng Lin, Lian-Zi Xing, Tracy Xin Chen, Yue-Jie Chen, and Yang Zhang.
