All materials—including the PsySummary application, its source code, documentation, and validation datasets—will be available here shortly.

**Download options**

From our website
https://www.psybuilder.com

• Run PsySummary inside PsyBuilder (Toolboxes → Results Data Aggregation), or

• Download and install it as a standalone program.

_**Direct installers (available on this page)**_

• Windows: PsySummary--Win.zip

• macOS (Apple Silicon): PsySummary--MacSilicon.dmg

• Linux: come soon...

_**Folder structure**_

• PsyData/   PsySummary source code

• dataValidation/ Validation datasets




# PsySummary: An Open-Source GUI for Robust and Transparent Reaction Time Outlier Exclusion with Methodological Framework and Decision Tree Guidelines

<p align="center"> Zhicheng Lin<sup>1,2</sup>, lian-zi Xing<sup>1</sup>, Xin Chen<sup>1</sup>, Yue-Jie Chen<sup>1</sup>, Yang Zhang<sup>1</sup> </p>

<sup>1</sup> _Department of Psychology, Soochow University, Suzhou, Jiangsu, China 215000_
<br>
<sup>2</sup> _Department of Psychology, Yonsei University, Seoul, Republic of Korea, 03722_


* [Step 1: Raw Data Management](#1)
* [Step 2: Data Filtering](#2)
* [Step 3: Generating Summary Statistics](#3)


A Brief Tutorial
To guide researchers through implementing these methods in their own workflow, we now provide a step-by-step tutorial. This brief tutorial illustrates the implementation of robust RT data analysis through PsySummary, demonstrating its key capabilities for applying recommended OEPs and facilitating methodological transparency and reproducibility through two key features: saving and loading filter configurations, and generating reproducible analysis code.

Data Processing Framework
PsySummary implements a three-stage data processing pipeline for robust RT analysis:
1.	Raw Data Management: Import data files, define variables, and apply transformations
2.	Data Filtering: Implement appropriate outlier exclusion procedures
3.	Summary Statistics Generation: Produce statistical outputs and code

The following sections provide step-by-step instructions for each stage, with emphasis on implementing methodologically sound outlier exclusion procedures aligned with current best practices.

<h2 id="1">Step 1: Raw Data Management</h2>
Step 1.1 Loading Data Files
PsySummary accepts multiple common data formats including .mat, .txt, .dat, and its native .psydata format. To load your data:
1.	Select File → Load Data from the menu bar in the Data Summary window
2.	Navigate to your data file location, select the file, and click Open
 ![image](https://raw.githubusercontent.com/yangzhangpsy/PsySummary/main/figs/f1.png)

3.	To view the imported data matrix, select File → View Data
Note: The Data Viewer window displays variable names in the top first row, trial numbers in the left first column, and corresponding values in each cell.
 ![image](https://raw.githubusercontent.com/yangzhangpsy/PsySummary/main/figs/f2.png)

To preserve your data for future sessions:
4.	Select File → Save Data to store your data in the .psydata format, which can be efficiently read by PsySummary in subsequent analyses
 ![image](https://raw.githubusercontent.com/yangzhangpsy/PsySummary/main/figs/f3.png)

Step 1.2 Configuring the Interface
Before proceeding with analysis, optimize your workspace:
1.	Select View from the menu bar to configure which windows appear in the main interface: 
–	The Output Window displays processing information and notifications about successful operations or errors
–	The Script Window shows the automatically generated analysis code that documents all processing steps
Maintaining visibility of both windows is recommended for monitoring analysis progress and facilitating reproducible reporting.
 ![image](https://raw.githubusercontent.com/yangzhangpsy/PsySummary/main/figs/f4.png)

Step 1.3 Transforming Variables
RT distributions are typically positively skewed, and transformations can improve normality when required for parametric analyses (Ratcliff, 1993; Whelan, 2008). To transform variables:
1.	Select Toolbox → Transform Variable
2.	In the pop-up Compute Variable window:
–	Enter a name for the new, transformed variable in Target Variable (e.g., “log_RT”)
–	Specify the transformation formula (e.g., for logarithmic transformation of “target_1.rt”)
–	Click OK to execute the transformation
 ![image](https://raw.githubusercontent.com/yangzhangpsy/PsySummary/main/figs/f5.png)

3.	The transformed variable will automatically appear in the Variables panel for subsequent analysis.
 ![image](https://raw.githubusercontent.com/yangzhangpsy/PsySummary/main/figs/f6.png)

<h2 id="2">Step 2: Data Filtering</h2>
Step 2.1 Configuring Data Structure 
Before implementing outlier exclusion, configure the structure of your analysis:
1.	From the Variables panel, drag subject identifiers to the Rows box
2.	Drag condition variables to the Columns box
3.	Drag dependent measures (e.g., RT variables) to the Data box

**_Important_**: Each variable can only be assigned to one attribute box (Rows, Columns, or Data). Double-assignment will trigger an error message.
	
 ![image](https://raw.githubusercontent.com/yangzhangpsy/PsySummary/main/figs/f7.png)
 ![image](https://raw.githubusercontent.com/yangzhangpsy/PsySummary/main/figs/f8.png)
Step 2.2 Choosing Statistical Operation
By default, PsySummary calculates the mean for variables in the Data box. To change the calculation:
1.	Double-click on the “@Mean” indicator
2.	Select the desired operation from the dropdown menu (e.g., median, standard deviation)
3.	Click Run to generate the statistical results (without filtering; see the next step for filtering).
 ![image](https://raw.githubusercontent.com/yangzhangpsy/PsySummary/main/figs/f9.png)
Step 2.3 Implementing Data Filtering and Outlier Exclusion
This critical stage involves implementing methodologically sound outlier exclusion procedures based on current best practices.

Step 2.3.1 Filtering Based on Accuracy
First, remove error trials before addressing RT outliers:
1.	Click Define Filters
 ![image](https://raw.githubusercontent.com/yangzhangpsy/PsySummary/main/figs/f10.png)
2.	From Variable Names, select your accuracy variable (e.g., “target_1.acc”)
3.	Click the Check List option
4.	Select “1” (correct) and leave “0” (incorrect) unchecked
5.	Click Add Filter to add this criterion to the Filter List
 ![image](https://raw.githubusercontent.com/yangzhangpsy/PsySummary/main/figs/f11.png)
Step 2.3.2 Filtering Based on RT
PsySummary supports multiple outlier exclusion methods aligned with methodological recommendations:
6.	From Variable Names, select your RT variable (e.g., “target_1.rt”)
7.	Click the Range option
 ![image](https://raw.githubusercontent.com/yangzhangpsy/PsySummary/main/figs/f12.png)
8.	Configure the outlier exclusion by:
–	Setting the appropriate inequality (“<” for upper bound, “>” for lower bound)
–	Selecting the exclusion method from the dropdown menu:
•	Raw value: Apply absolute cutoffs (e.g., exclude RTs < 200 ms or > 3000 ms)
•	SDs: Traditional standard deviation method (e.g., exclude RTs > 2.5 SD from mean)
•	Shifting Z: Implement the nonrecursive shifting z-score criterion (Van Selst & Jolicoeur, 1994), which adjusts cutoffs based on sample size
•	MAD: Apply the median absolute deviation method (Leys et al., 2013), which is robust to initial outliers
 ![image](https://raw.githubusercontent.com/yangzhangpsy/PsySummary/main/figs/f13.png)
9.	For a comprehensive approach, you may define both lower and upper bounds:
–	Lower bound: Set inequality to “>” and value to minimum acceptable RT (e.g., 200 ms)
–	Upper bound: Add another filter with “<” and select appropriate method (e.g., Shifting Z or MAD)
10.	Click Add Filter after configuring each criterion
Note on selection: The shifting z-score method is particularly recommended for datasets with fewer than 100 trials per condition or with unequal trial counts across conditions (Thompson, 2006). The MAD method is recommended for heavily skewed distributions (Leys et al., 2013).
 ![image](https://raw.githubusercontent.com/yangzhangpsy/PsySummary/main/figs/f14.png)

Step 2.3.3 Filtering Based on Pooling Cumulative Distribution Function (CDF)
11.	From Variable Names, select the RT variable (e.g., “target_1.rt”)
12.	Click the Pooling CDF option 
Note on requirements: The pooling CDF method identifies outliers by constructing the pooled cumulative distribution function (Miller, 2024). This method requires at least 50 data points per condition/cell to ensure reliable CDF estimation.
13.	Click OK to confirm your filter configuration.
Note on filter ordering and logic: PsySummary applies filters sequentially in top-down order:
–	Ensure logical ordering (typically remove error trials first, then address RT outliers)
–	To adjust the order, select and drag conditions within the Filter List
–	For complex filtering logic, use the “and”/“or” options when adding multiple criteria
 ![image](https://raw.githubusercontent.com/yangzhangpsy/PsySummary/main/figs/f15.png)

Step 2.3.4 Saving and Loading Filter Configurations
To ensure consistency across analyses:
14.	Click Save Filter to store your filter configuration
15.	In future sessions, use Load Filter to apply identical procedures
 ![image](https://raw.githubusercontent.com/yangzhangpsy/PsySummary/main/figs/f16.png)

<h2 id="3">Step 3: Generating Summary Statistics</h2>
Step 3.1 Saving Results
After configuring your analysis and outlier exclusion procedures:
1.	Click Run to process the data and generate the summary table
 ![image](https://raw.githubusercontent.com/yangzhangpsy/PsySummary/main/figs/f17.png)
2.	If you selected the Pooling CDF option in the previous steps, after clicking Run you’ll be presented with a visualization window showing: a histogram of your RT data distribution; an overlay of the fitted mixture density model; and a visual indication of potential outliers in the tails.

At this point, you need to make an informed decision about how to proceed:
-	Option 1: Skip CDF pooling entirely. If the visualization suggests the method isn’t appropriate for your data (e.g., due to multimodality or insufficient sample size). Click Abort CDF Pooling to return to the main interface without applying this method.
-	Option 2: Fine-tune the outlier detection sensitivity. The parameter ω (omega) controls the threshold for identifying outliers. To adjust this parameter, click directly on different points of the plot to see how different ω values affect the fitted mixture density curve. Watch how the highlighted outlier regions change with different values: a smaller ω value will be more conservative (fewer outliers identified); a larger ω value will be more liberal (more outliers identified). If you’re uncertain about the optimal value, click Estimated ω to use the statistically derived optimal value based on your data characteristics.
-	Option 3: Apply the optimized outlier detection. Once you’re satisfied with the ω parameter setting and the resulting outlier identification, click OK to finalize this step and apply the CDF-based outlier removal.

Note: The CDF pooling method works best with datasets containing at least 75/100 observations per condition—based on our experience, it is best used with big data integrating across multiple studies. With smaller samples, the visualization may show artifactual patterns, and you might consider using the MAD or shifting z-score methods instead.
 ![image](https://raw.githubusercontent.com/yangzhangpsy/PsySummary/main/figs/f18.png)
3.	To use the results in other applications:
–	Click Clipboard to copy the table to your clipboard
–	Click Export to save results as a text file
  ![image](https://raw.githubusercontent.com/yangzhangpsy/PsySummary/main/figs/f19.png)

Step 3.2 Generating Reproducible Analysis Code
PsySummary automatically generates Python code documenting all analysis steps:
1.	View the generated code in the Script window
2.	Export the python code (.py) by right-click anywhere inside this Script window, and select Export

Including this code in your method section or supplementary materials enhances transparency and facilitates replication.
 ![image](https://raw.githubusercontent.com/yangzhangpsy/PsySummary/main/figs/f20.png)
      
