# Piecewise Linear Regression (Python | Colab)


## Overview

**Piecewise Linear Regression**

Piecewise linear regression (also called segmented regression, or broken-stick regression) is a modeling technique used when a single straight line cannot fully describe the relationship between variables. Instead of fitting just one regression line to all of the data, the dataset is divided into segments, and a separate linear regression model is fitted to each segment. This allows the regression line to “bend” at specific points, called breakpoints.
When there is only one breakpoint, this special case is called bilinear regression.

***How it works***

*Identifying breakpoints, cut-offs, or thresholds in data*
- Detects the exact locations where the trend changes, which can help determine cut-off scores on psychological tests that are used to predict important outcomes.

*Comparing regression slopes before and after a breakpoint*
- Analyzes how the relationship between variables shifts across different segments, e.g., comparing growth rates before and after a policy intervention.

*Visualizing segmented trends*
- Generates plots where regression lines change slope at identified breakpoints, making patterns in the data easier to interpret.


## Features

- Automated breakpoint detection using `pwlf` (Piecewise Linear Fit)
- Two-segment linear regression models
- R² and slope reporting per segment
- Statistical significance test for slope difference via interaction terms
- Plotting data, regression lines, and breakpoints


## Input File Format

- The dataset must be in `.sav` format (SPSS)
- Ensure that column names in the `variable_pairs` list match the actual column names in your dataset


## Dependencies

- The script requires the following Python packages:

```bash
pip install pwlf pyreadstat matplotlib scikit-learn statsmodels 
```


## Example Output

- Location of the regression breakpoint
- R² values for Segment 1 and Segment 2
- Slopes and intercepts for each segment
- p-value for the slope difference between segments
- Visualization: 
  - Red segmented regression line
  - Green dashed line marking the breakpoint


## Notes

- All variable and file names in the code are generic


## Step by Step Code

- STEP 1: Install dependencies 

```bash
pip install pwlf pyreadstat matplotlib scikit-learn statsmodels 
```

- STEP 2: Upload .sav file 

```bash
from google.colab import files
uploaded = files.upload()
```

- STEP 3: Load dataset and imports
  -  Replace with actual name of your .sav dataset ❗❗❗

```bash
import pandas as pd
import pyreadstat

df, meta = pyreadstat.read_sav('DATASET.sav')
df.head()
```

- STEP 4: Import analysis libraries

```bash
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import numpy as np
import pwlf
```

- STEP 5: Test slope difference

```bash
def test_slope_difference(df, x_var, y_var, breakpoint):
    # Create a segmented version of the predictor
    df_segmented = df[[x_var, y_var]].dropna().copy()
    df_segmented['Segment'] = (df_segmented[x_var] >= breakpoint).astype(int)
    df_segmented['Interaction'] = df_segmented[x_var] * df_segmented['Segment']

    # Fit model with interaction term
    model = smf.ols(f"{y_var} ~ {x_var} + Segment + Interaction", data=df_segmented).fit()

    # Extract p-value and slope difference
    p_value = model.pvalues['Interaction']
    slope_diff = model.params['Interaction']

    return p_value, slope_diff
```

- STEP 6: Run segmented regression

```bash
def piecewise_analysis_segmented(df, x_var, y_var, n_segments=2):

    # Remove NaNs
    data = df[[x_var, y_var]].dropna()
    x = data[x_var].values
    y = data[y_var].values

    # Fit piecewise model
    model = pwlf.PiecewiseLinFit(x, y)
    model.fit(n_segments)
    breakpoints = model.fit_breaks

    # Generate smooth x values and predict corresponding y values
    x_hat = np.linspace(min(x), max(x), 100)
    y_hat = model.predict(x_hat)

    # Plotting
    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, alpha=0.4, label='Data')
    plt.plot(x_hat, y_hat, color='red', label='Piecewise regression')
    for bp in breakpoints[1:-1]:
        plt.axvline(x=bp, color='green', linestyle='--', label=f'Breakpoint: {bp:.2f}')
    plt.title(f'{y_var} vs. {x_var}')
    plt.xlabel(x_var)
    plt.ylabel(y_var)
    plt.legend()
    plt.grid(True)
    plt.show()

    # Assume n_segments = 2 → one breakpoint
    bp = breakpoints[1]

    # Segment 1
    mask1 = x <= bp
    x1 = x[mask1].reshape(-1, 1)
    y1 = y[mask1]

    # Segment 2
    mask2 = x > bp
    x2 = x[mask2].reshape(-1, 1)
    y2 = y[mask2]

    # Fit linear models
    linreg1 = LinearRegression().fit(x1, y1)
    linreg2 = LinearRegression().fit(x2, y2)

    y1_pred = linreg1.predict(x1)
    y2_pred = linreg2.predict(x2)

    r2_1 = r2_score(y1, y1_pred)
    r2_2 = r2_score(y2, y2_pred)

    slope1 = linreg1.coef_[0]
    slope2 = linreg2.coef_[0]

    intercept1 = linreg1.intercept_
    intercept2 = linreg2.intercept_

    # Statistical test
    p_value, slope_diff = test_slope_difference(df, x_var, y_var, bp)
    
    print(f'Breakpoint at x = {bp:.2f}')
    print(f'Segment 1 (x <= {bp:.2f}): R² = {r2_1:.4f}, slope = {slope1:.4f}')
    print(f'Segment 2 (x > {bp:.2f}): R² = {r2_2:.4f}, slope = {slope2:.4f}')
    print(f'Difference in slope: {slope_diff:.4f}')
    print(f'p-value for slope difference = {p_value:.4f}')

    return {
        'breakpoint': bp,
        'segment1_r2': r2_1,
        'segment2_r2': r2_2,
        'segment1_slope': slope1,
        'segment2_slope': slope2,
        'intercept1': intercept1,
        'intercept2': intercept2,
        'slope_diff': slope_diff,
        'p_value': p_value
}
```    

- STEP 7: Define variable pairs for analysis
  - Replace with actual column names from the .sav dataset❗❗❗

```bash
variable_pairs = [
    ('X1', 'Y1'),
    ('X1', 'Y2'),
    ('X2', 'Y1'),
    ('X2', 'Y2'),
]
```

- STEP 8: Run analysis and display results

```bash
results = []

for predictor, outcome in variable_pairs:
    print(f"\nAnalyzing: {outcome} predicted by {predictor}")
    res = piecewise_analysis_segmented(df, predictor, outcome, n_segments=2)
    results.append({
        'Predictor': predictor,
        'Outcome': outcome,
        'Breakpoint': res['breakpoint'],
        'Segment1_R2': res['segment1_r2'],
        'Segment2_R2': res['segment2_r2'],
        'Segment1_Slope': res['segment1_slope'],
        'Segment2_Slope': res['segment2_slope'],
    })

df_results = pd.DataFrame(results)
```
