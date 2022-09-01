<!-- omit in toc -->
# *ECON 3032-01: Applied Econometrics* 
<!-- omit in toc -->
# *Ethan Jiang, Vanderbilt University*
---
# **Table of Contents**
- [**Table of Contents**](#table-of-contents)
    - [Basic Information](#basic-information)
    - [Chapter 1](#chapter-1)
      - [Review: F-distributions](#review-f-distributions)
---
### Basic Information

- Python v. 3.6+
- TA office hours via Zoom
- precede any emails with "ECON3032"
- midterms cannot be rescheduled
- grading
    - Midterm 1 - **20%**
    - Midterm 2 - **25%**
    - Final - **35%**
    - Homework - **20%**
        - 10-12 total problem sets, lowest two dropped
        - name homework files as "hwX_FirstLast.pdf"
---
### Chapter 1

- econometrics: applying theory to data
    - can be used to estimate *effect* and/or *magnitude of effect* of policy, or to *forecast economic variables*
- different distributions can result in the same regression result (see **Anscombe's Quartet** as an example)
    - thus data visualization is important
- empirical data types
    - **observational**: correlation & causation hard to differentiate
    - **experimental**: may yield correlation/causation info

#### Review: F-distributions
- *Ratio of Two Variances Test*
  - assume $\exists$ two populations $i=1,2$ where $P_i \sim N(\mu_i, \sigma_i)$
    - population parameters are unknown - use test statistics to infer their values
  - test is used when:
    - variance values of populations are directly in question
    - determine whether equal variances can be assumed in **$t$-test for difference in means**
  - note: $\chi^2$ is used for variance of *one* population
    - an $F$-statistic is obtained by taking the ratio of two $\chi^2$ distributed variables, e.g. $s_1$ and $s_2$ (see below)
  - normal test hypotheses
    $$H_0:\frac{\sigma_1^2}{\sigma_2^2}=1$$
    - $H_1$ can be $\ne 1$ (two-tailed), $\lt 1$, or $\gt 1$ (latter two are one-tailed)
      - for variance analysis for mean difference $t$-test, use $\ne$ (as direction of possible difference does not matter)
  - $F$-statistic
    $$F=\frac{s_1^2}{s_2^2}$$
  - confidence interval
    $$\frac{F}{F_{\alpha/2, k_1, k_2}} < \frac{\sigma_1^2}{\sigma_2^2} < \frac{F}{F_{1-(\alpha/2), k_1, k_2}}$$
    - $k_i=n_i-1$ (degrees of freedom)
      - pay attention to which $k$ is which
      - $k_1$ corresponds to d.o.f. of the *numerator* of the $F$-statistic (i.e. $s_1$), $k_2$ to the *denominator*
    - obtaining $F$-distribution values using Python
      - example with typical $\alpha$ for two-tailed test
        ```python
        from scipy.stats import f
        def main():
            alpha = .05
            df1 = ...
            df2 = ...
            s1 = ...
            s2 = ...
            testStat = (s1/s2)^2
            # denominator values for CI bounds
            # if testStat exceeds lower or upper bound (see notes), reject
            # isf: "inverse survival function"
            lowerCV = f.isf(alpha/2, df1, df2)
            upperCV = f.isf(1-alpha/2, df1, df2)

            # p-value: gives us smallest alpha at which we can reject
            # multiply by 2 for two-tailed test
            p = 2*f.sf(testStat, df1, df2)
        ```
    - 

$$
   \begin{align*} 
   test &= test \\
   &= test
   \end{align*}
   $$