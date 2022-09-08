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
      - [Simple Regression](#simple-regression)
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
            testStat = pow(s1/s2, 2)
            # denominator values for CI bounds
            # if testStat exceeds lower or upper bound (see notes), reject
            # isf: "inverse survival function"
            lowerCV = f.isf(alpha/2, df1, df2)
            upperCV = f.isf(1-alpha/2, df1, df2)

            # p-value: gives us smallest alpha at which we can reject
            # multiply by 2 for two-tailed test
            p = 2*f.sf(testStat, df1, df2)
        ```

#### Simple Regression
- definition of **regression model**
  - simple linear: $y=f(x)$
  - multiple linear: $y=f(x_1,x_2,...)$
- issues
  - need to determine factors affecting dependent var
  - functional relationship b/w $y$ and $x_i$?
  - are we capturing a *ceteris paribus* relationship?
  - what are the consequences of omitting vars from the regression?
- **simple regression model** (one explanatory variable)
  $$y=\beta_0+\beta_1 x+u$$
  - $u \in \mathbb{R}$ is the error term
    - describes "intrinsic randomness" of human behavior not predictable by independent var(s)
  - $y$ is the dependent variable (dependent on the independent variable $x$)
  - $\beta_0, \beta_1$ are population constant & slope, respectively
  - interpreting $\beta_1$
    - it should be obvious that $\frac{\delta y}{\delta x}=\beta_1$ when $\Delta u = 0$, i.e. $u \in \mathbb{R}$
- **assumptions**
  1. $E(u) = 0$
     - assumed WLOG
     - presence of $\beta_0$ allows this, since we can always adjust it s.t. this assumption is held)
  2. $E(u|x) = E(u) = 0 \; \forall x$
     - " $u$ is *mean independent* of $x$ "
     - i.e. assumption 1 holds at each value of $x$
     - important for lurking vars, which may seem to demonstrate a false relationship between $x$ and $y$ (whereas some var $z$ actually affects $x$ and $y$ but is only accounted for in $u$)
     - note that this assumption implies Assumption 1 and $Cov(x,u)=0$
- **population regression function**
  $$\begin{align*} E(y|x) &= E[(\beta_0+\beta_1 x+u)|x] \\\\ &= E(\beta_0|x) + E(\beta_1 x | x) + E(u|x) \\\\ &= \beta_0+\beta_1 x \end{align*}$$
  - implies the population regression function is a linear function of $x$
- estimating $\beta_0$ & $\beta_1$ (**ordinary least squares (OLS) method**)
  - recall: we call these estimates $\hat\beta_0$, $\hat\beta_1$
  - let us have a random sample $(x_i, y_i) \; | \; i=1,2,\dots,n$
  - our previous assumptions apply: $E(u)=0$, $Cov(x,u)=0$
    - $Cov(x,u)=E(xu)-E(x)E(u)$, so $E(xu)=0$
  - deriving conditions
  $$ \begin{align*} E(u) &= E(y - \beta_0 - \beta_1 x) = 0 \\\\ 0 &= \frac{\sum_{i=1}^n (y_i - \hat\beta_0 \hat\beta_1 x_i)}{n} \tag{1} \end{align*} $$
  $$ \begin{align*} E(xu) &= E(x(y - \beta_0 - \beta_1 x)) = 0 \\\\ 0 &= \frac{\sum_{i=1}^n [x_i(y_i - \hat\beta_0 - \hat\beta_1 x_i)]}{n} \tag{2} \end{align*} $$
    - derivation continues (see *Chapter 2* slides on Brightspace)
  - **resulting equations**
  $$ \hat\beta_0 = \hat y - \hat\beta_1\bar x $$
  $$ \hat\beta_1 = \frac{\sum_{i=1}^n (x_i-\bar x)(y_i-\bar y)}{\sum_{i=1}^n {(x_i-\bar x)}^2} = \frac{\text{Sample Cov}(x_i, y_i)}{\text{Sample Var}(x_i)} $$
- fitted/predicted values for $y_i$
  - defined by
  $$\hat y_i = \hat\beta_0 + \hat\beta_1 x_i$$
- residual of $y_i$
  - defined by
  $$\hat u_i = y_i - \hat y_i = y_i - \hat \beta_0 - \hat\beta_1 x_i$$
  - we are minimizing the sum of residual squares, i.e.
  $$\sum_{i=1}^n \hat u_i^2$$
- algebraic properties of OLS
  - sample average is same as fitted average
  $$\begin{align*} \sum_{i=1}^n \hat u_i &= 0 \\\\ y_i &= \hat y_i + \hat u_i \\\\ \implies n^{-1} \sum_{i=1}^n y_i &= n^{-1} \sum_{i=1}^n \hat y_i + 0 \\\\ \implies \bar y &= \hat y \end{align*}$$
  - sample covariance (and thus correlation) b/w any $x_i$ and $\hat u_i$ is 0
  $$\sum_{i=1}^n x_i\hat u_i = 0$$
  - $\hat y_i$ is a linear function of $x_i$ so the above sum is also true for $\hat y_i$ and $\hat u_i$
  - $(\bar x, \bar y)$ is always on the regression line
- **assumptions of OLS method**
  1. population model is linear in parameters
  2. given a random sample $(x_i, y_i) \; | \; i \in \{1, \dots, n\}$ that follows population model $y = \beta_0 + \beta_1 x + u$
  3. $x_i$ differs in at least one value
  4. $E(u|x)=0$
  5. $Var(u|x)=\sigma^2$, i.e. error has same variance for any given value of $x$ (*homoskedasticity*)
  - bias of estimators dependent on assumptions 1-4, assumption 5 needed for $Var(\hat\beta_1)$
- recall: estimator bias
  - $\hat \theta$ is an unbiased estimator of $\theta$ if
  $$E(\hat\theta) = \theta$$
  - if $\hat \theta$ biased, then
  $$Bias(\hat \theta) \coloneqq E(\hat \theta) - \theta$$
- variance of $\hat \beta_1$
  - recall that we define the total sum of squares (SST) by
  $$SST_x \coloneqq \sum_{i=1}^n {(x_i - \bar x)^2}$$
  - thus $Var(\hat \beta_1)$ is
  $$Var(\hat \beta_1) = \frac{\sigma^2}{SST_x}$$
  ($\sigma^2 = Var(u)$, i.e. *error variance*)
  - implications
    - $Var(\hat \beta_1)$ increases if $\sigma^2$ increases (residuals tend to deviate more) or $SST_x$ decreases (less variation between $x_i$, which makes it harder to predict slope)
- variance of $\hat \beta_0$
  - 
