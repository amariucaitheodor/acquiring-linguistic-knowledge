# code from: https://www.statsmodels.org/devel/examples/notebooks/generated/linear_regression_diagnostics_plots.html
import numpy as np
import seaborn as sns
import statsmodels
from statsmodels.tools.tools import maybe_unwrap_results
from statsmodels.graphics.gofplots import ProbPlot
import matplotlib.pyplot as plt
from typing import Type

style_talk = 'seaborn-talk'  # refer to plt.style.available


class LMERegDiagnostic():
    """
    Diagnostic plots to identify potential problems in a linear regression fit.
    Mainly,
        a. non-linearity of data
        b. Correlation of error terms
        c. non-constant variance
        d. outliers
        e. collinearity

    Authors:
        Prajwal Kafle (p33ajkafle@gmail.com, where 3 = r)
        Does not come with any sort of warranty.
        Please test the code one your end before using.

        Matt Spinelli (m3spinelli@gmail.com, where 3 = r)
        (1) Fixed incorrect annotation of the top most extreme residuals in
            the Residuals vs Fitted and, especially, the Normal Q-Q plots.
        (2) Changed Residuals vs Leverage plot to match closer the y-axis
            range shown in the equivalent plot in the R package ggfortify.
        (3) Added horizontal line at y=0 in Residuals vs Leverage plot to
            match the plots in R package ggfortify and base R.
        (4) Added option for placing a vertical guideline on the Residuals
            vs Leverage plot using the rule of thumb of h = 2p/n to denote
            high leverage (high_leverage_threshold=True).
        (5) Added two more ways to compute the Cook's Distance (D) threshold:
            * 'baseR': D > 1 and D > 0.5 (default)
            * 'convention': D > 4/n
            * 'dof': D > 4 / (n - k - 1)
        (6) Fixed class name to conform to Pascal casing convention
        (7) Fixed Residuals vs Leverage legend to work with loc='best'

        Theodor Amariucai (amariucaitheodor@protonmail.com)
        (1) Added option to pass in a statsmodels.regression.mixed_linear_model.MixedLMResultsWrapper
        (2) Removed two of the plots (leverage, vif)
    """

    def __init__(self, results) -> None:
        """
        For a linear regression model, generates following diagnostic plots:

        a. residual
        b. qq
        c. scale location

        Args:
            results (Type[statsmodels.regression.mixed_linear_model.MixedLMResultsWrapper]):
                must be instance of statsmodels.regression.mixed_linear_model object

        Raises:
            TypeError: if instance does not belong to above object

        Example:
        >>> import numpy as np
        >>> import pandas as pd
        >>> import statsmodels.formula.api as smf
        >>> import assets.plots.thesis.diagnostics_linreg.Linear_Reg_Diagnostic

        >>> x = np.linspace(-np.pi, np.pi, 100)
        >>> y = 3*x + 8 + np.random.normal(0,1, 100)
        >>> df = pd.DataFrame({'x':x, 'y':y})
        >>> res = smf.ols(formula= "y ~ x", data=df).fit()
        >>> cls = Linear_Reg_Diagnostic(res)
        >>> cls(plot_context="seaborn-paper")

        In case you do not need all plots you can also independently make an individual plot/table
        in following ways

        >>> cls = Linear_Reg_Diagnostic(res)
        >>> cls.residual_plot()
        >>> cls.qq_plot()
        >>> cls.scale_location_plot()
        """

        if isinstance(results, statsmodels.regression.mixed_linear_model.MixedLMResultsWrapper) is False:
            raise TypeError(
                "result must be instance of statsmodels.regression.mixed_linear_model.MixedLMResultsWrapper object")

        self.results = maybe_unwrap_results(results)

        self.y_true = self.results.model.endog
        self.y_predict = self.results.fittedvalues
        self.xvar = self.results.model.exog
        self.xvar_names = self.results.model.exog_names

        self.residual = np.array(self.results.resid)
        self.residual_norm = self.residual - np.average(self.residual)
        self.residual_norm = self.residual / np.std(self.residual)
        self.nparams = len(self.results.params)
        self.nresids = len(self.residual_norm)

    def __call__(self, plot_context='seaborn-paper', **kwargs):
        with plt.style.context(plot_context):
            fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
            self.residual_plot(ax=ax[0])
            self.qq_plot(ax=ax[1])
            self.scale_location_plot(ax=ax[2])
            for a in ax:
                for item in ([a.title, a.xaxis.label, a.yaxis.label] +
                             a.get_xticklabels() + a.get_yticklabels()):
                    item.set_fontsize(12)
            fig.tight_layout()
            plt.show()

        return fig, ax

    def residual_plot(self, ax=None):
        """
        Residual vs Fitted Plot

        Graphical tool to identify non-linearity.
        (Roughly) Horizontal red line is an indicator that the residual has a linear pattern
        """
        if ax is None:
            fig, ax = plt.subplots()

        sns.residplot(
            x=self.y_predict,
            y=self.residual,
            lowess=True,
            scatter_kws={'alpha': 0.5},
            line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8},
            ax=ax)

        # annotations
        residual_abs = np.abs(self.residual)
        abs_resid = np.flip(np.argsort(residual_abs), 0)
        abs_resid_top_5 = abs_resid[:5]
        for i in abs_resid_top_5:
            ax.annotate(
                i,
                xy=(self.y_predict[i], self.residual[i]),
                color='C3')

        ax.set_title('Tukey-Anscombe', fontweight="bold")
        ax.set_xlabel('Fitted values')
        ax.set_ylabel('Residuals')
        return ax

    def qq_plot(self, ax=None):
        """
        Standarized Residual vs Theoretical Quantile plot

        Used to visually check if residuals are normally distributed.
        Points spread along the diagonal line will suggest so.
        """

        def __qq_top_resid(quantiles, top_residual_indices):
            """
            Helper generator function yielding the index and coordinates
            """
            offset = 0
            quant_index = 0
            previous_is_negative = None
            for resid_index in top_residual_indices:
                y = self.residual_norm[resid_index]
                is_negative = y < 0
                if previous_is_negative == None or previous_is_negative == is_negative:
                    offset += 1
                else:
                    quant_index -= offset
                x = quantiles[quant_index] if is_negative else np.flip(quantiles, 0)[quant_index]
                quant_index += 1
                previous_is_negative = is_negative
                yield resid_index, x, y

        if ax is None:
            fig, ax = plt.subplots()

        QQ = ProbPlot(self.residual_norm)
        fig = QQ.qqplot(line='45', alpha=0.5, lw=1, ax=ax)

        # annotations
        abs_norm_resid = np.flip(np.argsort(np.abs(self.residual_norm)), 0)
        abs_norm_resid_top_5 = abs_norm_resid[:5]
        for i, x, y in __qq_top_resid(QQ.theoretical_quantiles, abs_norm_resid_top_5):
            ax.annotate(
                i,
                xy=(x, y),
                ha='right',
                color='C3')

        ax.set_title('Normal Q-Q', fontweight="bold")
        ax.set_xlabel('Theoretical Quantiles')
        ax.set_ylabel('Standardized Residuals')
        return ax

    def scale_location_plot(self, ax=None):
        """
        Sqrt(Standarized Residual) vs Fitted values plot

        Used to check homoscedasticity of the residuals.
        Horizontal line will suggest so.
        """
        if ax is None:
            fig, ax = plt.subplots()

        residual_norm_abs_sqrt = np.sqrt(np.abs(self.residual_norm))

        ax.scatter(self.y_predict, residual_norm_abs_sqrt, alpha=0.5);
        sns.regplot(
            x=self.y_predict,
            y=residual_norm_abs_sqrt,
            scatter=False, ci=False,
            lowess=True,
            line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8},
            ax=ax)

        # annotations
        abs_sq_norm_resid = np.flip(np.argsort(residual_norm_abs_sqrt), 0)
        abs_sq_norm_resid_top_5 = abs_sq_norm_resid[:5]
        for i in abs_sq_norm_resid_top_5:
            ax.annotate(
                i,
                xy=(self.y_predict[i], residual_norm_abs_sqrt[i]),
                color='C3')

        ax.set_title('Scale-Location', fontweight="bold")
        ax.set_xlabel('Fitted values')
        ax.set_ylabel(r'$\sqrt{|\mathrm{Standardized\ Residuals}|}$');
        ax.grid(True)
        return ax
