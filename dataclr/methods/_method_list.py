from dataclr.methods.anova_filter import ANOVA
from dataclr.methods.boruta_wrapper import BorutaMethod
from dataclr.methods.cdf_filter import CumulativeDistributionFunction
from dataclr.methods.chi2_filter import Chi2
from dataclr.methods.cohens_d_filter import CohensD
from dataclr.methods.cramers_v_filter import CramersV
from dataclr.methods.distance_correlation_filter import DistanceCorrelation
from dataclr.methods.entropy_filter import Entropy
from dataclr.methods.hyperopt_wrapper import HyperoptMethod
from dataclr.methods.kendall_correlation_filter import KendallCorrelation
from dataclr.methods.kurtosis_filter import Kurtosis
from dataclr.methods.linear_correlation_filter import LinearCorrelation
from dataclr.methods.mad_filter import MeanAbsoluteDeviation
from dataclr.methods.mic_filter import MaximalInformationCoefficient
from dataclr.methods.mrmr_filter import mRMR
from dataclr.methods.mutual_information_filter import MutualInformation
from dataclr.methods.optuna_wrapper import OptunaMethod
from dataclr.methods.shap_wrapper import ShapMethod
from dataclr.methods.skewness_filter import Skewness
from dataclr.methods.spearman_correlation_filter import SpearmanCorrelation
from dataclr.methods.variance_threshold_filter import VarianceThreshold
from dataclr.methods.z_score_filter import ZScore

filter_classes = [
    ANOVA,
    CumulativeDistributionFunction,
    Chi2,
    CohensD,
    CramersV,
    DistanceCorrelation,
    Entropy,
    KendallCorrelation,
    Kurtosis,
    LinearCorrelation,
    MeanAbsoluteDeviation,
    MaximalInformationCoefficient,
    mRMR,
    MutualInformation,
    Skewness,
    SpearmanCorrelation,
    VarianceThreshold,
    ZScore,
]

wrapper_classes = [HyperoptMethod, OptunaMethod, BorutaMethod, ShapMethod]
