import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.special as sp
from scipy.stats import poisson
from scipy.special import beta as beta_fn
import math
from scipy.special import factorial
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def portfolio_generator(num_obligors=100, total_portfolio_value=1000, 
                        shape_exposure=1, degrees_of_freedom=1.5): 
                        #seed=10):
    """
    Generates a sample portfolio with exposures and default probabilities.

    Parameters:
    - num_obligors: Number of obligors (default 100)
    - total_portfolio_value: Total portfolio exposure value (default 1000)
    - shape_exposure: Shape parameter for Weibull distribution for exposures (default 1)
    - degrees_of_freedom: Degrees of freedom for chi-squared distribution for default probabilities (default 1.5)
    - seed: Random seed for reproducibility (default 10)

    Returns:
    - portfolio_df: DataFrame containing the portfolio with exposures and default probabilities
    - summary_stats: Dictionary with summary statistics (mean exposure and mean default probability)
    """
    #np.random.seed(seed)

    # Generate exposures using a Weibull distribution
    average_exposure = total_portfolio_value / num_obligors
    scale_exposure = average_exposure / np.random.weibull(shape_exposure, num_obligors).mean()
    exposures = np.random.weibull(shape_exposure, num_obligors) * scale_exposure

    # Generate default probabilities using a chi-square distribution centered around 1%
    scale_factor = 0.07 / np.random.chisquare(degrees_of_freedom, num_obligors).max()
    default_probabilities = np.random.chisquare(degrees_of_freedom, num_obligors) * scale_factor
    default_probabilities = np.clip(default_probabilities, 0, 0.07)  # Ensure values within 0% - 7%

    # Create a DataFrame for the portfolio
    portfolio_df = pd.DataFrame({
        'Obligor': range(1, num_obligors + 1),
        'Exposure': exposures,
        'Default_Probability': default_probabilities
    })

    # Normalize exposures to match total portfolio value
    portfolio_df['Exposure'] *= total_portfolio_value / portfolio_df['Exposure'].sum()

    # Calculate mean values for summary statistics
    mean_exposure = portfolio_df['Exposure'].mean()
    mean_default_prob = (portfolio_df['Default_Probability'] * 100).mean()
    summary_stats = {
        'Mean Exposure': mean_exposure,
        'Mean Default Probability (%)': mean_default_prob
    }

    return portfolio_df, summary_stats


def portfolio_plot(portfolio_df, exposure_bins=50, default_prob_bins=50):
    """
    Plots the distributions of exposures and default probabilities for a given portfolio.

    Parameters:
    - portfolio_df: DataFrame containing the portfolio with exposures and default probabilities
    - exposure_bins: Number of bins for exposure histogram (default 50)
    - default_prob_bins: Number of bins for default probability histogram (default 50)
    """
    # Calculate means for marking in plots
    mean_exposure = portfolio_df['Exposure'].mean()
    mean_default_prob = (portfolio_df['Default_Probability'] * 100).mean()

    # Compute relative frequencies for exposures and default probabilities
    exposure_counts, exposure_bins_vals = np.histogram(portfolio_df['Exposure'], bins=exposure_bins)
    exposure_relative_freq = exposure_counts / len(portfolio_df) * 100

    default_prob_counts, default_prob_bins_vals = np.histogram(portfolio_df['Default_Probability'] * 100, bins=default_prob_bins)
    default_prob_relative_freq = default_prob_counts / len(portfolio_df) * 100

    # Plotting
    plt.figure(figsize=(14, 6))

    # Default Probability Distribution Plot
    plt.subplot(1, 2, 1)
    plt.bar(default_prob_bins_vals[:-1], default_prob_relative_freq, width=np.diff(default_prob_bins_vals), color='salmon', edgecolor='black', align='edge')
    plt.axvline(mean_default_prob, color='red', linestyle='--', label=f'Mean Default Probability ({mean_default_prob:.2f}%)')
    plt.xlabel('Default Probability (%)')
    plt.ylabel('Relative Frequency (%)')
    plt.title('Default Probability Distribution')
    plt.legend()

    # Exposure Distribution Plot
    plt.subplot(1, 2, 2)
    plt.bar(exposure_bins_vals[:-1], exposure_relative_freq, width=np.diff(exposure_bins_vals), color='skyblue', edgecolor='black', align='edge')
    plt.axvline(mean_exposure, color='blue', linestyle='--', label=f'Mean Exposure (${mean_exposure:.2f})')
    plt.xlabel('Exposure ($)')
    plt.ylabel('Relative Frequency (%)')
    plt.title('Exposure Distribution')
    plt.legend()

    plt.tight_layout()
    plt.show()



def getBC(N, k):
    return sp.comb(N, k)

def independentBinomialLossDistribution(N, M, p, c, alpha):
    U = np.random.uniform(0, 1, [M,N])
    defaultIndicator = 1 * np.less(U,p)
    lossDistribution = np.sort(np.dot(defaultIndicator, c), axis = None)
    return lossDistribution

def computeRiskMeasures(M, lossDistribution, alpha):
    expectedLoss = np.mean(lossDistribution)
    unExpectedLoss = np.std(lossDistribution)
    expectedShortfall = np.zeros([len(alpha)])
    var = np.zeros([len(alpha)])
    for n in range(len(alpha)):
        myQuantile = min(int(np.ceil(alpha[n] * (M - 1))), M - 1)
        var[n] = lossDistribution[myQuantile]
        if myQuantile < M - 1:
            expectedShortfall[n] = np.mean(lossDistribution[myQuantile:M])
        else:
            expectedShortfall[n] = var[n]
    return expectedLoss, unExpectedLoss, var, expectedShortfall   

def independentBinomialSimulation(N,M,p,c,alpha):
    lossDistribution = independentBinomialLossDistribution(N,M,p,c,alpha)
    el, ul, var, es = computeRiskMeasures(M, lossDistribution, alpha)
    return el, ul, var, es

def independentBinomialAnalytic(N, p, c, alpha):
    pmfBinomial = np.zeros(N+1)
    for k in range(0, N+1):
        pmfBinomial[k] = getBC(N,k)*(p**k)*((1-p)**(N-k))
    cdfBinomial = np.cumsum(pmfBinomial)
    varAnalytic = c*np.interp(alpha, cdfBinomial, np.linspace(0,N,N+1))
    esAnalytic = analyticExpectedShortfall(N, alpha, pmfBinomial, c)
    return pmfBinomial, cdfBinomial, varAnalytic, esAnalytic

def analyticExpectedShortfall(N, alpha, pmf, c):
    cdf = np.cumsum(pmf)
    numberDefaults = np.linspace(0,N,N+1)
    expectedShortfall = np.zeros(len(alpha))
    for n in range(0, len(alpha)):
        myAlpha = np.linspace(alpha[n], 1, 1000)
        loss = c*np.interp(myAlpha, cdf, numberDefaults)
        prob = np.interp(loss, numberDefaults, pmf)
        expectedShortfall[n] = np.dot(loss, prob)/np.sum(prob)
    return expectedShortfall

def independentPoissonLossDistribution(N,M,p,c,alpha):
    lam = -np.log(1-p)
    H = np.random.poisson(lam, [M,N])
    defaultIndicator = 1*np.greater_equal(H,1)
    lossDistribution = np.sort(np.dot(defaultIndicator, c), axis = None)
    return lossDistribution

def independentPoissonAnalytic(N,c,myLam,alpha):
    pmfPoisson = np.array([poisson.pmf(k, myLam) for k in range(N + 1)])
    cdfPoisson = np.cumsum(pmfPoisson)
    varAnalytic = c*np.interp(alpha, cdfPoisson, np.linspace(0,N,N+1))
    esAnalytic = analyticExpectedShortfall(N, alpha, pmfPoisson, c)
    return pmfPoisson, cdfPoisson, varAnalytic, esAnalytic

def computeBeta(a,b):
    return sp.gamma(a) * sp.gamma(b) / sp.gamma(a + b)

def betaBinomialAnalytic(N, c, a, b, alpha):
    pmfBeta = np.zeros(N+1)
    den = computeBeta(a,b)
    for k in range(0,N+1):
        pmfBeta[k] = getBC(N,k)*computeBeta(a+k, b+N-k)/den
    cdfBeta = np.cumsum(pmfBeta)
    varAnalytic = c*np.interp(alpha, cdfBeta, np.linspace(0,N,N+1))
    esAnalytic = analyticExpectedShortfall(N, alpha, pmfBeta, c)
    return pmfBeta, cdfBeta, varAnalytic, esAnalytic

def calibrate_beta_parameters_with_means(mean_p, var_p):
    a = mean_p * (mean_p * (1 - mean_p) / var_p - 1)
    b = (1 - mean_p) * (mean_p * (1 - mean_p) / var_p - 1)
    return a, b

def calibrate_beta_parameters_with_correlation(mean_p, correlation):
    a_b_sum = (1 - correlation) / correlation

    a = mean_p * a_b_sum
    b = (1 - mean_p) * a_b_sum

    return a, b

def poissonGammaAnalytic(N, c, a, b, alpha):
    pmfPoisson = np.zeros(N+1)
    q = np.divide(b, b+1)
    den = math.gamma(a)
    for k in range(0, N+1):
        num = np.divide(math.gamma(a+k), sp.factorial(k))
        pmfPoisson[k] = np.divide(num, den)*np.power(q,a)*np.power(1-q,k)
    cdfPoisson = np.cumsum(pmfPoisson)
    varAnalytic = c*np.interp(alpha, cdfPoisson, np.linspace(0,N,N+1))
    esAnalytic = analyticExpectedShortfall(N,alpha,pmfPoisson,c)
    return pmfPoisson,cdfPoisson,varAnalytic,esAnalytic

def poissonGammaMooment(a,b,momentNumber):
    q1 = np.divide(b,b+1)
    q2 = np.divide(b,b+2)
    if momentNumber ==1:
        myMoment = 1 - np.power(q1,a)
    if momentNumber == 2:
        myMoment = 1 - 2*np.power(q1,a) + np.power(q2,a)
    return myMoment
def poissonGammaCalibrate(x,pTarget,rhoTarget):
    if x[1] <= 0:
        return [100,100]
    M1 = poissonGammaMoment(x[0], x[1],1)
    M2 = poissonGammaMoment(x[0],x[1],2)
    f1 = pTarget - M1
    f2 = rhoTarget*(M1-(M2**2)) - (M2 - (M2 - (M1**2)))
    return [f1,f2]

def dirty_calibration(rho, p_mean):
    return [p_mean/(rho*(1-p_mean)), 1/(rho*(1-p_mean))]

def poissonGammaSimulation(N, M,c,a,b,alpha):
    lam = np.random.gamma(a, 1/b,M)
    H = np.zeros([M,N])
    for m in range(0,M):
        H[m,:] = np.random.poisson(lam[m],[N])
    lossIndicator = 1*np.greater_equal(H,1)
    lossDistribution = np.sort(np.dot(lossIndicator, c), axis = None)
    el, ul, var, es = computeRiskMeasures(M, lossDistribution, alpha)
    return el, ul, var, es
























