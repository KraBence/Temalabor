import numpy as np
import math
from scipy.stats import norm
import pandas as pd

def calculate_R(PD):
    numerator = 1 - np.exp(-35 * PD)
    denominator = 1 - np.exp(-35)
    term1 = 0.03 * (numerator / denominator)
    term2 = 0.16 * (1 - (numerator / denominator))
    R = term1 + term2
    return R

def getGaussianY(N,M,p,rho):
    #Original: mean = 0, std = 1
    #Results form gdp:       mean = 0.6147826086956522 ,  std = 2.0027281164618476
    #Results from gdp folyo: mean = 0.09091380589777864 , std = 0.06073935449197835
    '''
    Mean, std method: 
    Hectic mean: 0.03881674477697323   Hectic var: 0.0012169976252323896 
    Quiet mean: 0.11187969634883446  Quiet var: 0.0031607742741719446
    
    GMM method: Estimated means of the two distributions: [0.06438739 0.15068682]
                Estimated variances of the two distributions: [0.00202484 0.00218029]
                Estimated mixture weights: [0.6926235 0.3073765]
    
    ''' 
    G = np.transpose(np.tile(np.random.normal(0.09091380589777864, np.power(0.06073935449197835,2),M),(N,1))) # the estimation from Becslesek.ipynb, results from GDP
    e = np.random.normal(0,1,[M,N])
    Y = math.sqrt(rho)*G + math.sqrt(1-rho)*e
    return Y

def getGaussianYCustom(N,M,p,rho, mu, var):
    G = np.transpose(np.tile(np.random.normal(mu, var,M),(N,1))) 
    e = np.random.normal(0,1,[M,N])
    Y = math.sqrt(rho)*G + math.sqrt(1-rho)*e
    return Y


def getY2r(N,M,p,myRho,rId,nu,P,isT):
    rhoVector = myRho[rId]
    rhoMatrix = np.tile(rhoVector,(M,1))
    G = np.transpose(np.tile(np.random.normal(0,1,M),(N,1)))
    e = np.random.normal(0,1,[M,N])
    systematic = np.multiply(np.sqrt(rhoMatrix),G)
    idiosyncratic = np.multiply(np.sqrt(1-rhoMatrix),e)
    if isT==1:
        W = np.transpose(np.sqrt(nu/np.tile(np.random.chisquare(nu,M),(N,1))))
        Y = np.multiply(W,systematic + idiosyncratic)
    else:
        Y = systematic + idiosyncratic
    return Y 

def calibrateGaussian(x,myP,targetRho):
    jointDefaultProb = ac.jointDefaultProbability(myP,myP,x)
    defaultCorrelation = np.divide(jointDefaultProb-myP**2,myP*(1-myP))
    return np.abs(defaultCorrelation-targetRho)

def oneFactorGaussianModel(N,M,p,c,rho,alpha):
    Y = getGaussianY(N,M,p,rho)
    K = norm.ppf(p)*np.ones((M,1))        
    lossIndicator = 1*np.less(Y,K)
    expected_defaults = lossIndicator.mean(axis=0).sum().astype(int)
    lossDistribution = np.sort(np.dot(lossIndicator,c),axis=None)
    el,ul,var,es=computeRiskMeasures(M,lossDistribution,alpha)
    return el,ul,var,es, expected_defaults   

def computeP(p,rho,g):
    num = norm.ppf(p)-np.multiply(np.sqrt(rho),g)
    pG = norm.cdata(np.divide(num,np.sqrt(1-rho)))
    return pG

def computeRiskMeasures(M, lossDistribution, alpha):
    lossDistribution = lossDistribution[~np.isnan(lossDistribution)]
    expectedLoss = np.mean(lossDistribution)
    unExpectedLoss = np.std(lossDistribution)
    expectedShortfall = np.zeros(len(alpha))
    var = np.zeros(len(alpha))
    for n in range(len(alpha)):
        index = int(np.ceil(alpha[n] * (M - 1)))
        if index < len(lossDistribution):
            expectedShortfall[n] = np.mean(lossDistribution[index:])
            var[n] = lossDistribution[index]
        else:
            expectedShortfall[n] = np.nan
            var[n] = np.nan
    return expectedLoss, unExpectedLoss, var, expectedShortfall  

def asrfDensity(x,p,rho):
    a = np.sqrt(np.divide(1-rho,rho))
    b = np.power(np.sqrt(1-rho)*norm.ppf(x)-norm.ppf(p),2)
    c = 0.5*(np.power(norm.ppf(x),2) - b/rho)
    return a*np.exp(c)

def asrfModel(myP,rho,c,alpha):
    myX = np.linspace(0.0001,0.9999,100)
    num = np.sqrt(1-rho)*norm.ppf(myX)-norm.ppf(myP)
    cdf = norm.cdf(num/np.sqrt(rho))
    pdf = asrfDensity(myX,myP,rho)
    varAnalytic = np.sum(c)*np.interp(alpha,cdf,myX)
    esAnalytic = asrfExpectedShortfall(alpha,myX,cdf,pdf,c,rho,myP)
    return pdf,cdf,varAnalytic,esAnalytic
    
def asrfExpectedShortfall(alpha, myX, cdf, pdf, c, rho, myP):
    expectedShortfall = np.zeros(len(alpha))
    for n in range(0, len(alpha)):
        myAlpha = np.linspace(alpha[n], 1, 1000)
        loss = np.interp(myAlpha, cdf, myX)
        loss = np.clip(loss, myX[0], myX[-1])  # Ensure valid range
        
        prob = np.interp(loss, myX, pdf)
        prob = np.maximum(prob, 1e-10)  # Avoid zero probabilities
        prob /= np.sum(prob)  # Normalize
        
        if np.sum(prob) > 0:
            expectedShortfall[n] = np.dot(loss, prob)
        else:
            expectedShortfall[n] = np.nan  # Handle edge case
    return expectedShortfall 



def convert_to_quarterly(date):

    date = pd.to_datetime(date)
    year = date.year
    
    if date.month in [1, 2, 3]:
        quarter = 'Q1'
    elif date.month in [4, 5, 6]:
        quarter = 'Q2'
    elif date.month in [7, 8, 9]:
        quarter = 'Q3'
    else:
        quarter = 'Q4'

    return f"{year}{quarter}"


month_map = {
    'január': '01',
    'február': '02',
    'március': '03',
    'április': '04',
    'május': '05',
    'június': '06',
    'július': '07',
    'augusztus': '08',
    'szeptember': '09',
    'október': '10',
    'november': '11',
    'december': '12'
}

def convert_date(date_str):
    parts = date_str.split()
    year = parts[0].strip('.')
    month = month_map[parts[1].strip('.')]
    day = parts[2].strip('.')
    return f"{year}-{month}-{day}"