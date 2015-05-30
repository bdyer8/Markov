import pymc
from pymc import DiscreteUniform, Exponential, deterministic, Poisson, Uniform, Normal
import numpy as np
from pymc.Matplot import plot
#from pymc.examples import disaster_model
from pymc import MCMC
from pylab import hist,show
#matplotlib qt


#%%

#disasters_array =   \
#     np.array([ 4, 5, 4, 0, 1, 4, 3, 4, 0, 6, 3, 3, 4, 0, 2, 6,
#                   3, 3, 5, 4, 5, 3, 1, 4, 4, 1, 5, 5, 3, 4, 2, 5,
#                   2, 2, 3, 4, 2, 1, 3, 2, 2, 1, 1, 1, 1, 3, 0, 0,
#                   1, 0, 1, 1, 0, 0, 3, 1, 0, 3, 2, 2, 0, 1, 1, 1,
#                   0, 1, 0, 1, 0, 0, 0, 2, 1, 0, 0, 0, 1, 1, 0, 2,
#                   3, 3, 1, 1, 2, 1, 1, 1, 1, 2, 4, 2, 0, 0, 1, 4,
#                   0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1])
                   
dataSet=np.ones(100)
#dataSet[20:60]=2
dataSet[60:100]=3
dataSet=dataSet+normal(0,.1,dataSet.size)

                   

#scatter(range(dataSet.size),dataSet,c=(.98,.4,.4))

changepoint = DiscreteUniform('changepoint', lower=0, upper=100)
early_mean=Normal('early_mean',2,1)
late_mean=Normal('late_mean',2,1)

@deterministic(plot=False)
def rate(s=changepoint, e=early_mean, l=late_mean):
    ''' Concatenate Normal means '''
    out = np.empty(len(dataSet))
    out[:s] = e
    out[s:] = l
    return out
    
datapoints = Normal('datapoints', mu=rate, tau=.1, value=dataSet, observed=True)

vars = [
        changepoint,
        early_mean,
        late_mean,
        datapoints]

M = MCMC(vars)
M.sample(iter=100000, burn=1000, thin=10)
#%%
hist(M.trace('late_mean')[:],100)
hist(M.trace('early_mean')[:],100)
hist(M.trace('changepoint')[:],100)



        
#        
##%%
#                   
#switchpoint = DiscreteUniform('switchpoint', lower=0, upper=110, doc='Switchpoint[year]')
#
#early_mean = Exponential('early_mean', beta=1.)
#late_mean = Exponential('late_mean', beta=1.)
#
#@deterministic(plot=False)
#def rate(s=switchpoint, e=early_mean, l=late_mean):
#    ''' Concatenate Poisson means '''
#    out = np.empty(len(disasters_array))
#    out[:s] = e
#    out[s:] = l
#    return out
#    
#disasters = Poisson('disasters', mu=rate, value=disasters_array, observed=True)
#
#
#M=MCMC(disaster_model)
#M.sample(iter=10000, burn=1000, thin=10)
#
#hist(M.trace('late_mean')[:])
#
#
#plot(M)

##%%
#import matplotlib.pyplot as plt
#from scipy import stats
#from pylab import *
#
#import matplotlib
#import matplotlib.font_manager as font_manager
#
#fontpath = '/Library/Fonts/MyriadPro-Regular.otf'
#
#prop = font_manager.FontProperties(fname=fontpath)
#matplotlib.rcParams['font.family'] = prop.get_name()
#matplotlib.rcParams['font.variant'] = 'small-caps'
#
#
#N = 50
#x = np.random.rand(N)
#av_mole = x
#
#HIG = 1+2*np.exp(x)+x*x+np.random.rand(N)
#area = np.pi * (15 * np.random.rand(N))**2
#
#mole_error = 0.1 + 0.1*np.sqrt(av_mole)
#hig_error = 0.1 + 0.2*np.sqrt(HIG)/10
#
#fig = plt.figure(1, facecolor='white',figsize=(10,7.5))
#ax = plt.subplot(1,1,1)
#
#obj = ax.scatter(av_mole, HIG, s=70, c=area, marker='o',cmap=plt.cm.jet, zorder=10)
#cb = plt.colorbar(obj)
#cb.set_label('Field Area (m2)',fontsize=20)
#
#ax.errorbar(av_mole, HIG, xerr=mole_error, yerr=hig_error, fmt='o',color='b')
#
#plt.xlabel('AVERAGE number of moles per sq. meter', fontsize = 18)
#plt.ylabel('Health Index for Gardeners (HIG)', fontsize = 18)
#plt.title('Mole population against gardeners health', fontsize = 24)
#
#slope, intercept, r_value, p_value, std_err = stats.linregress(av_mole, HIG)
#
#print 'slope = ', slope
#print 'intercept = ', intercept
#print 'r value = ', r_value
#print  'p value = ', p_value
#print 'standard error = ', std_err
#
#line = slope*av_mole+intercept
#plt.plot(av_mole,line,'m-')
#plt.title('Linear fit y(x)=ax+b, with a='+str('%.1f' % slope)+' and b='+str('%.1f' % intercept), fontsize = 24)
#
#plt.savefig('fig.svg')
