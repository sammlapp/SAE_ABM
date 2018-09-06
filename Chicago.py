
# coding: utf-8

# # Chicago: "we makin' cars"
# Implement the Formula SAE contectualized design problem from Zurita et al

# ## TO DO: CHECK UNITS
# everywhere, esp when importing tables

# In[95]:


import numpy as np
import pandas as pd
from scipy.stats import chi
# from opteval import benchmark_func as bf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import time as timer
import threading
import pickle

# from sklearn import linear_model


# ## Helper Functions

# In[2]:


def cp(x): #make a copy instead of reference
    return copy.deepcopy(x)

def isNumber(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def isNaN(num):
    return num != num

def mean(x):
    return np.mean(x)

def norm(x):
    return float(np.linalg.norm(x))

def dist(x,y):
    return np.linalg.norm(np.array(x)-np.array(y))

def bounds(x,low,high):
    if x > high:
        return high
    if x < low:
        return low
    return x


# In[3]:


def makeAiScore():
    ai = np.random.normal(97,17)
    ai = bounds(ai, 40,150)
    return ai

def makeIqScore(): #this IQ ranges from 0 to 1, bc it is basically efficiency
    iq = np.random.normal(0.5,0.2)
    iq = bounds(iq, 0.1,1.0)
    return iq

def pickWorseScore(betterScore,worseScore,temperature):
    if temperature <=0: #never pick worse answers, and avoid devide by 0
        return False
    if np.random.uniform(0,1) < np.exp((betterScore-worseScore)/temperature):  #
        return True 
    return False

def calculateDecay(steps,T0=1.0,Tf=0.01):
    if T0<=Tf or T0<=0: 
        return 0
    return (Tf / float(T0) ) ** (1/steps)
  
def calculateAgentDecay(agent, steps):
    E_N = normalizedE(agent.kai.E)
    E_transformed = np.exp((E_N*-1)+3.3)
    startEndRatio = bounds(1/E_transformed, 1E-10,1)
    T0 = agent.temp
    TF = T0 * startEndRatio
    return calculateDecay(steps,T0,TF)


# ## constants and params

# In[4]:


complexSharing = True #if False, shareBasic() is used, eg strictly-greedy one-dimension jump 
commBonus = 10 #increasing the communication bonus makes successful communication more likely
commSd = 20 #increasing standard deviation of communication succcess makes it less predictable 
selfBias = 0 #increasing self bias will make agents choose their solutions more over others
startRange = 10
nDims = 2

SO_STRENGTH = 1
RG_STRENGTH = 1
TEAM_MEETING_COST = 1 #1 turn

ROUGHNESS = 0.02

VERBOSE = False

AVG_SPEED = 1
SD_SPEED = .5
MIN_SPEED = 0
AVG_TEMP = 1
SD_TEMP = 0.5


# ## Load Table of parameters

# In[5]:


#FIRST TIME:
paramsDF = pd.read_csv("./SAE/paramDB.csv")
paramsDF.columns = ['paramID','name','variable','team','kind','minV','maxV','used']
# paramsDF.at[3,"maxV"] = np.pi/4
# paramsDF.at[10,"maxV"] = np.pi/4
paramsDF.at[17,"maxV"] = np.pi /4
paramsDF.used = pd.to_numeric(paramsDF.used)
paramsDF = paramsDF.drop(columns=["paramID"],axis=1)
#remove unused variables... print(len(paramsDF))
paramsDF = paramsDF.loc[paramsDF.used > 0 ]
paramsDF.to_csv("./SAE/paramDBreduced.csv")
paramsDF = pd.read_csv("./SAE/paramDBreduced.csv")
print(len(paramsDF))
paramsDF = paramsDF.drop(["used"],axis=1)
paramsDF.head()


# In[6]:


paramsDF.loc[paramsDF.variable == "asw"]


# In[7]:


#logical vector for parameters with fixed min/max (TRUE), or min/max as f(p) (FALSE)
hasNumericBounds = [True if isNumber(row.minV) and isNumber(row.maxV) else False for i, row in paramsDF.iterrows()]


# In[8]:


# materialsDF = pd.read_csv("/Users/samlapp/Documents/THRED Lab/SAE/materials.csv")
# materialsDF.q = [int(1 + np.random.uniform(0.98,1.02)*materialsDF.iloc[i]['q']) for i in range(len(materialsDF))]
# materialsDF.to_csv("/Users/samlapp/Documents/THRED Lab/SAE/materialsTweaked.csv")
materialsDF = pd.read_csv("./SAE/materialsTweaked.csv")
materialsDF.head()


# In[9]:


tiresDF = pd.read_csv("./SAE/tires.csv")
tiresDF


# In[10]:


# motorsDF = pd.read_csv("/Users/samlapp/Documents/THRED Lab/SAE/motors.csv")
# # first time: we want to make motors with the same power slightly different:
# motorsDF.Power = [int(1 + np.random.uniform(0.98,1.02)*motorsDF.iloc[i]['Power']) for i in range(len(motorsDF))]
# motorsDF.to_csv("/Users/samlapp/Documents/THRED Lab/SAE/motorsTweaked.csv")
enginesDF = pd.read_csv("./SAE/motorsTweaked.csv")
print("unique" if len(enginesDF)-len(np.unique(enginesDF.Power)) == 0 else "not uniuqe")
enginesDF.columns = ["ind","id","name","le","we","he","me","Phi_e","T_e"]
enginesDF.head()


# In[11]:


# susDF = pd.read_csv("/Users/samlapp/Documents/THRED Lab/SAE/suspension.csv")
# susDF.krsp = [int(np.random.uniform(0.98,1.02)*susDF.iloc[i]['krsp']) for i in range(len(susDF))]
# susDF.kfsp = susDF.krsp
# susDF.to_csv("/Users/samlapp/Documents/THRED Lab/SAE/suspensionTweaked.csv")
susDF = pd.read_csv("./SAE/suspensionTweaked.csv")
print("unique" if len(susDF)-len(np.unique(susDF.krsp)) == 0 else "not uniuqe")
susDF = susDF.drop(columns=[susDF.columns[0]])
susDF.head()


# In[12]:


# brakesDF = pd.read_csv("/Users/samlapp/Documents/THRED Lab/SAE/brakes.csv")
# brakesDF.columns = [a.strip() for a in brakesDF.columns]
# brakesDF.rbrk = [np.random.uniform(0.98,1.02)*brakesDF.iloc[i]['rbrk'] for i in range(len(brakesDF))]
# brakesDF.to_csv("/Users/samlapp/Documents/THRED Lab/SAE/brakesTweaked.csv")
brakesDF = pd.read_csv("./SAE/brakesTweaked.csv")
print("unique" if len(brakesDF)-len(np.unique(brakesDF['rbrk'])) == 0 else "not uniuqe")
brakesDF = brakesDF.drop(columns=[brakesDF.columns[0]])
brakesDF.head()


# In[13]:


paramsDF.variable


# In[14]:


class Params:
    def __init__(self,v = paramsDF):
        self.vars = v.variable
        self.team = v.team
        for i, row in v.iterrows():
            setattr(self, row.variable.strip(),-1)
p = Params()
for v in p.vars:
    value = np.random.uniform()
    setattr(p,v,value)
paramsDF.loc[paramsDF.variable=="hrw"]["team"][0]

teams = np.unique(paramsDF.team)
teamDimensions = [[row.team == t for i, row in paramsDF.iterrows()] for t in teams]
teamDictionary = {}
for i in range(len(teams)):
    teamDictionary[teams[i]] = teamDimensions[i]
paramList = np.array(paramsDF.variable)


# In[15]:


#convert parameter vector to Parameter object
def asParameters(pList):
    p = Params()
    pNames = paramsDF.variable
    for i in range(len(pList)):
        pName = pNames[i]
        pValue = pList[i]
        setattr(p,pName,pValue)
    return p

def asVector(params):
    r = np.zeros(len(paramsDF))
    for i in range(len(paramsDF)):
        pName = paramsDF.variable[i]
        pValue = getattr(params,pName)
        r[i] = pValue
    return r


# ## Objective Subfunctions

# ### constants
# 
# The car’s top velocity vcar is 26.8 m/s (60 mph).
# 
# The car’s engine speed x_e is 3600 rpm. 
# 
# The density of air q_air during the race is 1.225 kg/m3. 
# 
# The track radio of curvature r_track is 9 m. 
# 
# The pressure applied to the brakes Pbrk is 1x10^7 Pa
# 

# In[16]:


#scale parameters to go between unit cube (approximately) and SI units
paramMaxValues = []


# In[17]:


v_car = 26.8 #m/s (60 mph)
w_e = 3600 * 60 * 2 *np.pi #rpm  to radians/sec 
rho_air = 1.225 #kg/m3.
r_track = 9 #m
P_brk = 10**7 #Pascals
C_dc = 0.04 #drag coefficient of cabin
gravity = 9.81 #m/s^2


# In[18]:


#mass (minimize)
def mrw(p):
    return p.lrw * p.wrw *p.hrw * p.qrw
def mfw(p):
    return p.lfw * p.wfw *p.hfw * p.qfw
def msw(p):
    return p.lsw * p.wsw *p.hsw * p.qsw
def mia(p):
    return p.lia * p.wia *p.hia * p.qia
def mc(p):
    return 2*(p.hc*p.lc*p.tc + p.hc*p.wc*p.tc + p.lc*p.hc*p.tc)*p.qc
def mbrk(p):
    #CHRIS missing parameters: how is mbrk calculated? assuming lrw*rho
    return p.lbrk * p.wbrk * p.hbrk * p.qbrk
def mass(p): #total mass, minimize
    mass = mrw(p) + mfw(p) + 2 * msw(p) + 2*p.mrt + 2*p.mft + p.me + mc(p) + mia(p) + 4*mbrk(p) + 2*p.mrsp + 2*p.mfsp
    return mass


# In[19]:


#center of gravity height, minimize
def cGy(p): 
    t1 =  (mrw(p)*p.yrw + mfw(p)*p.yfw+ p.me*p.ye + mc(p)*p.yc + mia(p)*p.yia) / mass(p)
    t2 = 2* (msw(p)*p.ysw + p.mrt*p.rrt + p.mft*p.rft + mbrk(p)*p.rft + p.mrsp*p.yrsp + p.mfsp*p.yfsp) / mass(p)
    
    return t1 + t2  


# In[20]:


#Drag (minimize) and downforce (maximize)
def AR(w,alpha,l): #aspect ratio of a wing
    return w* np.cos(alpha) / l
    
def C_lift(AR,alpha): #lift coefficient of a wing
    return 2*np.pi* (AR / (AR + 2)) * alpha

def C_drag(C_lift, AR): #drag coefficient of wing
    return C_lift**2 / (np.pi * AR)

def F_down_wing(w,h,l,alpha,rho_air,v_car): #total downward force of wing
    wingAR = AR(w,alpha,l)
    C_l = C_lift(wingAR, alpha)
    return 0.5 * alpha * h * w * rho_air * (v_car**2) * C_l

def F_drag_wing(w,h,l,alpha,rho_air,v_car): #total drag force on a wing
    wingAR = AR(w,alpha,l)
#     print(wingAR)
    C_l = C_lift(wingAR, alpha)
#     print(C_l)
    C_d = C_drag(C_l,wingAR)
#     print(C_d)
    return F_drag(w,h,rho_air,v_car,C_d)
    
def F_drag(w,h,rho_air,v_car,C_d):
    return 0.5*w*h*rho_air*v_car**2*C_d

def F_drag_total(p): #total drag on vehicle
    cabinDrag = F_drag(p.wc,p.hc,rho_air,v_car,C_dc)
    rearWingDrag = F_drag_wing(p.wrw,p.hrw,p.lrw,p.arw,rho_air,v_car)
    frontWingDrag = F_drag_wing(p.wfw,p.hfw,p.lfw,p.afw,rho_air,v_car)
    sideWingDrag = F_drag_wing(p.wsw,p.hsw,p.lsw,p.asw,rho_air,v_car)
    return rearWingDrag + frontWingDrag + 2* sideWingDrag + cabinDrag

def F_down_total(p): #total downforce
    downForceRearWing = F_down_wing(p.wrw,p.hrw,p.lrw,p.arw,rho_air,v_car)
    downForceFrontWing = F_down_wing(p.wfw,p.hfw,p.lfw,p.afw,rho_air,v_car)
    downForceSideWing = F_down_wing(p.wsw,p.hsw,p.lsw,p.asw,rho_air,v_car)
    return downForceRearWing + downForceFrontWing + 2*downForceSideWing


# In[21]:


#acceleration (maximize)
def rollingResistance(p,tirePressure,v_car):
    C = .005 + 1/tirePressure * (.01 + .0095 * (v_car**2))
    return C * mass(p) * gravity

def acceleration(p):
    mTotal = mass(p)
    tirePressure = p.Prt #CHRIS should it be front or rear tire pressure?
    total_resistance = F_drag_total(p) + rollingResistance(p, tirePressure,v_car)
    
    w_wheels = v_car / p.rrt #rotational speed of rear tires
    
    efficiency = total_resistance * v_car / p.Phi_e
    
    torque = p.T_e
    
    #converted units of w_e from rpm to rad/s !!!
    F_wheels = torque * efficiency * w_e /(p.rrt * w_wheels) 
    
    return (F_wheels - total_resistance) / mTotal
# acceleration(p)


# In[22]:


#crash force (minimize)
def crashForce(p):
    return np.sqrt(mass(p) * v_car**2 * p.wia * p.hia * p.Eia / (2*p.lia))


# In[23]:


#impact attenuator volume (minimize)
def iaVolume(p):
    return p.lia*p.wia*p.hia


# In[24]:


#corner velocity (maximize)
y_suspension = 0.05 # m
dydt_suspension = 0.025 #m/s 
def suspensionForce(k,c):
    return k*y_suspension + c*dydt_suspension

def cornerVelocity(p):
    F_fsp = suspensionForce(p.kfsp,p.cfsp)
    F_rsp = suspensionForce(p.krsp,p.crsp)
    downforce = F_down_total(p)
    mTotal = mass(p)
    
    #CHRIS again, rear tire pressure?
    C = rollingResistance(p,p.Prt,v_car)
    forces = downforce+mTotal*gravity-2*F_fsp-2*F_rsp
    if forces < 0: 
        return 0
    return np.sqrt( forces * C * r_track / mTotal )
# cornerVelocity(p)


# In[25]:


#breaking distance (minimize)
def breakingDistance(p):
    mTotal = mass(p)
    C = rollingResistance(p,p.Prt,v_car)
    
    #CHRIS need c_brk break friction coef, and A_brk (rectangle or circle?)
    #breaking torque
    A_brk = p.hbrk * p.wbrk
    c_brk = .37 #?   most standard brake pads is usually in the range of 0.35 to 0.42
    Tbrk = 2 * c_brk * P_brk * A_brk * p.rbrk
    
    #y forces:
    F_fsp = suspensionForce(p.kfsp,p.cfsp)
    F_rsp = suspensionForce(p.krsp,p.crsp)
    Fy = mTotal*gravity + F_down_total(p) - 2 * F_rsp - 2*F_fsp
    
    #breaking accelleration
    #CHRIS front and rear tire radius are same? (rrt and rft)
    a_brk = Fy * C / mTotal + 4*Tbrk*C/(p.rrt*mTotal)
    
    #breaking distance
    return v_car**2 / (2*a_brk)
# breakingDistance(p)


# In[26]:


#suspension acceleration (minimize)
def suspensionAcceleration(p):
    Ffsp = suspensionForce(p.kfsp,p.cfsp)
    Frsp = suspensionForce(p.krsp,p.crsp)
    mTotal = mass(p)
    Fd = F_down_total(p)
    return (2*Ffsp - 2*Frsp - mTotal*gravity - Fd)/mTotal
# suspensionAcceleration(p)


# In[27]:


#pitch moment (minimize)
def pitchMoment(p):
    Ffsp = suspensionForce(p.kfsp,p.cfsp)
    Frsp = suspensionForce(p.krsp,p.crsp)
    
    downForceRearWing = F_down_wing(p.wrw,p.hrw,p.lrw,p.arw,rho_air,v_car)
    downForceFrontWing = F_down_wing(p.wfw,p.hfw,p.lfw,p.afw,rho_air,v_car)
    downForceSideWing = F_down_wing(p.wsw,p.hsw,p.lsw,p.asw,rho_air,v_car)
    
    #CHRIS assuming lcg is lc? and lf is ?
    lcg = p.lc
    lf = 0.5
    return 2*Ffsp*lf + 2*Frsp*lf + downForceRearWing*(lcg - p.lrw) - downForceFrontWing*(lcg-p.lfw) - 2*downForceSideWing*(lcg-p.lsw)  
# pitchMoment(p)


# ## Global Objective

# In[28]:


#Global objective: linear sum of objective subfunctions
#sub-objectives to maximize will be mirrored *-1 to become minimizing

subObjectives = [mass,cGy,F_drag_total,F_down_total,acceleration,crashForce,iaVolume,cornerVelocity,breakingDistance,suspensionAcceleration,pitchMoment]  
alwaysMinimize = [1,1,1,-1,-1,1,1,-1,1,1,1] #1 for minimizing, -1 for maximizing
weightsNull = np.ones(len(subObjectives)) / len(subObjectives)
weights1 = np.array([14,1,20,30,10,1,1,10,10,2,1])/100
weights2 = np.array([25,1,15,20,15,1,1,15,5,1,1])/100
weights3 = np.array([14,1,20,15,25,1,1,10,10,2,1])/100

weightsCustom = np.array([14,1,20,30,11,1,1,10,10,2,0])/100 #pitch moment is zero bc incorrect eqn

def objectiveDetailedNonNormalized(p,weights):
    score = 0
    subscores = []
    for i in range(len(subObjectives)):
        obj = subObjectives[i]
        subscore = obj(p)
        subscores.append(subscore)
        score += weights[i]*alwaysMinimize[i]*subscore
    return score,subscores

# subscoreMean = np.zeros(len(subObjectives))
# subscoreSd = np.ones(len(subObjectives))

def objective(p,weights):
    score = 0
    for i in range(len(subObjectives)):
        obj = subObjectives[i]
        subscore= obj(p)
        normalizedSubscore = (subscore - subscoreMean[i]) / subscoreSd[i]
        score += weights[i]*alwaysMinimize[i]*normalizedSubscore
    return score

def objectiveDetailed(p,weights):
    score = 0
    subscores = []
    for i in range(len(subObjectives)):
        obj = subObjectives[i]
        subscore= obj(p)
        normalizedSubscore = (subscore - subscoreMean[i]) / subscoreSd[i]
        subscores.append(normalizedSubscore)
        score += weights[i]*alwaysMinimize[i]*normalizedSubscore
    return score, subscores


# ## Constraints

# ## constraints not done
# I didnt actually do the constraint functions yet just bounds
# 

# In[29]:


#a list with all the min-max functions (!) which can be called to return max and min value as f(p)
minMaxParam = [None for i in range(len(paramsDF))]
def wrw(p):
    minV = 0.300
    maxV = r_track - 2 * p.rrt
    
    return minV, maxV
minMaxParam[paramsDF.loc[paramsDF.variable=="wrw"].index[0]] = wrw

# def xrw(p):
#     minV = p.lrw / 2
#     maxV = .250 - minV
#     return minV, maxV
# minMaxParam[paramsDF.loc[paramsDF.variable=="xrw"].index[0]] = xrw

def yrw(p):
    minV = .5 + p.hrw / 2
    maxV = 1.2 - p.hrw / 2
    return minV, maxV
minMaxParam[paramsDF.loc[paramsDF.variable=="yrw"].index[0]] = yrw
wheelSpace = .1 #?? don't have an equation for this rn, min is .075

aConst = wheelSpace
def lfw(p):
    minV = .05 
    maxV = .7 - aConst
    return minV, maxV
minMaxParam[paramsDF.loc[paramsDF.variable=="lfw"].index[0]] = lfw

f_track = 3 # bounds: 3, 2.25 m 
def wfw(p):
    minV = .3
    maxV = f_track
    return minV, maxV
minMaxParam[paramsDF.loc[paramsDF.variable=="wfw"].index[0]] = wfw

# def xfw(p):
#     minV = p.lrw + p.rrt + p.lc + p.lia + p.lfw/2
#     maxV = .25 + p.rrt + p.lc + p.lia + p.lfw/2
#     return minV, maxV
# minMaxParam[paramsDF.loc[paramsDF.variable=="xfw"].index[0]] = xfw

xConst = .030 #ground clearance 19 to 50 mm
def yfw(p):
    minV = xConst + p.hfw / 2
    maxV = .25 - p.hfw/2
    return minV, maxV
minMaxParam[paramsDF.loc[paramsDF.variable=="yfw"].index[0]] = yfw

# def xsw(p):
#     minV = p.lrw + 2*p.rrt + aConst + p.lsw / 2
#     maxV = .250 + 2*p.rrt + aConst + p.lsw / 2
#     return minV, maxV
# minMaxParam[paramsDF.loc[paramsDF.variable=="xsw"].index[0]] = xsw

def ysw(p):
    minV = xConst + p.hsw/2
    maxV = .250 - p.hsw/2
    return minV, maxV
minMaxParam[paramsDF.loc[paramsDF.variable=="ysw"].index[0]] = ysw

# def xrt(p):
#     minV = p.lrw + p.rrt
#     maxV = .250 + p.rrt 
#     return minV, maxV
# minMaxParam[paramsDF.loc[paramsDF.variable=="xrt"].index[0]] = xrt

# def xft(p):
#     minV = p.lrw + p.rrt + p.lc
#     maxV = .250 + p.rrt  + p.lc
#     return minV, maxV
# minMaxParam[paramsDF.loc[paramsDF.variable=="xft"].index[0]] = xft

# def xe(p):
#     minV = p.lrw + p.rrt - p.le / 2
#     maxV = p.lrw + aConst + p.rrt - p.le / 2
#     return minV, maxV
# minMaxParam[paramsDF.loc[paramsDF.variable=="xe"].index[0]] = xe

def ye(p):
    minV = xConst + p.he / 2
    maxV = .5 - p.he / 2
    return minV, maxV
minMaxParam[paramsDF.loc[paramsDF.variable=="ye"].index[0]] = ye

def hc(p):
    minV = .500
    maxV = 1.200 - xConst
    return minV, maxV
minMaxParam[paramsDF.loc[paramsDF.variable=="hc"].index[0]] = hc

# def xc(p):
#     minV = p.lrw + p.rrt + p.lc / 2
#     maxV = .250 + p.rrt + p.lc / 2
#     return minV, maxV
# minMaxParam[paramsDF.loc[paramsDF.variable=="xc"].index[0]] = xc

def yc(p):
    minV = xConst + p.hc / 2
    maxV = 1.200 - p.hc / 2
    return minV, maxV
minMaxParam[paramsDF.loc[paramsDF.variable=="yc"].index[0]] = yc

def lia(p):
    minV = .2
    maxV = .7  - p.lfw # what is l_fr?
    return minV, maxV
minMaxParam[paramsDF.loc[paramsDF.variable=="lia"].index[0]] = lia

# def xia(p):
#     minV = p.lrw + p.rrt + p.lc + p.lia / 2
#     maxV = .250 + p.rrt + p.lc + p.lia/ 2
#     return minV, maxV
# minMaxParam[paramsDF.loc[paramsDF.variable=="xia"].index[0]] = xia

def yia(p):
    minV = xConst + p.hia / 2
    maxV = 1.200 - p.hia / 2
    return minV, maxV
minMaxParam[paramsDF.loc[paramsDF.variable=="yia"].index[0]] = yia

def yrsp(p):
    minV = p.rrt
    maxV = p.rrt * 2
    return minV, maxV
minMaxParam[paramsDF.loc[paramsDF.variable=="yrsp"].index[0]] = yrsp

def yfsp(p):
    minV = p.rft
    maxV = p.rft * 2
    return minV, maxV
minMaxParam[paramsDF.loc[paramsDF.variable=="yfsp"].index[0]] = yfsp

#test:
# for f in minMaxParam:
#     if f is not None:
#         print(f(p))


# In[30]:


def getAttr(obj):
    return [a for a in dir(obj) if not a.startswith('__')]


# In[31]:


def findMaterialByDensity(rho):
    differences = abs(np.array(materialsDF.q) - rho)
    material = materialsDF.iloc[np.argmin(differences)]
    return material.Code, material.q, material.E
def findTireByRadius(radius):
    differences = abs(np.array(tiresDF.radius) - radius)
    tire = tiresDF.iloc[np.argmin(differences)]
    return tire.ID, tire.radius, tire.mass

def findEngineByPower(power):
    differences = abs(np.array(enginesDF.Phi_e) - power)
    engine = enginesDF.loc[np.argmin(differences)]
    return engine

def findSuspensionByK(k):
    differences = abs(np.array(susDF.krsp) - k)
    sus = susDF.loc[np.argmin(differences)]
    return sus

def findBrakesByR(r): #what is the driving variable for brakes??? r?
    differences = abs(np.array(brakesDF.rbrk) - r)
    brakes = brakesDF.loc[np.argmin(differences)]

    return brakes


# In[34]:


def constrain(p,dimsToConstrain=np.ones(len(paramsDF))):
#     p_attribute = getAttr(p)
    paramIndices = [i for i in range(len(dimsToConstrain)) if dimsToConstrain[i] ==1]
    for i in paramIndices: #range(len(paramsDF)): # we need to do the equations bounds last
#         if not dimsToConstrain[i]: #we don't need to check this dimension, it didn't change
#             continue
        param = paramsDF.loc[i]
        variable = param.variable
        value = getattr(p,variable)
        if param.kind == 1: #continuous param with min and max
            if hasNumericBounds[i]:
                newValue = bounds(value,float(param["minV"]),float(param["maxV"]))
                setattr(p,variable,newValue)
            #do the equation ones after setting all other parameters
                
        elif param.kind == 2: #choose a material based on density
            materialID,density,modulusE = findMaterialByDensity(value)
            setattr(p,variable,density)
            #find other variables that are driven by this one:
            for i, otherParam in paramsDF[paramsDF.team == variable].iterrows():#dimension is driven by this one
                setattr(p,otherParam.variable,modulusE)    
        elif param.kind == 3: #choose tires
            tireID,radius,weight = findTireByRadius(value)
            setattr(p,variable,radius)
            #find other variables that are driven by this one:
            for i, otherParam in paramsDF[paramsDF.team == variable].iterrows():
                setattr(p,otherParam.variable,weight)
        elif param.kind == 4: #choose motor
            tableRow= findEngineByPower(value) #Phi_e,l_e,w_e,h_e,T_e,m_e
            setattr(p,variable,tableRow[variable])
            #find other variables that are driven by this one:
            for i, otherParam in paramsDF[paramsDF.team == variable].iterrows():
                setattr(p,otherParam.variable,tableRow[otherParam.variable])
        elif param.kind == 5: #choose brakes
            tableRow = findBrakesByR(value) # r is driving variable

            setattr(p,variable,tableRow[variable]) #df columns need to be same as variable names
            #find other variables that are driven by this one:
            for i, otherParam in paramsDF[paramsDF.team == variable].iterrows():
            #their "team" is THIS VAR: dimension is driven by this
                setattr(p,otherParam.variable,tableRow[otherParam.variable])

        elif param.kind == 6: #choose suspension
            tableRow = findSuspensionByK(value) #kfsp, cfsp, mfsp
            setattr(p,variable,tableRow[variable])
            #find other variables that are driven by this one:
            for i, otherParam in paramsDF[paramsDF.team == variable].iterrows():#their "team" is THIS VAR: dimension is driven by this
                setattr(p,otherParam.variable,tableRow[otherParam.variable])
    
    #now we can do the ones that depend on other variables
    for i in paramIndices: 
        param = paramsDF.loc[i]
        variable = param.variable
        value = getattr(p,variable)
        if param.kind == 1 and not hasNumericBounds[i]:
            f = minMaxParam[i] #list of minMax functions for each variable
            minV, maxV = f(p) 
            newValue = bounds(value,minV,maxV)
            setattr(p,variable,newValue)
    return p


# ## create the scaling vector

# In[35]:


pVmax = asParameters([1e30 for i in range(len(paramsDF))])
maxVals = pVmax
maxVals = constrain(pVmax)
scalingVector = asVector(maxVals) #this acts as a scaling vector to map SI unit values to ~unit cube
# paramsDF.iloc[11]


# ## create realistic starting values

# In[36]:


def startParams(returnParamObject=True):
    nParams = len(paramsDF)
    pV = np.random.uniform(0,1,nParams) * scalingVector
    p = constrain(asParameters(pV))
#     print(asVector(p))
    return p if returnParamObject else asVector(p)
# objective(p,weightsCustom)


# In[37]:


# p = startParams()
# constrained = asVector(p)
# # constrained/ scalingVector
# for i, row in paramsDF.iterrows():
#     print(row[1] + " : \t"+str(constrained[i]))


# ### run random possible start values through objectives to get distribution of outputs

# In[40]:


# subscores = []
# for i in range(2000):
#     p = startParams()
#     _,ss = objectiveDetailedNonNormalized(p,weightsNull)
#     subscores.append(ss)
# s = np.array(subscores)


# In[41]:


# x = s[:,7]
# s[:,7 ] = [3000 if isNaN(x[i]) else x[i] for i in range(len(x)) ]


# ### capture the mean and standard deviations of subscores
# so that we can normalize them assuming Normal dist.
# Now, the objective function will fairly weight sub-objectives using custom weights

# In[42]:


# #FIRST TIME, need to run this if we don't have the file with saved values
# subscoreMean = []
# subscoreSd = []
# for i in range(len(subscores[0])):
#     subscoreMean.append(np.mean(s[:,i]))
#     subscoreSd.append(np.std(s[:,i]))

# subscoreStatsDF = pd.DataFrame(columns=["subscoreMean","subscoreSD"],data=np.transpose([subscoreMean,subscoreSd]))
# subscoreStatsDF.to_csv("./SAE/subscoreStatsDF.csv")

# #     plt.hist((np.array(s[:,i]) - subscoreMean[i])/subscoreSd[i])
# #     plt.show()


# In[43]:


subscoreStatsDF = pd.read_csv("./SAE/subscoreStatsDF.csv")
subscoreMean = list(subscoreStatsDF.subscoreMean)
subscoreSd = list(subscoreStatsDF.subscoreSD)


# In[44]:


objective(startParams(),weightsNull)


# # Create Virtual Population with represntative KAI scores 
# based on KAI score and subscore dataset provided by Dr. J

# In[45]:


kaiDF_DATASET = pd.read_csv("./KAI/KAI_DATA_2018_07_09.csv")
kaiDF_DATASET.columns = ["KAI","SO","E","RG"]
def makeKAI(n=1,asDF=True):
    pop = np.random.multivariate_normal(kaiDF_DATASET.mean(),kaiDF_DATASET.cov(),n)
    if asDF:
        popDF = pd.DataFrame(pop) 
        popDF.columns = kaiDF_DATASET.columns
        return popDF if n>1 else popDF.loc[0]
    else:
        return pop if n>1 else pop[0]
    
# def makeSubscores(kai,n=1,asDF=True):
#     pop = np.random.multivariate_normal(kaiDF_DATASET.mean(),kaiDF_DATASET.cov(),n)
kaiPopulation = makeKAI(100000)
kaiPopulation=kaiPopulation.round()


# In[46]:


def findAiScore(kai):
    kai = int(kai)
    a = kaiPopulation.loc[kaiPopulation['KAI'] == kai]
    ind = np.random.choice(a.index)
    me = kaiPopulation.loc[ind]
    return KAIScore(me) #this is a KAIScore object


# In[47]:


def normalizedAI(ai):
    return (ai - kaiDF_DATASET.mean().KAI)/kaiDF_DATASET.std().KAI
def normalizedRG(rg):
    return (rg - kaiDF_DATASET.mean().RG)/kaiDF_DATASET.std().RG
def normalizedE(E):
    return (E - kaiDF_DATASET.mean().E)/kaiDF_DATASET.std().E
def normalizedSO(SO):
    return (SO - kaiDF_DATASET.mean().SO)/kaiDF_DATASET.std().SO


# In[48]:


kaiDF_DATASET.mean().E
kaiDF_DATASET.std().E


# In[49]:


def dotNorm(a,b): #return normalized dot product (how parallel 2 vectors are, -1 to 1)
    if norm(a) <= 0 or norm(b)<= 0:
#         print("uh oh, vector was length zero")
        return 0
    a = np.array(a)
    b = np.array(b)
    dotAB = np.sum(a*b)
    normDotAB = dotAB / (norm(a)*norm(b))
    return normDotAB


# In[50]:


def plotCategoricalMeans(x,y):
    categories = np.unique(x)
    means = []
    sds = []
    for c in categories:
        yc = [y[i] for i in range(len(y)) if x[i] == c]
        means.append(np.mean(yc))
        sds.append(np.std(yc))
    plt.errorbar(categories,means,yerr=sds,marker='o',ls='none')
    
    return means


# In[51]:


#speed distributions:
df=1.9
def travelDistance(speed): #how far do we go? chi distribution, but at least go 0.1 * speed
    r = np.max([chi.rvs(df),0.1]) 
    return r * speed


# In[52]:


def memoryWeightsPrimacy(n):
    if n==1:
        return np.array([1])
    weights = np.arange(n-1,-1,-1)**3*0.4 + np.arange(0,n,1)**3
    weights = weights / np.sum(weights)
    return weights


# In[53]:


def aiColor(ai): #red for innovators, blue for adaptors
    ai01 = bounds((ai - 40)/ 120,0,1)
    red = ai01
    blue = 1 - ai01
    return (red,0,blue)


# ## Agent and Team Classes

# In[54]:


class Agent:
    def __init__(self, id=-1):
        self.id = id
        self.score = np.inf
        self.params = startParams()
        self.r = asVector(self.params)
        self.rNorm = self.r / scalingVector
        self.nmoves = 0
        self.kai = KAIScore()
        self.speed = bounds(AVG_SPEED + normalizedAI(self.kai.KAI) * SD_SPEED, MIN_SPEED ,np.inf)
        self.temp = bounds(AVG_TEMP + normalizedE(self.kai.E) * SD_TEMP, 0 ,np.inf)
        self.iq = 1 #makeIqScore()
        self.memory = [Solution(self.r,self.score,self.id,type(self))]
        self.team = -1
        self.decay = calculateAgentDecay(self,100)
        
    def move(self,soBias=False,groupConformityBias=False,teamPosition=None):
        if np.random.uniform()>self.iq: #I'm just thinking this turn
            return False 
        
        #pick a new direction
        d = np.random.uniform(-1,1,nDims)
        d = d * self.myDims #project onto the dimensions I can move
        #distance moved should be poisson distribution, rn its just my speed
        distance = travelDistance(self.speed) * nDims
        d = d / np.linalg.norm(d) * distance
#         print('considering moving '+str(d) + ' from '+str(self.r))
        candidateSolution = asParameters((self.rNorm + d)*scalingVector)
        candidateSolution = constrain(candidateSolution,self.myDims)
        
        acceptsNewSolution = self.evaluate(candidateSolution,soBias,groupConformityBias,teamPosition=teamPosition)
        if acceptsNewSolution: 
            self.moveTo(asVector(candidateSolution))
            return True
        self.score = self.f()
        return False

    def moveTo(self, r):
        self.r = r
        self.rNorm = self.r / scalingVector
        self.params = asParameters(self.r)
        self.score = self.f()
        self.memory.append(Solution(self.r,self.score,self.id,type(self)))
        self.nmoves += 1
        
    def startAt(self,position):
        self.r = position
        self.rNorm = self.r / scalingVector
        self.params = asParameters(self.r)
        self.memory = [Solution(r=self.r,score=self.f(),owner_id=self.id,agent_class=type(self))]

    def wantsToTalk(self,pComm):
        if(np.random.uniform() < pComm):
            return True
        return False
    
    def getBestScore(self):
        bestScore = self.score
        for s in self.memory:
            if s.score < bestScore:
                bestScore = s.score
        return bestScore
    
    def getBestSolution(self):
        bestSolution = self.memory[0]
        for m in self.memory:
            if m.score < bestSolution.score:
                bestSolution = m
        return bestSolution
    
    def soBias(self,currentPosition,candidatePosition): #influences preference for new solutions, f(A-I)
        #positions should be given as NORMALIZED positions on unit cube! 
        soNorm = normalizedSO(self.kai.SO) #normalized score for Sufficiency of Originality
        memSize = len(self.memory) 
        if memSize < 2: return 0 #we don't have enough places be sticking around them
        
        candidateDirection = candidatePosition - currentPosition #in unit cube space
        
        memDirection = 0 # what is the direction of past solns from current soln?
        weights = memoryWeightsPrimacy(memSize) #weights based on temporal order, Recency and Primacy Bias
        for i in range(memSize-1): #don't include current soln
            past_soln = self.memory[i]
            pairwiseDiff = past_soln.rNorm - currentPosition
            memDirection += pairwiseDiff * weights[i]
        #now we see if the new solution is in the direction of the memories or away from the memories
        paradigmRelatedness = dotNorm(memDirection, candidateDirection)
        raw_PR_score = soNorm * (paradigmRelatedness + 0) #shifting the x intercept #biasOfSO(PR,soNorm)
        sufficiency_of_originality = raw_PR_score*SO_STRENGTH #the agent should have a memory of their path & interactions
            
        return sufficiency_of_originality
    
    def groupConformityBias(self,teamPosition,currentPosition,candidatePosition): #influences preference for new solutions, f(A-I)
        #Positions given should be normalized onto Unit Cube! 
        #rgScore = self.rg , normalized... 
        rgNorm = normalizedRG(self.kai.RG) #normalized score for Rule/Group Conformity
        candidateDirection = candidatePosition - currentPosition 
        
        #If we want to bias the weights of team members...
#         teamSize = team.nAgents
#         if teamSize < 2: return 0 #we don't have any teammates
#         teamDirection = 0 # what is the direction of teammates' solns from current soln?
#         teamWeights = [ 1/teamSize for i in range(teamSize)] # all teammates are equally important... for now...
#         for i in range(teamSize-1): #don't include self
#             teammateSoln = team.agents[i].r
#             pairwiseDiff = teammateSoln.r - currentPosition
#             teamDirection += pairwiseDiff * teamWeights[i]
        
        teamDirection = teamPosition - currentPosition
        
        #now we see if the new solution is in the direction of the team or away from the team
        groupConformity = dotNorm(teamDirection, candidateDirection)
        nominalGC = 0 #can change intercept with -0 (this is the dot product of direction, is perpendicular the null case?)
        groupConformityBias = (groupConformity-nominalGC)*rgNorm*RG_STRENGTH 
        if VERBOSE:
            print("current position: "+str(currentPosition))
            print("candidate position: "+str(candidatePosition))
            print("dot product: "+str(groupConformity))
            print("bias: "+str(groupConformityBias))
        return groupConformityBias
    
    def evaluate(self,candidateSolution,soBias=False,groupConformityBias=False,teamPosition=None): #implements simulated annealing greediness
        candidateSolutionNorm = asVector(candidateSolution) / scalingVector
        candidateScore = self.fr(candidateSolution)
        if soBias:
            candidateScore += self.soBias(self.rNorm,candidateSolutionNorm)
        if groupConformityBias:
            gcB = self.groupConformityBias(teamPosition,self.rNorm,candidateSolutionNorm)
            candidateScore += gcB
        #if better solution, accept
        if candidateScore < self.score:
            return True
        #accept worse solution with some probability, according to exp((old-new )/temp)
        elif pickWorseScore(self.score,candidateScore,self.temp):
            self.score = candidateScore #(its worse, but we go there anyways)
            return True              
        return False
    
    
#Solutions are objects
class Solution():
    def __init__(self, r,  score, owner_id=None, agent_class=None):
        self.r = cp(r)
        self.rNorm = self.r / scalingVector
        self.score = cp(score)
        self.owner_id = cp(owner_id)
        self.agent_class = cp(agent_class)
        
#KAI scores are objects 
class KAIScore():
    def __init__(self,subscores=None):
        if subscores is None:
            subscores = makeKAI(1,True)
        self.KAI = subscores.KAI
        self.SO = subscores.SO
        self.E = subscores.E
        self.RG = subscores.RG
    
#subclasses (Types) of Agents
class carDesigner(Agent):
    def __init__(self, id=-1):
        Agent.__init__(self,id)
        self.myDims = [1 for t in paramsDF.team ] #owns all dimensions by default
#         self.params = startParams() ##these already set in class Agent()
#         self.r = asVector(params)
#         self.rNorm = self.r / scalingVector
         
    def f(self):
        return objective(self.params,weightsCustom)
    def fr(self,params):
        return objective(params,weightsCustom)
    
class cabinPerson(Agent):
    def __init__(self, id=-1):
        Agent.__init__(self,id)
        self.myDims = [( 1 if t == "c" else 0 ) for t in paramsDF.team ]
#         self.r = np.random.uniform(-1*startRange,startRange,nDims)
        self.params = startParams() # asParameters(self.r)
        
    def f(self):
        return objective(self.params,weightsCustom)
    def fr(self,params):
        return objective(params,weightsCustom)
    
# class Steinway(Agent): #tuneable roughness
#     def __init__(self, id=-1):
#         Agent.__init__(self,id)
#         self.myDims = np.ones(nDims)
#         self.r = np.random.uniform(-1*startRange,startRange,nDims)
        
#     def f(self):
#         return testy(self.r,ROUGHNESS)
#     def fr(self,r):
#         return testy(r,ROUGHNESS)

# class LewisMultiDim(Agent): #objective is beauty
#     def __init__(self,id=-1):
#         Agent.__init__(self,id)
#         self.myDims = np.ones(nDims) #logical vector, LewD controls/varies x0 AND x1
        
#     def f(self):
#         return ellipsoid(self.r)
#     def fr(self,r):
#         return ellipsoid(r)


def tryToShare(a1,a2):
    deltaAi = abs(a1.kai.KAI - a2.kai.KAI) #hard to communicate above 20, easy below 10
    #increasing commBonus makes sharing easier 
    c = np.random.normal(20+commBonus,commSd)
    successful =  deltaAi - c < 0 #or np.random.uniform()<0.3 #always give a 30% chance of success
    if successful: #in share(), agents might adopt a better solution depending on their temperature
        share(a1,a2) if complexSharing else shareBasic(a1,a2)
        return True
    return False

# def softShare(a1,a2): #share position with some error
#     deltaAi = abs(a1.ai - a2.ai) #hard to communicate above 20, easy below 10
#     #the higher the deltaAi, the more noise in sharing position
#     e = np.random.normal(10,10)
#         return True
#     return False
    
# def shareBasic(a1,a2): #always share
#     for i in range(len(a1.myDims)):
#         if(a1.myDims[i]>0):
#             #a1 controls this dimension, but a2 only follows if it helps them
#             candidateSoln = a2.params #other dimensions won't change
#             candidateSoln[i] = a1.r[i] #tells a1 where to go in This [i] dimension only
#             if(a2.fr(candidateSoln)<a2.score): #self-bias goes here
#                 a2.r = candidateSoln
#                 a2.params = asParameters(a2.r)
#                 a2.score = a2.f()
# #                 print('shared')
         
#         elif(a2.myDims[i]>0):
#             #a1.r[i]=a2.r[i] #for naive follow
#             candidateSoln = a1.r 
#             candidateSoln[i] = a2.r[i] 
#             if(a1.fr(candidateSoln)<a1.score): 
#                 a1.r = candidateSoln
#                 a1.params = asParameters(a1.r)
#                 a1.score = a1.f()
# #                 print('shared')
#     return True
        
def considerSharedSoln(me,sharer): #,dim): #will only move (jump) in the dimensions that sharer controls
#         candidateSoln = me.r #other dimensions won't change
#         candidateSoln[dim] = sharer.r[dim] #tells a1 where to go in This [i] dimension only
        candidateSolution = sharer.params
        candidateScore = me.fr(candidateSolution)
        myScore = me.score - selfBias #improve my score by selfBias
        #Quality Bias Reduction? would go here
        if(candidateScore<myScore):
            if not pickWorseScore(candidateScore,myScore,me.temp): #sometimes choose better, not always
                me.moveTo(asVector(candidateSolution))  #(but never take a worse score from a teammate)
    
def share(a1,a2): #agent chooses whether to accept new solution or not, holistic NOTTTTTT dimension by dimension
    copyOfA1 = cp(a1)
    considerSharedSoln(a1,a2)
    considerSharedSoln(a2,copyOfA1) #so they could theoretically swap positions...
#     for i in range(len(a1.myDims)):
#         if(a1.myDims[i]>0):
#             considerSharedSoln(a2,a1,i)
         
#         elif(a2.myDims[i]>0):
#             considerSharedSoln(a1,a2,i)
      
    return True


# In[55]:


class Team(): #a group of agents working on the same dimension and objective function
    def __init__(self, nAgents, agentConstructor, dimensions = np.ones(nDims), specializations = None, temp=None,speed=None,aiScore=None,aiRange=None,startPositions=None):
        self.agents = []
        self.dimensions = dimensions
        if (aiScore is not None) and (aiRange is not None):
            minScore = np.max([40, aiScore-aiRange/2.0])
            maxScore = np.min([150,aiScore+aiRange/2.0])
            aiScores = np.linspace(minScore,maxScore,nAgents)
        for i in range(nAgents):
            a = agentConstructor(id = i)
            if startPositions is not None:
                a.startAt(startPositions[i])
            if (aiScore is not None) and (aiRange is not None):
                aiScore = aiScores[i]
                a.kai = findAiScore(aiScore)
                a.speed = bounds(AVG_SPEED + normalizedAI(a.kai.KAI) * SD_SPEED, MIN_SPEED ,np.inf)
                a.temp = bounds(AVG_TEMP + normalizedE(a.kai.E) * SD_TEMP, 0 ,np.inf)
            if speed is not None:
                a.speed = speed
            if temp is not None:
                a.temp = temp
            a.myDims = dimensions #default: all dimensions owned by every agent
            self.agents.append(a)
        self.nAgents = nAgents
        aiScores = [a.kai.KAI for a in self.agents]
        self.dAI = np.max(aiScores)- np.min(aiScores)
        self.nMeetings = 0
        self.nTeamMeetings = 0
        
        self.scoreHistory = []
        
        #if there are subteams owning certain dimensions, each subteams dimensions are listed in a matrix
        self.specializations = specializations
        
        #we should give them same position in the dimensions they don't control
        a0 = self.agents[0]
        for i in range(len(a0.myDims)):
            if not a0.myDims[i]:  #this isn't our dimension to control
                for a in self.agents: 
                    a.r[i] = a0.r[i] 

    def getSharedPosition(self): #this is in the normalized space
        positions = np.array([a.rNorm for a in self.agents])
        return [np.mean(positions[:,i]) for i in range(len(positions[0]))]
    
    def getBestScore(self):
        return np.min([a.getBestScore() for a in self.agents])
    
    def getBestTeamSolution(self,team=-1): #returns a Solution object 
        bestIndividualSolns = [a.getBestSolution() for a in self.agents if a.team == team ]
        bestScoreLocation = np.argmin([s.score for s in bestIndividualSolns])
        return bestIndividualSolns[bestScoreLocation]
    
    def haveMeetings(self,talkers):
        nMeetings = 0
        for i in np.arange(0,len(talkers)-1,2):
            #if you don't have a partner, you don't talk to anyone?
            #this needs to be adjusted
            a1 = talkers[i]
            a2 = talkers[i+1]
            didShare = tryToShare(a1,a2)
            if didShare: 
#                 print(str(a1.id) + ' and '+str(a2.id)+' shared!')
                nMeetings +=1
        self.nMeetings += nMeetings
        return nMeetings
    
    def haveTeamMeeting(self):
        #they all go to the best position of the group
        bestSolution = self.agents[0].getBestSolution()
        for a in self.agents:
            agentBest = a.getBestSolution()
            if agentBest.score < bestSolution.score:
                bestSolution = agentBest
        #now move all agents to this position
        for a in self.agents:
            a.moveTo(bestSolution.r)
            
        return bestSolution
    
    def haveInterTeamMeeting(self):
        consensusPosition = np.zeros(nDims)
        #get the best solution from each specialized subteam, and extract their specialized dimensions 
        for team in range(len(self.specializations)):
            bestTeamSoln = self.getBestTeamSolution(team)
            specializedInput = bestTeamSoln.r * self.specializations[team]
            consensusPosition += specializedInput
        consensusPositionP = constrain(asParameters(consensusPosition))
        consensusPosition = asVector(consensusPositionP)
        consensusScore = self.agents[0].fr(consensusPositionP)
        #now move all agents to this position
        for a in self.agents:
            a.moveTo(consensusPosition)
        
        self.nTeamMeetings += 1
        return [consensusScore, consensusPosition]
    
    
    def step(self,pComm,showViz=False,soBias=False,groupConformityBias=False):
        #what happens during a turn for the team? 
        #each agents can problem solve or interact (for now, inside team)
        talkers = []
        for a in self.agents:
            if a.wantsToTalk(pComm):
                talkers.append(a)
            else:
                teamPosition = self.getSharedPosition() if groupConformityBias else None #position is on unit cube
                didMove = a.move(soBias=soBias,groupConformityBias = groupConformityBias, teamPosition=teamPosition)
#         print(len(talkers))
#         print("number of talkers: "+str(len(talkers)))
        nMeetings = self.haveMeetings(talkers)
#         print("number of successful meetings: "+str(nMeetings))
        
#         if showViz:
#             self.plotPositions()
        
        self.updateTempSpeed()
            
        return nMeetings
    
    def updateTempSpeed(self):
        for a in self.agents:
            a.temp *= a.decay
            a.speed *= a.decay
            
        
    def plotPositions(self):
        xs = [a.r[0] for a in self.agents]
        ys = [a.r[1] for a in self.agents]
        cs = [aiColor(a.kai.KAI) for a in self.agents]
        plt.scatter(xs,ys, c=cs)
#         teamPosition = self.getSharedPosition()
#         plt.scatter(teamPosition[0],teamPosition[1],c='orange')
        
                


# # Individual Exploration

# In[56]:


def work(AgentConstructor,steps=100,ai=None,temp=None, speed=None, showViz=False, soBias=False, groupConformityBias = False, color = 'red',startPosition = None,teamPosition=None):    
    a = AgentConstructor()
    if ai is not None:
        a.kai = findAiScore(ai)
        self.speed = bounds(AVG_SPEED + normalizedAI(self.kai.KAI) * SD_SPEED, MIN_SPEED ,np.inf)
        self.temp = bounds(AVG_TEMP + normalizedE(self.kai.E) * SD_TEMP, 0 ,np.inf)
    if startPosition is not None:
        a.startAt(startPosition)
    if temp is not None:
        a.temp = temp
    if speed is not None:
        a.speed = speed
    a.decay = calculateAgentDecay(a, steps)

    scores = []
    shareSuccess = []          

    for i in range(steps):
        didMove = a.move(soBias=soBias,groupConformityBias = groupConformityBias,teamPosition = teamPosition)
        if didMove:
            scores.append(copy.deepcopy(a.score))
            if(showViz and a.nmoves>0):
#                     plt.scatter(a.rNorm[0],a.rNorm[1],c=color)
                plt.scatter(a.rNorm[0],a.score,c=color)
        a.speed *= a.decay
        a.temp *= a.decay

    return a


# # Team Work

# In[57]:


# meetingTimes = 20
def teamWork(teamSize,agentConstructor, pComm, steps=100, soBias=False,groupConformityBias=False, speed=None, temp=None, showViz=False,aiScore=None,aiRange=None,startPositions=None):
    meetingTotals = []
    squad = Team(teamSize,agentConstructor,temp=temp,speed=speed,aiScore=aiScore,aiRange=aiRange,startPositions=startPositions)
    for a in squad.agents:
        a.decay = calculateAgentDecay(a,steps)
    
    meetingTotal = 0
    i = 0
    while i < steps:
        meetingTotal += squad.step(pComm,showViz,soBias,groupConformityBias) 
        if showViz: 
            rGroup = squad.getSharedPosition()
            plt.scatter(rGroup[0],rGroup[1],marker='o',s=100,c='black')
        
        if (i+1)%meetingTimes == 0: 
            squad.haveTeamMeeting()
            squad.nTeamMeetings +=1
            i += TEAM_MEETING_COST
            if(showViz): 
                plt.show()
        i += 1
    if showViz: plt.show()
    meetingTotals.append(meetingTotal)
    
    return squad

# meetingTimes = 20
def teamWorkSpecialized(teamSize,agentConstructor,teamSpecializations,agentTeams, pComm, steps=100, soBias=False,groupConformityBias=False, speed=None, temp=None, showViz=False,aiScore=None,aiRange=None,startPositions=None):
    meetingTotals = []
    squad = Team(teamSize,agentConstructor,temp=temp,speed=speed,aiScore=aiScore,aiRange=aiRange,startPositions=startPositions,specializations = teamSpecializations)
    for i in range(len(squad.agents)):
        a = squad.agents[i]
        aTeam = agentTeams[i]
        a.team = aTeam
        a.myDims = teamSpecializations[aTeam]
        a.decay = calculateAgentDecay(a,steps)
    
    meetingTotal = 0
    i = 0 #not for loop bc we need to increment custom ammounts inside loop
    while i < steps:
#                 pCi = pComm #*(i/steps) #we can make them wait until later to communicate
        meetingTotal += squad.step(pComm,showViz,soBias,groupConformityBias) 
        score = squad.getBestScore()
        squad.scoreHistory.append(score)
        if showViz: 
#             rGroup = squad.getSharedPosition()
            plt.scatter(i,score,marker='o',s=100,c='black')
        
        if (i+1)%meetingTimes == 0: 
            squad.haveInterTeamMeeting()
            squad.nTeamMeetings +=1
            i += TEAM_MEETING_COST
#             if showViz: 
#                 plt.show()
        i += 1
    if showViz: plt.show()
    meetingTotals.append(meetingTotal)
    
    return squad


# In[58]:


#define the team specializations and assign agents to teams
def specializedTeams(nAgents,nDims,nTeams):
    teamDimensions = np.array([[1 if t%nTeams == dim%nTeams else 0 for dim in range(nDims)] for t in range(nTeams)])
    agentTeams = np.array([a%nTeams for a in range(nAgents)])
    return teamDimensions, agentTeams

teams = ['brk', 'c', 'e', 'ft', 'fw', 'ia','fsp','rsp', 'rt', 'rw', 'sw']
teamsDict = { i:teams[i] for i in range(10)}
def saeTeams(nAgents):
    paramTeams = paramsDF.team
    nTeams = len(teams)
    teamDimensions = [[ 1 if paramTeam == thisTeam else 0 for paramTeam in paramTeams] for thisTeam in teams]
    for i in range(nAgents):
        agentTeams = np.array([a%nTeams for a in range(nAgents)])
    return teamDimensions, agentTeams


# In[59]:


def showScoreHistory(agent):
    mem = agent.memory
    for i in range(len(mem)):
        m = mem[i]
        plt.scatter(i,m.score,c='b')
    plt.show()
# a0 = team.agents[9]
# showScoreHistory(a0)
# a0.myDims


# # visualize the solution?

# ## Run SAE team

# In[61]:


t = timer.time()
nDims = len(paramsDF)
selfBias = 0
complexSharing = True
steps = 100 #500
nAgents = 22#22

RG_STRENGTH = 2 #was 10
SO_STRENGTH = 2
AVG_SPEED = 1E-2 #.1 #.3
SD_SPEED = 0.7E-3#.06
MIN_SPEED = 1E-4
AVG_TEMP = 1
SD_TEMP = 0.8
constructor = carDesigner
ROUGHNESS = 0.05
pComm = 0.2

VERBOSE = False
showViz = False

reps = 2 #$40 #5
scores = []
aiTeamMeans = []
aiTeamRange = []
nMeetings = []
teamMeetings = []

aiScoresMeans = np.linspace(80,120,3)
aiRanges = np.linspace(0,30,2)

meetingTimes = 10
TEAM_MEETING_COST = 1

teamDims, agentTeams = saeTeams(nAgents)

scoreMatrix = np.zeros([len(aiRanges),len(aiScoresMeans)])

for k in range(reps):
#     if k%10 == 0:
#         startPositions = np.random.uniform(-1*startRange,startRange,[teamSize,nDims])
    for i in range(len(aiRanges)):
        aiRange = aiRanges[i]
        for j in range(len(aiScoresMeans)):
            aiScore = aiScoresMeans[j]
            team = teamWorkSpecialized(nAgents,constructor,teamDims,agentTeams,showViz=showViz,speed=None,pComm=pComm,steps=steps,groupConformityBias=True,soBias=True,aiScore=aiScore,aiRange=aiRange)
            score = team.getBestScore()
            scores.append(score)
            aiScores = [a.kai.KAI for a in team.agents]
            aiTeamMeans.append(np.mean(aiScores))
            aiTeamRange.append(np.max(aiScores) - np.min(aiScores))
            nMeetings.append(team.nMeetings)
            teamMeetings.append(team.nTeamMeetings)
            scoreMatrix[i,j] += score
            plt.clf()
            plt.plot(range(len(team.scoreHistory)),team.scoreHistory)
            plt.ylim([-3,0])
            plt.xlim([0,100])
            plt.savefig("./figs/teamScores_mu:"+str(aiScore)+"_sd:"+str(aiRange)+"_"+str(timer.time())+".pdf")
#             print(team.nMeetings)
scoreMatrix = np.array(scoreMatrix) / reps #the average score
print("time to complete: "+str(timer.time() - t))
print("ai ranges: "+str(aiRanges))
print("ai means: "+str(aiScoresMeans))
print("reps:" +str(reps))
np.savetxt("./results/scoreMatrix_+"+str(timer.time())+".csv", scoreMatrix, delimiter=",")


# In[62]:


scoreMatrix


# # multithreading

# In[119]:


t0 = timer.time()
nDims = len(paramsDF)
selfBias = 0
complexSharing = True
steps = 100 #500
nAgents = 22#22

RG_STRENGTH = 2 #was 10
SO_STRENGTH = 2
AVG_SPEED = 1E-2 #.1 #.3
SD_SPEED = 0.7E-3#.06
MIN_SPEED = 1E-4
AVG_TEMP = 1
SD_TEMP = 0.8
constructor = carDesigner
ROUGHNESS = 0.05
pComm = 0.2

VERBOSE = False
showViz = False

reps = 1 #$40 #5
scores = []
aiTeamMeans = []
aiTeamRange = []
nMeetings = []
teamMeetings = []

aiScoresMeans = np.linspace(80,120,3)
aiRanges = np.linspace(0,30,2)

meetingTimes = 10
TEAM_MEETING_COST = 1

teamDims, agentTeams = saeTeams(nAgents)

scoreMatrix = np.zeros([len(aiRanges),len(aiScoresMeans)])

exitFlag = 0

allTeamObjects = []

class simulationThread (threading.Thread):
    def __init__(self, threadID):
        threading.Thread.__init__(self)
        self.threadID = threadID
    def run(self):
        print("Starting " + str(self.threadID))
        t=timer.time()
        team = runTeamThread(self.threadID)
        allTeamObjects.append(team)
        pickle.dump(team,teamFile)
#         print("team score:" +str(team.getBestScore()))
#         print("Exiting " + str(self.threadID) +" after runtime: "+str(int(timer.time()-t)))
#         print("total runtime: "+str(int(timer.time()-t0)))

def runTeamThread(threadName):
    if exitFlag:
        threadName.exit()
    team = teamWorkSpecialized(nAgents,constructor,teamDims,agentTeams,showViz=showViz,speed=None,pComm=pComm,steps=steps,groupConformityBias=True,soBias=True,aiScore=aiScore,aiRange=aiRange)
#         print("%s: %s" ,(threadName, time.ctime(time.time())))
    return team


teamFileUrl = './savedTeams_'+str(timer.time())+'.obj'
teamFile = open(teamFileUrl, 'wb')
for i in range(1):
    thread = simulationThread(i)
    thread.start()
    thread.join()
print("time to complete: "+str(round(timer.time()-t0,1)) + " seconds")

teamFile.close()



# In[113]:


filehandler = open(teamFileUrl, 'rb')
allTeamObjectsFromFile = []
success = True
while success:
    try:
        allTeamObjectsFromFile.append(pickle.load(filehandler))
    except:
        success = False
print(len(allTeamObjectsFromFile))


# In[117]:


T0 = allTeamObjectsFromFile[0]
T0.agents


# In[76]:


# x = teamMeetings
# y = scores
# # z = np.polyfit(x, y, 4)
# # p = np.poly1d(z)
# # xp = np.linspace(min(teamMeetings), max(teamMeetings), 50)
# # _ = plt.plot(x, y, '.', xp, p(xp), '-')
# plotCategoricalMeans(x,y)
# plt.title("Score vs number of team meetings")
# plt.show()


# In[156]:


# !jupyter nbconvert --to script Chicago.ipynb

