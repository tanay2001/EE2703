import numpy as np 
import matplotlib.pyplot as plt 
from scipy.integrate import quad 
from scipy.linalg import lstsq
def exponential(x):
    return np.exp(x)

def coscos(x):
    return np.cos(np.cos(x))


class wrapperplots():
    def __init__(self, xrange = None ,label = None, yrange = None, path  = None,icon ='r-', plottype = 'plot', clear  = True, save = True):
        super(wrapperplots, self).__init__()
        self.Xrange = xrange
        self.Yrange = yrange
        self.label = label
        self.img_path = path
        self.plottype = plottype ## TODO add more types currently only plot, semilogy, semilogx,
        self.saver = save
        self.cleaner = clear
        self.icon = icon
        self.seteverything()

    def plot(self):
        #create plot object
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.ax.grid()

    def seteverything(self):
        self.plot()
        if self.Xrange!=None:
            self.ax.set_xlim(self.Xrange)
        if self.Yrange!=None:
            self.ax.set_ylim(self.Yrange)
        if not self.label:
            self.ax.legend()

    def __call__(self, x, y):
        assert len(x) ==len(y), "fucntions not same length"
        if self.plottype =='plot':
            self.ax.plot(x,y, self.icon)
        elif self.plottype =='semilogy':
            self.ax.semilogy(x, y, self.icon)
        elif self.plottype =='semilogx':
            self.ax.semilogx(x, y, self.icon)
        elif self.plottype =='loglog':
            self.ax.loglog(x, y, self.icon)
        else:
            print("not supported")
        if self.saver:
            self.ax.figure.savefig(self.img_path+'.png',bbox_inches='tight')
            print("file saved at {}".format(self.img_path+'.png'))
        if self.cleaner:
            pass
            #self.ax.clf()




#TODO convert this to a wrapper class for multiple types of plots
def plotdata(x,y, xrange = None ,label = None, yrange = None, path  = None, plottype = None, clear  = True):
    assert len(x) ==len(y), "fucntions not same length"
    if plottype =='semilogy': #TODO ad more plot types
        plt.semilogy(x,y,'ro' ,label =label)
    else:
        plt.plot(x,y, label = label)
    plt.grid(True)
    if not xrange:
        plt.xlim(xrange)
    if not yrange:
        plt.ylim(yrange)
    plt.legend()
    plt.savefig(path+'.png',bbox_inches='tight')
    print("file saved at {}".format(path+'.png'))
    if clear:
        plt.clf()

def fourier_coeff(n,func):
    coeff = np.empty(n)
    u = lambda x,k: func(x)*np.cos(k*x)
    v = lambda x,k: func(x)*np.sin(k*x)
    coeff[0]= quad(func,0,2*np.pi)[0]/(2*np.pi)
    for i in range(1,n,2): 
        coeff[i] = quad(u,0,2*np.pi,args=((i+1)/2))[0]/np.pi
    for i in range(2,n,2):
        coeff[i] = quad(v,0,2*np.pi,args=(i/2))[0]/np.pi
    return coeff

def leastSquareCoef(func):
    A = np.empty((400,51))
    x = np.linspace(0,2*np.pi, 401)
    x = x[:-1]
    A[:,0] =1 
    for i in range(1,26):
        A[:,2*i-1] = np.cos(i*x)
        A[:,2*i] = np.sin(i*x)
    b = func(x)

    return A, b


if __name__ == "__main__":
    x = np.linspace(-2*np.pi,4*np.pi,300)
    #true is x 
    #fourier will handle only (0,2pi) and make that periodic so plot this also
    #plotdata(x, exponential(x), xrange=(-2*np.pi, 4*np.pi), path = 'exp_plot',label ='True function' ,plottype='semilogy', clear= False)

    xnew =np.linspace(0,2*np.pi,100)
    ynew = exponential(xnew)
    #note its 3 peroids so 
    y = np.tile(ynew, 3)
    #plotdata(x, y, xrange=(-2*np.pi, 4*np.pi), path = 'exp_plot',label= 'predicted fucntion', plottype='semilogy')


    #tr = wrapperplots(xrange=(-2*np.pi, 4*np.pi),label='cos(cos(x))', path = 'cos_plot2')
    #tr(x, coscos(x))

    #plotdata(x, coscos(x), xrange=(-2*np.pi, 4*np.pi),label='cos(cos(x))', path = 'cos_plot')

    Fcoef_cos = fourier_coeff(51,coscos)
    Fcoef_exp = fourier_coeff(51,exponential)

    #cos_ff = wrapperplots(icon= 'ro',plottype='semilogy', path='coeff_cos_semilog')
    #cos_ff(range(1,52), np.abs(Fcoef_cos))


    ##############################################################################################################

    A, bexp = leastSquareCoef(exponential)
    Lcoef_exp = lstsq(A,bexp)[0]

    A, bcos = leastSquareCoef(coscos)
    Lcoef_cos = lstsq(A,bcos)[0]

    #cos_ff = wrapperplots(icon= 'ro',plottype='semilogy', path='lstq_cos')
    #cos_ff(range(1,52), np.abs(Lcoef_cos))


    #cos_ff = wrapperplots(icon= 'ro',plottype='semilogy', path='lstq_exp')
    #cos_ff(range(1,52), np.abs(Lcoef_exp))

    plt.semilogy(range(1,52), np.abs(Lcoef_exp), 'ro')
    plt.savefig("example.png")


    ###########################################################################################################################












