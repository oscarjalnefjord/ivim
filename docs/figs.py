import os
import json
import numpy as np
import matplotlib.pyplot as plt
import ivim

fig,axes = plt.subplots(2,1,figsize=(12, 5))
deltas = [15e-3,8e-3]
Deltas = [25e-3,10e-3] 
T = 45e-3
TE = 60e-3
titles = ['Monopolar','Bipolar']

# Pulse sequence diagrams
for ax,delta,Delta,title in zip(axes,deltas,Deltas,titles):

    # RF timing
    dur_rf = 5e-3 # ad hoc
    n_rf = 100
    t_rf = np.linspace(0,dur_rf,n_rf) 

    # plot pulse sequence
    if title == 'Monopolar':
        encs = ['NC']
    else:
        encs = ['FC','NC']
    style = {'FC':'-','NC':'--'}
    color = {'FC':'gray','NC':'black'}
    label = {'FC':'Flow-compensated','NC':'Non-flow-compensated'}
    for enc in encs:
        if title == 'Monopolar':
            t = [TE/2-(Delta-delta)/2-delta,TE/2-(Delta-delta)/2-delta,TE/2-(Delta-delta)/2,TE/2-(Delta-delta)/2,
                 TE/2+(Delta-delta)/2,TE/2+(Delta-delta)/2,TE/2+(Delta-delta)/2+delta,TE/2+(Delta-delta)/2+delta]
            g = [0,0.8,0.8,0,0,0.8,0.8,0]
        else:
            t = [TE/2-T/2,TE/2-T/2,TE/2-T/2+delta,TE/2-T/2+delta,TE/2-T/2+Delta,TE/2-T/2+Delta,TE/2-T/2+Delta+delta,TE/2-T/2+Delta+delta,
                TE/2+T/2-Delta-delta,TE/2+T/2-Delta-delta,TE/2+T/2-Delta,TE/2+T/2-Delta,TE/2+T/2-delta,TE/2+T/2-delta,TE/2+T/2,TE/2+T/2]
            sign = (-1)**(enc=='NC')
            g = [0,0.8,0.8,0,0,-0.8,-0.8,0,0,sign*0.8,sign*0.8,0,0,sign*-0.8,sign*-0.8,0]

        ax.plot(t,g,style[enc],color=color[enc],label=label[enc]) # gradient waveforms
    if title == 'Bipolar':
        ax.legend(loc='upper right')
    
    if title == 'Monopolar':
        dur_gr = delta + Delta
    else:
        dur_gr = T
    ax.annotate(text='',xy=(TE/2-dur_gr/2,-0.55),xytext=(TE/2-dur_gr/2+Delta,-0.55),arrowprops={'arrowstyle':'<->','shrinkA':0,'shrinkB':0})
    ax.text(TE/2-dur_gr/2+Delta/2,-0.6,r'$\Delta$',verticalalignment='top',horizontalalignment='center',fontsize=10) # Delta
    ax.annotate(text='',xy=(TE/2-dur_gr/2,-0.25),xytext=(TE/2-dur_gr/2+delta,-0.25),arrowprops={'arrowstyle':'<->','shrinkA':0,'shrinkB':0})
    ax.text(TE/2-dur_gr/2+delta/2,-0.3,r'$\delta$',verticalalignment='top',horizontalalignment='center',fontsize=10) # delta
    if title == 'Bipolar':
        ax.annotate(text='',xy=(TE/2-T/2,-0.9),xytext=(TE/2+T/2,-0.9),arrowprops={'arrowstyle':'<->','shrinkA':0,'shrinkB':0})
        ax.text(TE/2,-0.95,r'$T$',verticalalignment='top',horizontalalignment='center',fontsize=10) # encoding time

    ax.plot([-dur_rf,TE+10e-3],[0,0],'k') # time axis

    ax.plot(t_rf-dur_rf/2,np.sinc(np.linspace(-2,2,n_rf))/2,'k')       #  90 degree pulse
    ax.plot(t_rf-dur_rf/2+TE/2,np.sinc(np.linspace(-3,3,n_rf)),'k')    # 180 degree pulse

    ax.text(TE/2+dur_gr/2+5e-3,0,'EPI readout',horizontalalignment='left',verticalalignment='center',bbox={'facecolor':'white','alpha':1}) # readout

    ax.set_ylim([-1.1,1.1])
    ax.axis('off')
    #ax.set_title(title)
    ax.text(TE/2,1.1,title,ha='center',fontsize=12)

fig.savefig(os.path.join('docs','figs','seqs.png'),bbox_inches='tight')