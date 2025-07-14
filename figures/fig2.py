
import numpy as np
import matplotlib.pyplot as plt
from readData.DataLoader import DataLoader
from matplotlib.colors import BoundaryNorm
import matplotlib as mpl
import pandas as pd
import cd_function
import shared_data
import matplotlib as mpl
def main():
    #plt.rc('axes', labelsize=20)
    s1=18
    s2=24
    s7 = pd.read_csv('ups7rcs_min038.txt', header=None)[0]
    s8 = pd.read_csv('ups8rcs_min038.txt', header=None)[0]
    s9 = pd.read_csv('ups9rcs_min038.txt', header=None)[0]
    s10 = pd.read_csv('ups10rcs_min038.txt', header=None)[0]
    fig, ax = shared_data.set_plot_params(ncols=2)
    rc_ax = ax[1]
    a_ax = ax[0]
    m10_rc = np.empty([4,5])
    m10_rc[0] = np.round(s7, decimals=2)
    m10_rc[1] = np.round(s8, decimals=2)
    m10_rc[2] = np.round(s9, decimals=2)
    m10_rc[3] = np.round(s10, decimals=2)
    s_arr = np.array([0.5, 1.0, 2.5, 5.0])/0.6909
    #change extent so gridlines surround each square
    extent = (0, m10_rc.shape[1], m10_rc.shape[0],0)
    rc_ax.set_yticks([0,1,2,3],labels=np.round(s_arr[::-1], 1))
    rc_ax.set_xticks([0,1,2,3,4],labels=['1','5','10','20','31'])
    norm = mpl.colors.LogNorm(vmin=0.2, vmax=90)
    #cmap = mpl.colors.LinearSegmentedColormap.from_list('my_colormap',['blue','red'],256)
    cmap = mpl.cm.plasma
    img = rc_ax.imshow(m10_rc, interpolation='nearest',cmap=cmap, norm=norm)
    #workaround to get gridlines
    #plt.hlines(y=np.arange(0,3)+.5, xmin=np.full(3,0)-.5, xmax=np.full(3,5)-.5, color='w')
    #plt.vlines(x=np.arange(0,5)+.5, ymin=np.full(5,0)-.5, ymax=np.full(5,4)-.5, color='w')
    for (j,i), label in np.ndenumerate(m10_rc):
        rc_ax.text(i,j, fr'\boldmath${label}$', ha='center', va='center',color='w', size=18)
    #fig.colorbar(img, ax=ax, norm=norm).set_label(r'$\alpha$')
    cb = plt.colorbar(img, ax=rc_ax, norm=norm)
    cb.set_label(r'$r_{\rm{c}}~\rm{(kpc)}$')
    cb.minorticks_off()
    ticks = [1.0, 5.0, 10.0, 30.0]
    cb.set_ticks(ticks=ticks,labels=ticks)
    #cb.set_ticks(ticks=[0.5, 1.0, 2.5, 5.0], labels=['0.5', '1.0','2.5', '5.0'])
    #h, l = cb.ax.get_legend_handles_labels()
    #cb.ax.axhline(y=-1.14, color='white',label='smooth growth')
    #cb.ax.legend(bbox_to_anchor=[6,-0.05],facecolor='lightgray', framealpha=.5)
    #set coordinates of the labels
    size_x = m10_rc.shape[0]
    size_y = m10_rc.shape[1]
    x_end = m10_rc.shape[1]
    x_start = 0
    y_end = 0
    y_start = m10_rc.shape[1]
    jump_x = (x_end - x_start)/(2.0*size_x)
    jump_y = (y_end - y_start)/(2.0*size_y)
    x_positions = np.linspace(start=x_start, stop=x_end, num=size_x, endpoint=False)
    y_positions = np.linspace(start=y_start, stop=y_end, num=size_y, endpoint=False)
   # for y_index, y in enumerate(y_positions):
    #    for x_index, x in enumerate(x_positions):
     #       label = m10_alphas[x_index, y_index]
      #      text_x = x + jump_x
       #     text_y = y + jump_y
        #    ax.text(text_x, text_y, label, color='white', ha='center', va='center'))
    #ax.set_xticks([0,1,2,3,4],labels=['1','5','10','20','31'])
    #ax.set_yticks([0,1,2],labels=['0.5','1','2.5'][::-1])
    rc_ax.set_xlabel('Number of Bursts', size=s2)
    rc_ax.set_ylabel(r'Mass Expelled Per Burst ($10^{7}M_{\odot}$)', size=s1)
    #ax.grid(True, color='black')


    #with alpha
    s7 = pd.read_csv('ups7slope.txt', header=None)[0]
    s8 = pd.read_csv('ups8slope.txt', header=None)[0]
    s9 = pd.read_csv('ups9slope.txt', header=None)[0]
    s10 = pd.read_csv('ups10slope.txt', header=None)[0]
    m10_alphas = np.empty([4,5])
    m10_alphas[0] = np.round(s7, decimals=2)
    m10_alphas[1] = np.round(s8, decimals=2)
    m10_alphas[2] = np.round(s9, decimals=2)
    m10_alphas[3] = np.round(s10, decimals=2)
    a_ax.set_yticks([0,1,2,3],labels=np.round(s_arr[::-1], 1))
    a_ax.set_xticks([0,1,2,3,4],labels=['1','5','10','20','31'])
    norm = mpl.colors.Normalize(vmax=0)
    img = a_ax.imshow(m10_alphas, interpolation='nearest',cmap=cmap, norm=norm)
    for (j,i), label in np.ndenumerate(m10_alphas):
        a_ax.text(i,j, fr'\boldmath${label}$', ha='center', va='center',color='w', size=18)
    cb = plt.colorbar(img, ax=a_ax, norm=norm)
    cb.set_label(r'$\alpha$')
    ticks = np.array([-1.2,-1.0,-0.8,-0.6,-0.4,-0.2])
    cb.set_ticks(ticks=ticks,labels=ticks)
    cb.minorticks_off()
    a_ax.set_xlabel('Number of Bursts', size=s2)
    a_ax.set_ylabel(r'Mass Expelled Per Burst ($10^{7}M_{\odot}$)', size=s1)

    fig.savefig('full_cmap.pdf',bbox_inches='tight')
    return


if __name__=='__main__':
	main()
 
