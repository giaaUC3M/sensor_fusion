#############################
# author: lfornis@pa.uc3m.es
# methodology for the evaluation of sensor fusion design in real drone flights
# This code is an Python port of a Matlab script of our department
#############################
#############################################################
#                       IMPORTS                             #
#############################################################
import csv
from genericpath import isfile
import math
from ntpath import join
import os
import statistics
import pandas as pd
from pandas import read_table
import numpy as np
import navpy
import matplotlib.pyplot as plt
import matplotlib.colors as plot_color
import multiprocessing
import subprocess
import toCSV
import shutil
import argparse

#############################################################
#                 Command line interface                    #
#############################################################

parser = argparse.ArgumentParser(description='methodology for the evaluation of sensor fusion design in real drone flights')
parser.add_argument('-p', '--position',help='-p if you DONT want to use position data to calculate the rms innovation',action='store_false')
parser.add_argument('-v', '--velocity',help='-v if you DONT want to use velocity data to calculate the rms innovation',action='store_false')
parser.add_argument('-m', '--magnetometer',help='-m if you DONT want to use magnetometer data to calculate the rms innovation',action='store_false')
parser.add_argument('-ff', '--figures_format',type=str,help='-ff to indicate the figures format such as .png or .svg',default='.svg')
parser.add_argument('-s', '--superposition',help='-s if you want the graphs to show the data of each flight superimposed, this way you can better observe differences for the same time instant',action='store_true',default=False)
parser.add_argument('-pd', '--padding',help='enter the value in minutes of separation between one flight and another, to use it superposition must mut be desactivate. By default: -pd 1',default=1,type=int)
args = parser.parse_args()

ask_for=[]
if args.position: 
    ask_for.append('position')
if args.velocity: 
    ask_for.append('velocity')
if args.magnetometer:
    ask_for.append('magnetometer')
f_format=args.figures_format
superposition=args.superposition
pad=args.padding
#############################################################
#                         FUNCTIONS                         #
#############################################################
def rmsValue(array):
    n = len(array)
    squre = 0.0
    root = 0.0
    mean = 0.0
    
    #calculating Squre
    for i in range(0, n):
        squre += (array[i] ** 2)
    #Calculating Mean
    mean = (squre/ (float)(n))
    #Calculating Root
    root = math.sqrt(mean)
    return root

def create_dir_if_is_not(path):
    if not os.path.isdir(path):
        os.mkdir(path)

def get_variance(name_col):
    s=str(name_col).split('[')
    return s[0]+'_var['+s[1]

#############################################################
#                TRANSFORM .ULG TO .CSV                     #
#############################################################
wd=os.getcwd()
path=wd+'/ulogs_mision/'
files = [f for f in os.listdir(path) if 
             isfile(join(path, f))]
wd=wd+'/outputs/'
if  os.path.isdir(wd):
    shutil.rmtree(wd)
print('Creating output directory {:}'.format(wd))
os.mkdir(wd)
    
p_flights=wd+'csv_flights'
create_dir_if_is_not(p_flights)

p_EKFparam=wd+'EKFparameters'
create_dir_if_is_not(p_EKFparam)

flights=0  
ps=[]
for i, f in enumerate(files):
        if  not os.path.isdir(p_flights+'/flight'+str(i+1)):
            os.mkdir(p_flights+'/flight'+str(i+1))
        flights+=1
        p=multiprocessing.Process(
            target=toCSV.convert_ulog2csv,
            args=(path+f,
            'estimator_status,ekf2_innovations,vehicle_global_position,sensor_combined,vehicle_gps_position'
            ,p_flights+'/flight'+str(i+1),',',False),
        )
        ps.append(p)
        p.start()
        
        subprocess.run(['python','params.py','-i',path+f,p_EKFparam+'/params_flight_'+str(i+1)+'.csv'])

for p in ps:
    p.join()
#############################################################
#                    ARGUMENTS AND VARS                     #
#############################################################

wd=os.getcwd()
tables=[] #pasara a ser un np_array vuelosXparametro
param_names=[]#nombre de los parametros 

traduction={
    'position':['vel_pos_innov[3]','vel_pos_innov[4]','vel_pos_innov[5]'],
    'velocity':['vel_pos_innov[0]','vel_pos_innov[1]','vel_pos_innov[2]'],
    'magnetometer':['mag_innov[0]','mag_innov[1]','mag_innov[2]']
}
traduction2={0:'X',1:'Y',2:'Z'}
gps_ekf_dic_lon_lat={'lon':'Longitude (°)','lat':'Latitude (°)'}
gps_ekf_dic={'vel_n':'velocity North (m/s)','vel_e':'velocity East (m/s)','vel_d':'velocity Down (m/s)','alt':'Height (m)'}
orientation=['Roll','Pitch','Yaw']
f_breaks={
    'position':'pos_test_ratio',
    'velocity':'vel_test_ratio',
    'magnetometer':'mag_test_ratio',
    'height':'hgt_test_ratio'
}

#############################################################
#                   LECTURA DE DATOS                        #
#############################################################
wd=wd+'/outputs'
create_dir_if_is_not(wd)

for i in range(flights):
    path=wd+'/csv_flights/flight'+str(i+1)+'/'  
    files = [f for f in os.listdir(path) if 
                isfile(join(path, f))]
    for j, f in enumerate(files):
        tables.append(read_table(path+f,sep=','))
        if i==0:
            f_s=f.split('_')
            param_name=''
            for k,fs in enumerate(f_s):
                if k!=0 and k!=1 and k!=2 and k!=(len(f_s)-1):
                    if param_name=='':
                        param_name=fs
                    else: 
                        param_name=param_name+'_'+fs
            param_names.append(param_name)
        
tables=np.asarray(tables,dtype=object)
tables=tables.reshape(flights,len(files))
#parameters
param_ekf=[]
path=wd+'/EKFparameters/' 
files = [f for f in os.listdir(path) if isfile(join(path, f))]
for j, f in enumerate(files):
    if j==0:
        param_ekf_modified=pd.read_table(path+f,sep=',', header=None)
    else: 
        param_ekf=pd.read_table(path+f,sep=',', header=None)
        param_ekf.columns=[0,(j+1)]
        param_ekf_modified=pd.concat([param_ekf_modified,param_ekf[(j+1)]], axis=1)

rows_to_drop=[]
for i in param_ekf_modified.index:
    param_ekf=[]
    for j,col in enumerate(param_ekf_modified.columns):
        if j!=0:
            param_ekf.append(param_ekf_modified[col][i])
    if len(set(param_ekf))==1:
        rows_to_drop.append(i)
param_ekf_modified=param_ekf_modified.drop(param_ekf_modified.index[rows_to_drop])
param_ekf_modified.index=list(param_ekf_modified[0])
param_ekf_modified.drop([0], axis=1,inplace=True)
param_ekf_modified.columns=range(flights)

print()
print("-- PARAMETERS MODIFIED:")
print(param_ekf_modified)
print()

param_ekf_modified.to_csv(wd+'/param_ekf_modified.csv')
param_ekf_modified.to_csv(wd+'/param_ekf_modified.txt', sep="\t", quoting=csv.QUOTE_NONE, escapechar=" ")
#############################################################
#              QUATERNIONES y T.INNOVACIONES                #
#############################################################
status=param_names.index('estimator_status')
innov=param_names.index('ekf2_innovations')

angles=[]
rms_innov_tot=[]
innov_tot=[]
rms_innov=[]
parmaters=[]

for v in range(flights):

    angles.append(navpy.quat2angle(
    tables[v,status]['states[0]'],[tables[v,status]['states[1]'],
    tables[v,status]['states[2]'],tables[v,status]['states[3]']]))

    d = pd.DataFrame(np.zeros((tables[v,innov].shape[0], 1)))
    rms_innov_input=[]
    cols_names=[]
    for a in ask_for:
        cols=traduction.get(a)
        for i,c in enumerate(cols):
            tri=tables[v,innov][c]/(tables[v,innov][get_variance(c)].pow(1/2))
            rms_innov_input.append(rmsValue(tri))
            d[0]=d[0]+tri.pow(2)
            cols_names.append('rms innov '+a+traduction2.get(i))
    rms_innov.append(rms_innov_input)
    innov_tot.append(d[0])
    rms_innov_tot.append(rmsValue(d[0]))
    
df_rms_innov= pd.DataFrame(rms_innov,columns=cols_names)
df_rms_innov_tot= pd.DataFrame(rms_innov_tot,columns=['rms innov total'])
df_rms=pd.concat([df_rms_innov, df_rms_innov_tot], axis=1)
df_rms=df_rms.T

print()
print("-- RMS INNOVATIONS:")
print(df_rms)
print()

df_rms.to_csv(wd+'/rms_innovations.csv')
df_rms.to_csv(wd+'/rms_innovations.txt', sep="\t", quoting=csv.QUOTE_NONE, escapechar=" ")

angles=np.asarray(angles,dtype=object)
angles=angles.reshape(flights,3)
#############################################################
#                       FIGURES                             #
#############################################################
gps=param_names.index('vehicle_gps_position')
global_pos=param_names.index('vehicle_global_position')
sensors=param_names.index('sensor_combined')

#___________PATHS:

path=wd+'/figures/' 
create_dir_if_is_not(path)
pg='graphics/'
#path graficas
ph='histogram/'
#path boxplot
pb='boxPlots/'
#NormalizedInnov
pn=path+'NormalizedInnov/'
create_dir_if_is_not(pn)
create_dir_if_is_not(pn+pg)
create_dir_if_is_not(pn+pb)
for a in ask_for:
    create_dir_if_is_not(pn+pg+a)
    create_dir_if_is_not(pn+pb+a)
pgyr=path+'Gyroscope/'
create_dir_if_is_not(pgyr)
create_dir_if_is_not(pgyr+pg)
create_dir_if_is_not(pgyr+ph)
pacc=path+'Accelerometer/'
create_dir_if_is_not(pacc)
create_dir_if_is_not(pacc+pg)
create_dir_if_is_not(pacc+ph)
pfb=path+'FusionBreaks/'
create_dir_if_is_not(pfb)
create_dir_if_is_not(pfb+pg)
pgpsekf=path+'GPSinputs_EKFoutputs/'
create_dir_if_is_not(pgpsekf)
create_dir_if_is_not(pgpsekf+pg)
pao=path+'AngularOrientation/'
create_dir_if_is_not(pao)
create_dir_if_is_not(pao+pg)

def plot_graphic_boxplot_hist(xs,ys,col,xlabel,ylabel,path,pathBoxplot=None,pathHist=None):
    if pathBoxplot!=None:
        x_bp=[]
    if pathHist!=None:
        x_h=[]
    fig,grap=plt.subplots()#figsize=(12, 4)) 
    delay=[None]*len(ys)
    for v in range(flights):
        color=False
        for i,y in enumerate(ys):    
            time=tables[v,y]['timestamp']
            if not superposition:
                if v==0: 
                    in_zero=time.iloc[0]
                else:
                    time=time-(time.iloc[0]-delay[i]-pad) 
                delay[i]=time.iloc[len(time)-1]
                time=time-in_zero
            else:
                time=time-time.iloc[0]
            if y==gps and str(col[0]).find('vel')!=-1:
                y_data=tables[v,y][col[0]+'_m_s']
            else:
                y_data=tables[v,y][col[0]]
            if y==innov:
                y_data=y_data/tables[v,y][get_variance(col[0])].pow(1/2)
            
            if not color:                
                if xs[0]=='Time':
                    line,=grap.plot(time,y_data)
                else:
                    line,=grap.plot(tables[v,y][col[0]],tables[v,y][col[1]])
                color=line.get_color()
            else:
                if xs[0]=='Time':
                    line,=grap.plot(time,y_data,color=color)
                else:
                    line,=grap.plot(tables[v,y][col[0]],tables[v,y][col[1]],color=color)
            
            if y==gps:
                line.set_color('None')
                line.set_marker('o')
                line.set_markeredgecolor(plot_color.hex2color(color))
                line.set_markersize(line.get_markersize()*2)
                line.set_markevery(2)
                line.set_label('GPS input flight '+str(v+1))
            elif y==global_pos:
                line.set_label('EKF output flight '+str(v+1))
            else:
                line.set_label('Flight '+str(v+1))
            if pathBoxplot!=None:
                x_bp.append(y_data)#solo para una y
            if pathHist!=None:
                x_h.append(y_data)#solo para una y
    grap.set_xlabel(xlabel)
    grap.set_ylabel(ylabel)
    grap.grid(color='gray', linestyle='dashed', linewidth=1, alpha=0.4)
    grap.legend()
    if path[-1]!='/':
        path+='/'
    s=path.split('figures/')
    s=s[1].split('/')
    grap.set_title(s[0])
    grap_ylabel=ylabel.replace('/s^2','*s^-2')
    grap_ylabel=grap_ylabel.replace('/s','*s^-1')
    fig.savefig(path+xlabel+'-'+grap_ylabel+f_format)
    plt.close(fig)
    if pathBoxplot!=None:
        plot_boxplot(x_bp,ylabel,pathBoxplot)
    if pathHist!=None:
        plot_hist(x_h,ylabel,pathHist)

def plot_boxplot(x_bp,ylabel,pathBoxplot):
    if pathBoxplot[-1]!='/':
        pathBoxplot+='/'
    fig,bp=plt.subplots()
    bp.boxplot(x_bp,range(1,flights+1))
    bp.set_xlabel('flight')
    bp.set_ylabel(ylabel)
    bp.grid(color='gray', linestyle='dashed', linewidth=1, alpha=0.4)
    ylabel=ylabel.replace('/s^2','*s^-2')
    ylabel=ylabel.replace('/s','*s^-1')
    fig.savefig(pathBoxplot+ylabel+f_format)

def plot_hist(x_h,xlabel,pathHist):
    x_h=pd.DataFrame(x_h)
    x_h=x_h.values.tolist()
    xi=[]
    for xh in x_h:
        xi+=xh
    xi = [x for x in xi if np.isnan(x) == False]
    mean = statistics.mean(xi)
    std = statistics.stdev(xi)
    x=[]
    for i in xi:
        if abs(i-mean)<5*std:
            x.append(i)
    fig,h=plt.subplots()
    h.hist(x,50)
    h.set_xlabel(xlabel)
    h.set_title(xlabel+' histogram')
    h.grid(color='gray', linestyle='dashed', linewidth=1, alpha=0.4)
    xlabel=xlabel.replace('/s^2','*s^-2')
    xlabel=xlabel.replace('/s','*s^-1')
    fig.savefig(pathHist+xlabel+f_format)


def plot_orientation(x,y,xlabel,ylabel,path):
    fig,grap=plt.subplots()
    color=[False]*3
    for v in range(flights):
        time=tables[v,x]['timestamp']
        if not superposition:
            if v==0: 
                in_zero=time.iloc[0]
            else:
                time=time-(time.iloc[0]-delay-pad) 
            delay=time.iloc[len(time)-1]
            time=time-in_zero
        else:
            time=time-time.iloc[0]
        for i in range(3):
            line,=grap.plot(time,y[v][i])
            if not color[i]:
                color[i]=line.get_color()
            else:
                line.set_color(color[i])
    grap.set_xlabel(xlabel)
    grap.set_ylabel(ylabel)
    grap.grid(color='gray', linestyle='dashed', linewidth=1, alpha=0.4)
    grap.legend(orientation)
    if path[-1]!='/':
        path+='/'
    s=path.split('/')
    grap.set_title(s[-2])
    ylabel=ylabel.replace('/s^2','*s^-2')
    ylabel=ylabel.replace('/s','*s^-1')
    fig.savefig(path+xlabel+'-'+ylabel+f_format)
    plt.close(fig)
#preprocesado
min=(60*(10**6))
for v in range(flights):
    tables[v,innov]['timestamp']=tables[v,innov]['timestamp']/min
    tables[v,gps]['timestamp']=tables[v,gps]['timestamp']/min
    tables[v,global_pos]['timestamp']=tables[v,global_pos]['timestamp']/min
    tables[v,status]['timestamp']=tables[v,status]['timestamp']/min
    tables[v,sensors]['timestamp']=tables[v,sensors]['timestamp']/min
    tables[v,gps]['alt']=tables[v,gps]['alt']*(10**-3)
    tables[v,gps][['lat','lon']]=tables[v,gps][['lat','lon']]*(10**-7)

print("Making graphics, boxplots, histograms...")
      
for a in ask_for:
    cols=traduction.get(a)
    for i,c in enumerate(cols):
        plot_graphic_boxplot_hist(['Time'],[innov],[c],'Time (min)',a+traduction2.get(i)+'_normalized_innovations',pn+pg+a,pathBoxplot=pn+pb+a)
    plot_boxplot(innov_tot,'Total normalized innovations',pn+pb) 
    plot_graphic_boxplot_hist(['Time'],[status],[f_breaks.get(a)],'Time (min)',a+' test ratio',pfb+pg)
plot_graphic_boxplot_hist(['Time'],[status],[f_breaks.get('height')],'Time (min)','height test ratio',pfb+pg)
for b in gps_ekf_dic:    
    plot_graphic_boxplot_hist(['Time'],[gps,global_pos],[b],'Time (min)',gps_ekf_dic.get(b),pgpsekf+pg)
for i in range(3):
    plot_graphic_boxplot_hist(['Time'],[sensors],['gyro_rad['+str(i)+']'],'Time (min)','Gyro '+traduction2.get(i)+' reading (rad/s)',pgyr+pg,pathHist=pgyr+ph)
    plot_graphic_boxplot_hist(['Time'],[sensors],['accelerometer_m_s2['+str(i)+']'],'Time (min)','Acc '+traduction2.get(i)+' reading (m/s^2)',pacc+pg,pathHist=pacc+ph)
plot_graphic_boxplot_hist([gps,global_pos],[gps,global_pos],['lon','lat'],gps_ekf_dic_lon_lat.get('lon'),gps_ekf_dic_lon_lat.get('lat'),pgpsekf+pg)
plot_orientation(status,angles,'Time (min)','Attitude: roll,pitch & yaw (rad)',pao+pg)

#######
#   PARA EL VIDEO
#########
def plot_graphic_video(xs,ys,col,xlabel,ylabel,path,pathBoxplot=None,pathHist=None):
    path=os.getcwd()+'/video/'
    fig,grap=plt.subplots()
    n=150
    for v in range(2):
        color=False
        for i,y in enumerate(ys):
            if v==0:  
                if not color:                
                    line,=grap.plot(tables[v,y][col[0]].iloc[0:n],tables[v,y][col[1]].iloc[0:n])
                    color=line.get_color()
                else:
                    line,=grap.plot(tables[v,y][col[0]].iloc[0:n],tables[v,y][col[1]].iloc[0:n],color=color)
                if y==gps:
                    line.set_color('None')
                    line.set_marker('o')
                    line.set_markeredgecolor(plot_color.hex2color(color))
                    line.set_markersize(line.get_markersize()*2)
                    line.set_markevery(2)
                    line.set_label('GPS input flight '+str(v+1))
                elif y==global_pos:
                    line.set_label('EKF output flight '+str(v+1))
                else:
                    line.set_label('Flight '+str(v+1))
        if v==1:
            for j in range(n):
                if not color:   
                    line1,=grap.plot(tables[v,ys[1]][col[0]].iloc[j],tables[v,ys[1]][col[1]].iloc[j])          
                    line1.set_label('EKF output flight '+str(v+1))
                    color=line1.get_color()
                    line,=grap.plot(tables[v,ys[0]][col[0]].iloc[j],tables[v,ys[0]][col[1]].iloc[j],color=color)   
                    line.set_label('GPS input flight '+str(v+1))
                else:
                    line1,=grap.plot(tables[v,ys[1]][col[0]].iloc[j],tables[v,ys[1]][col[1]].iloc[j],color=color)
                    line,=grap.plot(tables[v,ys[0]][col[0]].iloc[j],tables[v,ys[0]][col[1]].iloc[j],color=color)
                    
                line.set_color('None')
                line.set_marker('o')
                line.set_markeredgecolor(plot_color.hex2color(color))
                line.set_markersize(line.get_markersize()*2)
                line.set_markevery(2)

                grap.set_xlabel(xlabel)
                grap.set_ylabel(ylabel)
                grap.grid(color='gray', linestyle='dashed', linewidth=1, alpha=0.4)
                grap.legend()
                fig.savefig(path+str(j)+f_format)
                
    plt.close(fig)
 
plot_graphic_video([gps,global_pos],[gps,global_pos],['lon','lat'],gps_ekf_dic_lon_lat.get('lon'),gps_ekf_dic_lon_lat.get('lat'),pgpsekf+pg)