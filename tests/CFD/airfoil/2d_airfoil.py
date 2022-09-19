# ------------------------------------------------------------------------------------#
#  2-D airfoil parameterization and transonic flow simulation                         #
#                                                                                     #
#  Author: Ahmed H. Bayoumy                                                           #
#  email: ahmed.bayoumy@mail.mcgill.ca                                                #
#                                                                                     #
#  This program is free software: you can redistribute it and/or modify it under the  #
#  terms of the GNU Lesser General Public License as published by the Free Software   #
#  Foundation, either version 3 of the License, or (at your option) any later         #
#  version.                                                                           #
#                                                                                     #
#  This program is distributed in the hope that it will be useful, but WITHOUT ANY    #
#  WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A    #
#  PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more details.   #
#                                                                                     #
#  You should have received a copy of the GNU Lesser General Public License along     #
#  with this program. If not, see <http://www.gnu.org/licenses/>.                     #
#                                                                                     #
#  You can find information on simple_mads at                                         #
#  https://github.com/Ahmed-Bayoumy/RAF/tree/master/tests/CFD/airfoil                 #
# ------------------------------------------------------------------------------------#

from cmath import pi
import numpy as np
from dataclasses import dataclass
from typing import Dict, List
from abc import ABC
from matplotlib import pyplot as plt
import os
from shutil import copy
from subprocess import CalledProcessError, Popen, PIPE



@dataclass
class airfoil_generator:
  """ Class for generating airfoil 2d geometries """
  chord: float = 1.
  _nCPs: int = 3
  nPts: int = 3
  degree_t: int = 4
  degree_b: int = 4
  params: List[float] = None
  tegap: float = 0.001263
  _CPt: List[float] = None
  _CPb: List[float] = None
  _t: List[float] = None
  _dim: int = 2
  xt: List = None
  xb: List = None
  yt: List = None
  yb: List = None
  thickness: List = None
  xs: List = None
  ys: List = None
  Pxt: List = None
  Pyt: List = None
  Pxb: List = None
  Pyb: List = None
  cpx: List = None
  cpu: List = None
  cpl: List = None

  def __init__(self, chord = 1, nPts = 3, degree_t = 4, degree_b = 4, params = None, tegap = 0.001263):
    self.tegap = tegap
    self.chord = chord
    self.nPts = nPts
    self.degree_t = degree_t
    self.degree_b = degree_b
    self.params = params

  def generate(self):
    self.Pxt = np.zeros((self.nPts+2))
    self.Pyt = np.zeros((self.nPts+2))
    self.Pxb = np.zeros((self.nPts+2))
    self.Pyb = np.zeros((self.nPts+2))

    for i in range(self.nPts-1):
      dob_i = (2*(i+1))-1
      # x-coordinates of top
      self.Pxt[i+2] = self.params[dob_i]
      # x-coordinates of bottom
      self.Pxb[i+2] = self.params[dob_i-1+2*self.nPts]
      # y-coordinates of top
      self.Pyt[i+1] = self.params[dob_i-1]
      # y-coordinates of bottom
      self.Pyb[i+1] = self.params[dob_i-2+2*self.nPts]

    #set last y-points
    self.Pyt[self.nPts]=self.params[2*(self.nPts)-2]
    self.Pyb[self.nPts]=self.params[4*(self.nPts)-3]
    #set trailing edge points
    self.Pxt[self.nPts+1]=1
    self.Pxb[self.nPts+1]=1
    self.Pyt[self.nPts+1]=0.5* self.tegap
    self.Pyb[self.nPts+1]=-0.5* self.tegap
    
    self._CPt = []
    self._CPb = []
    for i in range(len(self.Pxt)):
      self._CPt.append([self.Pxt[i], self.Pyt[i], 0.])
      self._CPb.append([self.Pxb[i], self.Pyb[i], 0.])

    # top x and y values
    self.bspline_f(self.degree_t, self._CPt, 100, "top")

    # bottom x and y values
    self.bspline_f(self.degree_b, self._CPb, 100, "bottom")

    xt = self.xt
    xb = [x for x in self.xb]
    yt = self.yt
    yb = [y for y in self.yb]

    self.thickness = np.zeros((100, 1))

    for i in range(1,99):
      #find nearest x-location on bottom surface
      #first check bottom point with the same index, and see if points adjacent are nearer or closer to to xt
      dist0=abs(xt[i]-xb[i])             #point with same index
      distright=abs(xt[i]-xb[i+1])		   #point with index nearer to TE
      distleft=abs(xt[i]-xb[i-1])		     #point with index nearer LE
      if dist0<distright and dist0<distleft:
        self.thickness[i] = yt[i] - yb[i]
      elif dist0>distright and dist0<distleft:
        j = i + 1
        while j<99 and abs(xt[i]-xb[j+1])<abs(xt[i]-xb[j]):
          j+=1
        if j == 148 and abs(xt[i]-xb[j+1])<abs(xt[i]-xb[j]):
          j+=1
        self.thickness[i] = yt[i] -yb[j]
      elif dist0<distright and dist0 > distleft:
        j=i-1
        while j>0 and abs(xt[i]-xb[j-1])<abs(xt[i]-xb[j]):
          j-=1
        if j == 1 and abs(xt[i] - xb[j-1])<abs(xt[i]-xb[j]):
          j-=1
        self.thickness[i]=yt[i]-yb[j]

    self.thickness[99] = self.tegap

    # max airfoil thickness
    t_max = max(self.thickness)
    # min airfoil thickness
    t_min = min(self.thickness)

    # Trailing edge wedge angle
    slbot = (yb[96]-yb[99])/(xb[95]-xb[99])
    sltop = (yt[96]-yt[99])/(xt[95]-xt[99])

    te_ang = (np.arctan(slbot)-np.arctan(sltop))*180/np.pi
    self.xs: List = np.flip(xt).tolist()
    self.ys: List = np.flip(yt).tolist()
    [self.xs.append(xb[i]) for i in range(len(xb))]
    [self.ys.append(yb[i]) for i in range(len(yb))]
    out = []
    for i in range(len(self.xs)):
      out.append([self.xs[i], self.ys[i]])
    
    return self.xs,self.ys,t_max,t_min,te_ang

  def visualize(self):
    plt.xlim([0,1])
    plt.ylim([-0.3, 0.3])
    plt.plot(self.xs, self.ys)
    plt.plot(self.Pxt, self.Pyt, '--o')
    plt.plot(self.Pxb, self.Pyb, '--o')
    plt.show()
    


  def bspline_f(self, d, p, npts, side: str):
    # number of control points - 1
    n = len(p)-1
    type = 1

    if len(p[0]) < 2 :
      raise IOError('Control points should have at least x and y components.')
    elif len(p[0])-1 > 2:
      raise IOError('Incorrect control points definition. P should have as many rows as the spatial dimensions (either 2 or 3).')

    if d > n:
      raise IOError('Degree must be less than the number of control points.')
    
    if d <1:
      raise IOError('Degree must be an integer 1 or greater.')
    self._dim = len(p[0])
    #Create knots vector
    knots=np.zeros(d+n+2)
    for i in range(d+n+2):
        if i<d+2-1:
            knots[i]=0
        elif d+2-1<=i and i<n+2-1:
            knots[i]=(i+1-d-1)/(n+1-d)
        else:
            knots[i]=1

    #orm t vector
    if type==1:  #linear spacing
        self.t=np.linspace(0,1,npts)
    else:        #log spacing
        self.t=np.logspace(-1,np.log10(1.1),npts)-0.1
    
    Q = np.zeros((self._dim, npts))
    N = np.zeros(n+1)

    for l in range(npts):
      l
      for i in range(-1, n):
        v = self.basis_fun(d, i+1, knots, self.t[l])
        N[i+1] = v
        #COMPLETE: Check the rest of this loop
        st = [list(x) for x in zip(*p)]
        s = [x[i+1]*N[i+1] for x in st]
        # 
        Q[:, l] = np.add(Q[:, l], s)
    
    if side == "top":
      self.xt = Q[0,:]
      self.yt = Q[1, :]
    else:
      self.xb = Q[0,:]
      self.yb = Q[1, :]
    

  def basis_fun(self, j, i, knots, t) -> int:
    m = len(knots)
    val: int = 0
    k = knots
    if j == 0:
      if k[i] <= t and t < k[i+1]:
        val = 1
      elif k[i] <= t and t == k[i+1] and k[i+1] == 1:
        val = 1 
      else:
        val = 0
    else:
      if k[i] < k[i+j]:
        val = (t-k[i])/(k[i+j]-k[i]) * self.basis_fun(j-1, i, k, t)
      else:
        val = 0
      if i < m:
        if k[i+1] < k[i+j+1]:
          val += (k[i+j+1]-t)/(k[i+j+1]-k[i+1])*self.basis_fun(j-1, i+1, k, t)
    return val
  
  def create_sim_files(self, initFile:str = 'Initial.inp', id: int = 0):
    # COMPLETE: Prepare required input files to run both TSFOIL and OPENFOAM CFD simulations
    I = np.ones((len(self.xs)))
    x1 = np.flip([x for x in self.xs[0:int(len(self.xs)/2)]]).tolist()
    x2 = [x for x in self.xs[int(len(self.xs)/2):]]
    y1 = np.flip([y for y in self.ys[0:int(len(self.ys)/2)]]).tolist()
    y2 = [y for y in self.ys[int(len(self.ys)/2):]]
    x: List = x1
    x+=x2
    y: List = y1
    y+=y2
    L = [x[int(len(x)/2):], y[int(len(y)/2):]]
    U = [x[0:int(len(x)/2)], y[0:int(len(y)/2)]]
    srcf = os.path.join(os.getcwd(), 'tests/CFD/TSFOIL/'+initFile) 
    destf = os.path.join(os.getcwd(), f'tests/CFD/TSFOIL/Design{id}.inp') 

    copy(srcf, destf)
    with open(destf, 'a') as f:
      f.write('\n')
      for i in range(len(U[0])):
        f.write(f'   {U[0][i]:1.5f}   {U[1][i]:1.5f} \n')
      f.write(' LOWER SURFACE \n')
      for i in range(len(L[0])):
        f.write(f'   {L[0][i]:1.5f}   {L[1][i]:1.5f} \n')
  
  def run(self, id: int = 0):
    old = os.getcwd()
    os.chdir(os.path.join(old, 'tests/CFD/TSFOIL'))
    cmd = './TSFOIL'
    input_data = os.linesep.join(['default', f'Design{id}.out', f'Design{id}.inp', os.linesep])
    p = Popen(cmd, stdin=PIPE, bufsize=0)
    p.communicate(input_data.encode('ascii'))
    if p.returncode != 0:
      raise CalledProcessError(p.returncode, cmd)
  
  def get_responses(self, id: int = 0):
    destf = os.path.join(os.getcwd(), f'Design{id}.out') 
    f = open(destf, 'r')
    h = 0
    index = []
    iBlock = -1
    table = []
    get = False
    for line in f:
      iBlock += 1
      temp = line
      if get and temp.strip().split('   ')[0] != 'AIRFOIL TRAILING EDGE' and temp.strip().split('   ')[0] != 'AIRFOIL LEADING EDGE' and line != '0\n' and line != '0' and line != '\n':
        table.append([float(x) for x in line.strip().split(' ') if x.lstrip('-').replace('.','',1).isdigit()])
      if line == '   I        X          CP          M1              CP          M1\n' or line == '   I        X          CP          M1              CP          M1\n':
        table = []
        get = True
      if line == '0\n':
        get = False
    
    f.close()
    self.cpx = [s[1] for s in table]
    self.cpl = [s[3] for s in table]
    self.cpu = [s[5] for s in table]

    

  def plot_cp(self):
    plt.plot(self.cpx, self.cpl)
    plt.plot(self.cpx, self.cpu)
    plt.title('Cp')
    plt.xlim([0,1])
    plt.show()



if __name__ == "__main__":
  # COMPLETE: Ready for use
  design = airfoil_generator(tegap=0.001263,
  degree_t=4,
  degree_b=4,
  chord = 1,
  nPts = 3,
  #        [yt1 , xt2,  yt2, xt3,  yt3,  yb1,xb2,   yb2,xb3, yb3  ]
  params = [0.05,0.25,0.155,0.85,0.043,-0.07,0.4,-0.045,0.8,-0.005]
  )

  design.generate()
  design.visualize()
  design.create_sim_files()
  design.run()
  design.get_responses()
  design.plot_cp()
