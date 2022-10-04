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
import pandas as pd
from matplotlib import pyplot as plt
import os
from shutil import copy
from subprocess import CalledProcessError, Popen, PIPE
import copy
from numpy import linspace, zeros, ones, sin, cos, arctan, pi
import subprocess



@dataclass
class airfoil_generator:
  """ Class for generating airfoil 2d geometries """
  chord: float = 1.
  _nCPs: int = 3
  nPts: int = 3
  degree_t: int = 4
  degree_b: int = 4
  params: List[float] = None
  te_gap: float = 0.001263
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
  Xu: List = None
  Zu: List = None
  Xl: List = None
  Zl: List = None
  num_pts: int = 64

  def __init__(self, chord = 1, nPts = 3, degree_t = 4, degree_b = 4, params = None, tegap = 0.001263):
    self.te_gap = tegap
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
    self.Pyt[self.nPts+1]=0.5* self.te_gap
    self.Pyb[self.nPts+1]=-0.5* self.te_gap
    
    self._CPt = []
    self._CPb = []
    for i in range(len(self.Pxt)):
      self._CPt.append([self.Pxt[i], self.Pyt[i], 0.])
      self._CPb.append([self.Pxb[i], self.Pyb[i], 0.])

    # top x and y values
    self.b_spline_f(self.degree_t, self._CPt, self.num_pts, "top")

    # bottom x and y values
    self.b_spline_f(self.degree_b, self._CPb, self.num_pts, "bottom")

    xt = self.xt
    xb = [x for x in self.xb]
    yt = self.yt
    yb = [y for y in self.yb]

    self.thickness = np.zeros((self.num_pts, 1))

    for i in range(1,self.num_pts-1):
      #find nearest x-location on bottom surface
      #first check bottom point with the same index, and see if points adjacent are nearer or closer to to xt
      dist0=abs(xt[i]-xb[i])             #point with same index
      dist_right=abs(xt[i]-xb[i+1])		   #point with index nearer to TE
      dist_left=abs(xt[i]-xb[i-1])		     #point with index nearer LE
      if dist0<dist_right and dist0<dist_left:
        self.thickness[i] = yt[i] - yb[i]
      elif dist0>dist_right and dist0<dist_left:
        j = i + 1
        while j<99 and abs(xt[i]-xb[j+1])<abs(xt[i]-xb[j]):
          j+=1
        if j == 148 and abs(xt[i]-xb[j+1])<abs(xt[i]-xb[j]):
          j+=1
        self.thickness[i] = yt[i] -yb[j]
      elif dist0<dist_right and dist0 > dist_left:
        j=i-1
        while j>0 and abs(xt[i]-xb[j-1])<abs(xt[i]-xb[j]):
          j-=1
        if j == 1 and abs(xt[i] - xb[j-1])<abs(xt[i]-xb[j]):
          j-=1
        self.thickness[i]=yt[i]-yb[j]

    self.thickness[self.num_pts-1] = self.te_gap

    # max airfoil thickness
    t_max = max(self.thickness)
    # min airfoil thickness
    t_min = min(self.thickness)

    # Trailing edge wedge angle
    sl_bot = (yb[self.num_pts-4]-yb[self.num_pts-1])/(xb[self.num_pts-5]-xb[self.num_pts-1])
    sl_top = (yt[self.num_pts-4]-yt[self.num_pts-1])/(xt[self.num_pts-5]-xt[self.num_pts-1])

    te_ang = (np.arctan(sl_bot)-np.arctan(sl_top))*180/np.pi
    self.Xu = copy.deepcopy(xt)
    self.Zu = copy.deepcopy(yt)
    self.Xl = copy.deepcopy(xb)
    self.Zl = copy.deepcopy(yb)
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
    


  def b_spline_f(self, d, p, n_pts, side: str):
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
        self.t=np.linspace(0,1,n_pts)
    else:        #log spacing
        self.t=np.logspace(-1,np.log10(1.1),n_pts)-0.1
    
    Q = np.zeros((self._dim, n_pts))
    N = np.zeros(n+1)

    for l in range(n_pts):
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
  
  def create_TSFOIL_sim_files(self, initFile:str = 'Initial.inp', id: int = 0):
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
    src_f = os.path.join(os.getcwd(), 'tests/CFD/TSFOIL/'+initFile) 
    dest_f = os.path.join(os.getcwd(), f'tests/CFD/TSFOIL/Design{id}.inp') 

    copy(src_f, dest_f)
    with open(dest_f, 'a') as f:
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
  
  def get_responses(self, file_name: str = 'Design0', solver: str = "TSFOIL"):
    if solver == "TSFOIL":
      dest_f = os.path.join(os.getcwd(), file_name) 
      f = open(dest_f, 'r')
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
    elif solver == "SU2":
      f = os.path.join(os.getcwd(), file_name) 
      data = pd.read_csv(f, header=None)
      self.CL = data.iloc[-1, 7]
      self.CD = data.iloc[-1, 6]
      self.LDR = data.iloc[-1, 8]
    else:
      raise IOError(f'The flow solver {solver} is not supported! We only support SU2 and TSFOIL.')
    
    return self.CL, self.CD, self.LDR

  def gen_OF_blockmeshdict_from_coords(self, alpha_deg=4):
    """ This function generates an OpenFoam block mesh file from the airfoil coordinates """
    #TODO: This routine is ready for use, but current numerical setup takes too long so will revisit to make it simpler and faster 
    Xu = copy.deepcopy(self.Xu)
    Zu = copy.deepcopy(self.Zu)
    Xl = copy.deepcopy(self.Xl)
    Zl = copy.deepcopy(self.Zl)
    thickness = copy.deepcopy(self.thickness)

    alpha = np.deg2rad(alpha_deg)  # Angle of attack (in radians)
    c = 1.
    

    # Mesh dimensions
    scale = 1            # Scaling factor
    H = 8                # *Half* height of channel
    W = 0.5              # *Half* depth of foil (y-direction)
    D = 16               # Length of downstream section

    # Mesh resolution parameters
    Ni = 400             # Number of interpolation points along the foil
    Nx = 250             # Number of mesh cells along the foil
    ND = 150             # Number of cells in the downstream direction
    NT = 100             # Number of cells the transverse direction
    NW = 1               # Number of cells in the y-direction (along the foil axis)

    # Expansion rates
    ExpT = 500           # Expansion rate in transverse direction
    ExpD = 100           # Expansion rate in the downstream direction
    ExpArc = 50          # Expansion rate along the inlet arc

    # Create a vector with x-coordinates, camber and thickness
    beta = linspace(0, pi, Ni)
    x = c*(0.5*(1 - cos(beta)))
    # plt.plot(Xu, Zu)
    # plt.plot(Xl, Zl)
    # plt.show()
    # Rotate foil to reach specified angle of attack
    upper = np.matrix([[cos(alpha), sin(alpha)],
                       [-sin(alpha), cos(alpha)]])*np.vstack((self.Xu, Zu))
    lower = np.matrix([[cos(alpha), sin(alpha)],
                      [-sin(alpha), cos(alpha)]])*np.vstack((Xl, Zl))

    Xu = upper[0, :].conj().transpose()
    Zu = upper[1, :].conj().transpose()
    Xl = lower[0, :].conj().transpose()
    Zl = lower[1, :].conj().transpose()

    


    # Otherwise use location of max. thickness
    C_max_idx = np.where(thickness == max(thickness))[0][0]


    # Move point of mesh "nose"
    NoseX = (-H + Xu[C_max_idx])*cos(alpha)
    NoseZ = -(-H + Xu[C_max_idx])*sin(alpha)


    # Calculate the location of the vertices on the positive y-axis and put them in a matrix
    nv = 12
    vertices = zeros((nv, 3))

    vertices[0, :] = [NoseX[0], W, NoseZ[0]]
    vertices[1, :] = [Xu[C_max_idx], W, H]
    vertices[2, :] = [Xu[-1], W, H]
    vertices[3, :] = [D, W, H]
    vertices[4, :] = [Xu[0], W, Zu[0]]
    vertices[5, :] = [Xu[C_max_idx], W, Zu[C_max_idx]]
    vertices[6, :] = [Xl[C_max_idx], W, Zl[C_max_idx]]
    vertices[7, :] = [Xu[-1], W, Zu[-1]]
    vertices[8, :] = [D, W, Zu[-1]]
    vertices[9, :] = [Xl[C_max_idx], W, -H]
    vertices[10, :] = [Xu[-1], W, -H]
    vertices[11, :] = [D, W, -H]

    fig = plt.figure()
    ax = plt.axes()


    # ax.plot3D(vertices[:,0], vertices[:,1], vertices[:,2], '--r')
    # ax.view_init(0, -90)
    # [ax.text(vertices[i,0], vertices[i,1], vertices[i,2], str(i)) for i in range(nv)]

    # Create vertices for other side (negative y-axis)
    vertices2 = vertices.copy()
    vertices2[:, 1] *= -1

    # ax.plot3D(vertices2[:,0], vertices2[:,1], vertices2[:,2], '-b')
    # ax.view_init(0, -90)
    # [ax.text(vertices2[i,0], vertices2[i,1], vertices2[i,2], str(i)) for i in range(nv, 2*nv)]
    vertices = np.vstack((vertices, vertices2))




    # Edge 4-5 and 16-17
    pts1 = np.concatenate([Xu[1:C_max_idx], W*ones(np.shape(Xu[1:C_max_idx])),
                           Zu[1:C_max_idx]], axis=1)
    
    pts5 = np.concatenate([pts1[:, 0], -pts1[:, 1], pts1[:, 2]], axis=1)
    


    # Edge 5-7 and 17-19
    pts2 = np.concatenate([Xu[C_max_idx + 1:Ni - 1],
                           W*ones(np.shape(Xu[C_max_idx + 1:Ni - 1])),
                           Zu[C_max_idx + 1:Ni - 1]], axis=1)
    

    pts6 = np.concatenate([pts2[:, 0], -pts2[:, 1], pts2[:, 2]], axis=1)
    


    # Edge 4-6 and 16-18
    pts3 = np.concatenate([Xl[1:C_max_idx], W*ones(np.shape(Xl[1:C_max_idx])),
                           Zl[1:C_max_idx]], axis=1)
    

    pts7 = np.concatenate([pts3[:, 0], -pts3[:, 1], pts3[:, 2]], axis=1)
    


    # Edge 6-7 and 18-19
    pts4 = np.concatenate([Xl[C_max_idx + 1:Ni - 1],
                          W*ones(np.shape(Xl[C_max_idx + 1:Ni - 1])),
                          Zl[C_max_idx + 1:Ni - 1]], axis=1)
    

    pts8 = np.concatenate([pts4[:, 0], -pts4[:, 1], pts4[:, 2]], axis=1)
    


    # Edge 0-1 and 12-13
    pts9 = np.array([-H*cos(pi/4) + Xu[C_max_idx, 0], W, H*sin(pi/4)])
    pts11 = np.array([pts9[0], -pts9[1], pts9[2]])

    # Edge 0-9 and 12-21
    pts10 = np.array([-H*cos(pi/4) + Xu[C_max_idx, 0], W, -H*sin(pi/4)])
    pts12 = np.array([pts10[0], -pts10[1], pts10[2]])


    ax.plot(np.array(pts1)[:,0], np.array(pts1)[:,2], '--r')
    ax.plot(np.array(pts5)[:,0], np.array(pts5)[:,2], '--b')                       
    ax.plot(np.array(pts2)[:,0], np.array(pts2)[:,2], '--r')
    ax.plot(np.array(pts6)[:,0], np.array(pts6)[:,2], '--b')
    ax.plot(np.array(pts3)[:,0], np.array(pts3)[:,2], '--r')
    ax.plot(np.array(pts7)[:,0], np.array(pts7)[:,2], '--b')
    ax.plot(np.array(pts4)[:,0], np.array(pts4)[:,2], '--r')
    ax.plot(np.array(pts8)[:,0], np.array(pts8)[:,2], '--b')

    ax.plot(np.array(pts9)[0], np.array(pts9)[2], '*r')
    ax.plot(np.array(pts11)[0], np.array(pts11)[2], 'ob')

    ax.plot(np.array(pts10)[0], np.array(pts10)[2], '*r')
    ax.plot(np.array(pts12)[0], np.array(pts12)[2], 'ob')

    # Calculate number of mesh points along 4-5 and 4-6
    #N_leading = (C_max_idx/Ni)*Nx
    N_leading = int((x[C_max_idx]/c)*Nx)

    # Calculate number of mesh points along 5-7 and 6-7
    N_trailing = Nx - N_leading

    # Open file
    print(os.getcwd())
    f = open("tests/CFD/openfoam_sim/system/blockMeshDict", "w")


    # Write file
    f.write("/*--------------------------------*- C++ -*----------------------------------*\\ \n")
    f.write("| =========                 |                                                 | \n")
    f.write("| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           | \n")
    f.write("|  \\\\    /   O peration     | Version:  3.0.x                                 | \n")
    f.write("|   \\\\  /    A nd           | Web:      www.OpenFOAM.com                      | \n")
    f.write("|    \\\\/     M anipulation  |                                                 | \n")
    f.write("\\*---------------------------------------------------------------------------*/ \n")
    f.write("FoamFile                                                                        \n")
    f.write("{                                                                               \n")
    f.write("    version     2.0;                                                            \n")
    f.write("    format      ascii;                                                          \n")
    f.write("    class       dictionary;                                                     \n")
    f.write("    object      blockMeshDict;                                                  \n")
    f.write("}                                                                               \n")
    f.write("// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * // \n")
    f.write("\n")
    f.write("convertToMeters %f; \n" % scale)
    f.write("\n")
    f.write("vertices \n")
    f.write("( \n")
    for vertex in vertices:
        f.write("    (%f %f %f)\n" % tuple(vertex))
    f.write("); \n")
    f.write("\n")
    f.write("blocks \n")
    f.write("( \n")
    f.write("    hex (4 5 1 0 16 17 13 12)     (%i %i %i) edgeGrading (1 %f %f 1 %f %f %f %f 1 1 1 1) \n" % (N_leading, NT, NW, 1/ExpArc, 1/ExpArc, ExpT, ExpT, ExpT, ExpT))
    f.write("    hex (5 7 2 1 17 19 14 13)     (%i %i %i) simpleGrading (1 %f 1) \n" % (N_trailing, NT, NW, ExpT))
    f.write("    hex (7 8 3 2 19 20 15 14)     (%i %i %i) simpleGrading (%f %f 1) \n" % (ND, NT, NW, ExpD, ExpT))
    f.write("    hex (16 18 21 12 4 6 9 0)     (%i %i %i) edgeGrading (1 %f %f 1 %f %f %f %f 1 1 1 1) \n" % (N_leading, NT, NW, 1/ExpArc, 1/ExpArc, ExpT, ExpT, ExpT, ExpT))
    f.write("    hex (18 19 22 21 6 7 10 9)    (%i %i %i) simpleGrading (1 %f 1) \n" % (N_trailing, NT, NW, ExpT))
    f.write("    hex (19 20 23 22 7 8 11 10)   (%i %i %i) simpleGrading (%f %f 1) \n" % (ND, NT, NW, ExpD, ExpT))

    f.write("); \n")
    f.write("\n")
    f.write("edges \n")
    f.write("( \n")

    f.write("    spline 4 5 \n")
    f.write("        ( \n")
    for pt in np.array(pts1):
        f.write("            (%f %f %f) \n" % tuple(pt))
    f.write("        ) \n")

    f.write("    spline 5 7 \n")
    f.write("        ( \n")
    for pt in np.array(pts2):
        f.write("            (%f %f %f)\n" % tuple(pt))
    f.write("        ) \n")

    f.write("    spline 4 6 \n")
    f.write("        ( \n")
    for pt in np.array(pts3):
        f.write("            (%f %f %f)\n" % tuple(pt))
    f.write("        ) \n")

    f.write("    spline 6 7 \n")
    f.write("        ( \n")
    for pt in np.array(pts4):
        f.write("            (%f %f %f)\n" % tuple(pt))
    f.write("        ) \n")

    f.write("    spline 16 17 \n")
    f.write("        ( \n")
    for pt in np.array(pts5):
        f.write("            (%f %f %f)\n" % tuple(pt))
    f.write("        ) \n")

    f.write("    spline 17 19 \n")
    f.write("        ( \n")
    for pt in np.array(pts6):
        f.write("            (%f %f %f)\n" % tuple(pt))
    f.write("        ) \n")

    f.write("    spline 16 18 \n")
    f.write("        ( \n")
    for pt in np.array(pts7):
        f.write("            (%f %f %f)\n" % tuple(pt))
    f.write("        ) \n")

    f.write("    spline 18 19 \n")
    f.write("        ( \n")
    for pt in np.array(pts8):
        f.write("            (%f %f %f)\n" % tuple(pt))
    f.write("        ) \n")

    f.write("    arc 0 1 (%f %f %f) \n" % tuple(pts9))
    f.write("    arc 0 9 (%f %f %f) \n" % tuple(pts10))
    f.write("    arc 12 13 (%f %f %f) \n" % tuple(pts11))
    f.write("    arc 12 21 (%f %f %f) \n" % tuple(pts12))

    f.write("); \n")
    f.write("\n")
    f.write("boundary \n")
    f.write("( \n")

    f.write("    inlet \n")
    f.write("    { \n")
    f.write("        type patch; \n")
    f.write("        faces \n")
    f.write("        ( \n")
    f.write("            (1 0 12 13) \n")
    f.write("            (0 9 21 12) \n")
    f.write("        ); \n")
    f.write("    } \n")
    f.write("\n")

    f.write("    outlet \n")
    f.write("    { \n")
    f.write("        type patch; \n")
    f.write("        faces \n")
    f.write("        ( \n")
    f.write("            (11 8 20 23) \n")
    f.write("            (8 3 15 20) \n")
    f.write("        ); \n")
    f.write("    } \n")
    f.write("\n")

    f.write("    topAndBottom \n")
    f.write("    { \n")
    f.write("        type patch; \n")
    f.write("        faces \n")
    f.write("        ( \n")
    f.write("            (3 2 14 15) \n")
    f.write("            (2 1 13 14) \n")
    f.write("            (9 10 22 21) \n")
    f.write("            (10 11 23 22) \n")
    f.write("        ); \n")
    f.write("    } \n")
    f.write("\n")

    f.write("    airfoil \n")
    f.write("    { \n")
    f.write("        type wall; \n")
    f.write("        faces \n")
    f.write("        ( \n")
    f.write("            (5 4 16 17) \n")
    f.write("            (7 5 17 19) \n")
    f.write("            (4 6 18 16) \n")
    f.write("            (6 7 19 18) \n")
    f.write("        ); \n")
    f.write("    } \n")
    f.write("); \n")
    f.write(" \n")
    f.write("mergePatchPairs \n")
    f.write("( \n")
    f.write("); \n")
    f.write(" \n")
    f.write("// ************************************************************************* // \n")

    # Close file
    f.close()

  def gmsh_generator(self, path:str):
    """ This is the generator of the 2D spatial domain mesh from the airfoil 2d Cartesian points"""
    #COMPLETE: Tested and ready for use
    file = open(os.path.join(os.getcwd(), path + '/airfoil.geo'), 'w')
    for i in range(len(self.xt)):

        point = 'Point(' + str(i+1) + ') = '
        coords = '{' + str(self.xt[len(self.xt)-i-1]) + ', ' + str(self.yt[i]) + ', ' + '0};' + '\n'

        file.write(point+coords)

    for i in range(len(self.xb)):

        point = 'Point(' + str(i+1+len(self.xt)) + ') = '
        coords = '{' + str(self.xb[i]) + ', ' + str(self.yb[i]) + ', ' + '0};' + '\n'

        file.write(point+coords)

    Line = 'Line(1) = {'

    for i in range(len(self.xt) + len(self.xb)):
        if not(i == len(self.xt) + len(self.xb) - 1):
            Line = Line + str(i+1) + ', '
        else:
            Line = Line + str(i+1) + ', 1};'

    file.write(Line)
    file.close()

    old = os.getcwd()
    os.chdir(os.path.join(old, path))
    cmd = f'gmsh -3 {os.path.join(old, path)}/meshBlocks_2d.geo -o {os.path.join(old, path)}/airfoil.su2'
    p = subprocess.run(cmd, shell=True) #Popen(cmd, stdin=PIPE, bufsize=0)
    os.chdir(old)
    if p.returncode != 0:
      raise CalledProcessError(p.returncode, cmd)    

  def create_openfoam_sim_files(self, alpha_deg=4):
    """"""
    self.gen_OF_blockmeshdict_from_coords(alpha_deg)
    

  def plot_cp(self):
    plt.plot(self.cpx, self.cpl)
    plt.plot(self.cpx, self.cpu)
    plt.title('Cp')
    plt.xlim([0,1])
    plt.show()

  def load_force_coeffs_single(self, time_dir):
    """Load force coefficients into a DataFrame for the given time directory."""
    fpath = "postProcessing/forceCoeffs/{}/forceCoeffs.dat".format(time_dir)
    data = np.loadtxt(fpath, skiprows=9)
    df = pd.DataFrame()
    df["time"] = data[:, 0]
    df["cl"] = data[:, 3]
    df["cd"] = data[:, 2]
    df["cm"] = data[:, 1]
    return df


  def load_force_coeffs(self, steady=False):
      """Load force coefficients from file.

      If steady, the file from the `0` directory is used, and the last values are
      returned. Otherwise, arrays are loaded from the latest file.
      """
      if steady:
          return self.load_force_coeffs_single("0")
      else:
          df = pd.DataFrame()
          time_dirs = sorted(os.listdir("postProcessing/forceCoeffs"))
          for t in time_dirs:
              df = df.append(self.load_force_coeffs_single(t), ignore_index=True)
      return df.sort_values(by="time")

  def runSU2CFD(self, path):
    old = os.getcwd()
    os.chdir(os.path.join(old, path))
    cmd = f'SU2_CFD airfoil.cfg'
    p = subprocess.run(cmd, shell=True) #Popen(cmd, stdin=PIPE, bufsize=0)
    if p.returncode != 0:
      raise CalledProcessError(p.returncode, cmd)  
    os.chdir(old)  



if __name__ == "__main__":
  # COMPLETE: Ready for use
  design = airfoil_generator(tegap=0.001263,
  degree_t=4,
  degree_b=4,
  chord = 1,
  nPts = 3,
  #        [yt1,  xt2, yt2,  xt3,yt3,    yb1,xb2,     yb2,xb3,    yb3  ]
  params = [0.05,  0.3,0.1,  0.6,0.1,  -0.05,0.35,   0.02,0.5,   -0.004]
  )


  design.generate()
  # design.visualize()
  design.gmsh_generator(os.path.abspath('tests/CFD/SU2'))
  design.runSU2CFD(os.path.abspath('tests/CFD/SU2'))
  design.get_responses(os.path.abspath('tests/CFD/SU2/history.csv'), 'SU2')
  # design.create_TSFOIL_sim_files()
  # design.create_openfoam_sim_files(4)
  # design.run()
  # design.get_responses(f'Design{id}.out', 'TSFOIL')
  # design.plot_cp()
