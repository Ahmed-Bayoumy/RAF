# ------------------------------------------------------------------------------------#
#  Relative adequacy framework - RAF                                      #
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
#  https://github.com/Ahmed-Bayoumy/RAF                                               #
# ------------------------------------------------------------------------------------#

from ast import Param
import copy
import csv
from genericpath import isfile
from gettext import find
import json
from multiprocessing.dummy import Process
from multiprocessing.sharedctypes import Value

from dataclasses import dataclass, field
import os
from sys import argv
import sys
from tkinter.messagebox import NO
from typing import List, Dict, Any, Callable, Protocol, Optional

import numpy as np
from numpy import cos, exp, pi, prod, sin, sqrt, subtract, inf

from OMADS import Point, PreMADS, PostMADS, OrthoMesh, Parameters,  Evaluator, Cache, Dirs2n, Options
import OMADS
from enum import Enum, auto
import matplotlib.pyplot as plt

import SML


import time

from DMDO import DA, DA_Data

class ST(Enum):
  KRIGING: int = auto()
  RBF: auto()

@dataclass
class inadequacy_data:
  """ """
  
@dataclass
class DFTRM:
  Delta: List[float] = None
  Delta_AA: float = 1.
  R1: float = 0.01
  R2: float = 0.9
  c1: float = 0.5
  c2: float = 1.5
  rho: float = np.inf
  nsamples: int = 10
  params: Param = None
  xold: Point = None
  xold_hat: Point = None

  def constructTheModel(self):
    """ Construct the model """
    v = np.array([[-2.0, 2.0], [-2.0, 2.0]])
    n = 50

    sampling = SML.FullFactorial(ns=n, vlim=v, w=None, c=False)

    xt = sampling.generate_samples()
    yt = self.RB(xt, None)

    opts: Dict = {}
    # opts["weights"] = np.empty((1, 1))

    # sm = RBF(type="train", x=xt, y=yt, options=opts, rbf_func="cubic")
    sm = SML.Kriging(type="train", x=xt, y=yt, options=opts)
    sampling = SML.LHS(ns=n, vlim=v)
    sampling.options["criterion"] = "ExactSE"
    xp = sampling.generate_samples()
    sm.xt.points = sm.x.points
    sm.x.points = xp
    sm.predict()

  def solveSubProblem(self):
    """ TODO: to be integrated with OMADS """
  
  def calculateStep(self):
    """ TODO: to be integrated with OMADS """
  
  def updateRadius(self, xnew: Point, xnew_hat: Point, index: int):
    """  """
    if abs(self.xold_hat.f-xnew_hat.f) != np.inf and abs(self.xold_hat.f-xnew_hat.f) != 0.:
      self.rho = (self.xold.f-xnew.f) / (self.xold_hat.f-xnew_hat.f)
    elif abs(self.xold_hat.f-xnew_hat.f) == 0.:
      self.rho = np.inf * (self.xold.f-xnew.f)
    else:
      self.rho = 0.
    if self.rho < self.R1:
      self.Delta[index] *= self.c1
    elif self.rho > self.R2:
      self.Delta[index] *= self.c2

  def updateRadius_AA(self, xnew: Point, xnew_hat: Point):
    """  """
    if abs(self.xold_hat.f-xnew_hat.f) != np.inf and abs(self.xold_hat.f-xnew_hat.f) != 0.:
      self.rho = (self.xold.f-xnew.f) / (self.xold_hat.f-xnew_hat.f)
    elif abs(self.xold_hat.f-xnew_hat.f) == 0.:
      self.rho = np.inf * (self.xold.f-xnew.f)
    else:
      self.rho = 0.
    if self.rho < self.R1:
      self.Delta_AA *= self.c1
    elif self.rho > self.R2:
      self.Delta_AA *= self.c2
    # self.xold = copy.deepcopy(xnew)
    # self.xold_hat = copy.deepcopy(xnew_hat)

    
#  TODO: add a protocole object for models
@dataclass
class model_data:
  nsamples: int = 10
  nevals: int = 0
  rank: int = 0
  cache: Cache = Cache()
  bb: Callable = None
  xmin: Point = None
  err: float = 0.
  rtime: int = 0.
  AA : np.ndarray = np.empty((0,0))
  RA : np.ndarray = np.empty((0,0))
  AA_old : np.ndarray = np.empty((1,1))
  RA_old : np.ndarray = np.empty((1,1))
  xt: List[Point] = None
  nvars: int = 0
  nt: int = 0
  sm_opts: Dict = field(default=dict)
  sm_RA: List[SML.RBF] = None
  sm_AA: SML.RBF = None
  Lambda: float = np.inf
  Delta: float = 0.5
  TRM: DFTRM = None
  RA_min: np.ndarray = np.empty((1,1))
  AA_min: float = 0.
  RA_min_old: np.ndarray = np.empty((1,1))
  AA_min_old: float = 0.
  Lambda_AA: float = 0.
  

@dataclass
class model(model_data):
  def evaluate_samples(self, xs: List[Point]):
    self.nt += len(xs)
    self.nvars = xs[0].n_dimensions
    for i in range(len(xs)):
      if not self.cache.is_duplicate(xs[i]):
        bbout = self.bb(xs[i].coordinates)
        xs[i].__eval__(bbout)
        xs[i].f += self.err + xs[i].h
        self.cache.add_to_cache(xs[i])
        self.nevals += 1
        if self.xmin is None:
          self.xmin = copy.deepcopy(xs[i])
        else:
          if xs[i] < self.xmin:
            self.xmin = copy.deepcopy(xs[i])
    if self.xt is None:
      self.xt = copy.deepcopy(xs)
    else:
      for e in xs: self.xt.append(e)
    return xs
  
  def build_surrogate_errors(self, x: List[Point]):
    """ """
    if not isinstance(self.sm_RA, list): 
      self.sm_RA = []
      xtemp: np.ndarray = np.zeros((len(x), self.nvars))
      # Populate the coordinates of the sampling points in a temporary array
      for i in range(len(x)):
        xtemp[i] = np.array(x[i].coordinates)
      
      # Loop over the number of relative adequacy vectors measured between this model and other avialable models
      for i in range(self.RA.shape[0]):
        # Avoid building adequacy surrogates for zeros vector (Typically that vector has the current model index)
        if not np.all(self.RA[i] == 0.):
          self.sm_opts = {}
          self.sm_opts["weights"] = np.empty((1, 1))
          self.sm_RA.append(SML.RBF(type="train", x=xtemp, y=self.RA[i], options=self.sm_opts, rbf_func="cubic"))
        else:
          self.sm_RA.append(None)
      
      self.sm_AA = SML.RBF(type="train", x=xtemp, y=self.AA, options=self.sm_opts, rbf_func="cubic")
    else:
      xtemp: np.ndarray = np.zeros((len(x), self.nvars))
      # Populate the coordinates of the sampling points in a temporary array
      for i in range(len(x)):
        xtemp[i] = np.array(x[i].coordinates)
      
      # Loop over the number of relative adequacy vectors measured between this model and other avialable models
      for i in range(self.RA.shape[0]):
        # Avoid building adequacy surrogates for zeros vector (Typically that vector has the current model index)
        if not np.all(self.RA[i] == 0.):
          self.sm_opts = {}
          self.sm_opts["weights"] = np.empty((1, 1))
          self.sm_RA[i].addToTrainingSet(xtemp, self.RA[i]) 
          self.sm_RA[i].train()
      self.sm_AA.addToTrainingSet(xtemp, self.AA)
      self.sm_AA.train()

    
  def predict_RA(self, x: List[float], mindex: int):
    X= np.array(x)
    
    self.sm_RA[mindex].x.points = np.append([X], self.sm_RA[mindex].xt.points, axis=0)
    self.sm_RA[mindex].predict()
    return (self.sm_RA[mindex].yp.points[0]).real
  
  def predict_AA(self, x: List[float]):
    X= np.array(x)
    self.sm_AA.x.points = np.append([X], self.sm_AA.xt.points, axis=0)
    self.sm_AA.predict()
    return (self.sm_AA.yp.points[0]).real
  
  def evaluate_point(self, x: Point):
    bbout = self.bb(x.coordinates)
    x.__eval__(bbout)
    self.nevals += 1
    x.f += self.err + x.h
    self.nevals += 1
    self.cache.add_to_cache(x)
    if self.xmin is None:
        self.xmin = copy.deepcopy(x)
    else:
      if x < self.xmin:
        self.xmin = copy.deepcopy(x)
    
    return x
  
  def calculate_error(self, x: Point) -> float:
    xtemp: Point = Point()
    xtemp.coordinates = copy.deepcopy(x.coordinates)
    if self.cache.is_duplicate(xtemp):
      return 0.
    else:
      xtemp = self.evaluate_point(xtemp)
      return abs(xtemp.__df__(x)) + abs(xtemp.__dh__(x))

  def rectify(self, x: Point, err: float, r):
    if self.cache.is_duplicate(x):
      k = self.cache.get_index(x)
      xk = self.cache.cache_dict[self.cache.hash_id[k]]
      xk.f += err + xk.h
      return xk
    else:
      x.f += err + x.h + self.Lambda[r]
      if x < self.xmin:
        self.xmin = copy.deepcopy(x)
      return x


def rosen(x, *argv):
    x = np.asarray(x)
    y = [np.sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0,
    axis=0), [0]]
    time.sleep(0.05)
    return y

def rosen1(x, *argv):
  x = np.asarray(x)
  y = [np.sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0,
  axis=0)+ 0.5 * np.sum(np.sin(x)), [0]]
  time.sleep(0.01)
  return y

def rosen2(x, *argv):
  x = np.asarray(x)
  y = [np.sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0, 
  axis=0) + np.sum(np.cos(x)), [0]]
  time.sleep(0.005)
  return y

def rosen3(x):
  x = np.asarray(x)
  y = [np.sum(np.add(np.sin(x), 100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0), axis=0), [0]]
  time.sleep(0.001)
  return y  


def sphere1(x):
  time.sleep(0.3)
  return [sum(np.power(x, 2)), [0.0]]

def sphere2(x):
  time.sleep(0.2)
  return [sum(np.power(x, 2))+0.5, [0.0]]

def sphere3(x):
  time.sleep(0.1)
  return [sum(np.power(x, 2))+3., [0.0]]

@dataclass
class RAF_p:
  bb_models: List[str] = field(default_factory=list)
  max_it: int = 10
  tol: float = 1e-9
  levels: List[int] = field(default_factory=list)
  neval_total: int = 1000
  display_overview: bool = False
@dataclass
class optimizer:
  optimizer: str = "OMADS"
  seed: int = 0
  budget: int = 1000
  tol: float = 1e-9
  psize_init: float = 1.0
  display: bool = False
  opportunistic: bool = False
  check_cache: bool = False
  store_cache: bool = False
  collect_y: bool = False
  rich_direction: bool = False
  precision: str = "high"
  save_results: bool = False
  save_coordinates: bool = False
  # save_all_best can take the following values:
  # false for saving all designs in the MADS.out file
  # true for saving best designs only
  save_all_best: bool = False
  parallel_mode: bool = False

@dataclass
class RAF_data:
  RAF_params: RAF_p
  optimizer_param: optimizer
  prob_params: Parameters 
  models: List[model] = model
  rtime: np.ndarray = None
  nvars: int = None
  sampling_t: int = SML.SAMPLING_METHOD.LH
  ns: int = 1
  xmin: Point = None
  hashtable: Cache = None
  # max_it: int = 100
  # tol: float = 1e-12
  # psize: float = 1.
  Delta: np.ndarray = np.empty((1,1))
  Lambda: np.ndarray = np.empty((1,1))
  TRM: DFTRM = None
  nm: int = 1
  AAM: np.ndarray = field(default=np.empty((1)))
  RAM: np.ndarray = field(default=np.empty((1)))
  # levels: np.ndarray = field(default=np.empty((1)))
  poll_success: bool = False
  terminate: bool = False
  iter: int = 0
  neval: int = 0
  # eval_budget: int = 100 
  switch_to_search: bool = False
  seed: int = 0
  display_poll: bool = False
  RAM_old: np.ndarray = np.empty((0,0))
  xs_hist: List[List] = None
  Delta_init: List = None
  retrain_surrogates: bool = False
  offLineRunTime: float = 0.
  onLineRunTime: float = 0.
  totalRunTime: float = 0.
  opportunistic: bool = False
  vicinity_ratio: np.ndarray = field(default=np.ones((1)))
  succ_search: bool = False

@dataclass
class DOE_RAF():
  """ Using RAF to manage the use of models during a DOE study """

@dataclass
class OMADS_RAF(RAF_data):
  
  def generate_sample_points(self, nsamples: int = None) -> List[Point]:
    """ Generate the sample points """
    xlim = []
    self.nvars = len(self.prob_params.baseline)
    v = np.empty((self.nvars, 2))
    if self.xmin and self.iter > 1:
      for i in range(len(self.prob_params.lb)):
        lb = copy.deepcopy(self.xmin.coordinates[i]-abs(self.xmin.coordinates[i] * self.vicinity_ratio[i]))
        ub = copy.deepcopy(self.xmin.coordinates[i]+abs(self.xmin.coordinates[i] * self.vicinity_ratio[i]))
        if lb <= self.prob_params.lb[i]:
          lb = copy.deepcopy(self.prob_params.lb[i])
        elif  lb >= self.prob_params.ub[i]:
          lb = self.xmin.coordinates[i]
        if ub >= self.prob_params.ub[i]:
          ub = copy.deepcopy(self.prob_params.ub[i])
        elif ub <= self.prob_params.lb[i]:
          ub = self.xmin.coordinates[i]
        v[i] = [lb, ub]
    else:
      for i in range(len(self.prob_params.lb)):
        lb = copy.deepcopy(self.prob_params.lb[i])
        ub = copy.deepcopy(self.prob_params.ub[i])
        v[i] = [lb, ub]
    if nsamples is None:
      nsamples = int((self.nvars+1)*(self.nvars+2)/2)
    
    self.ns = nsamples

    if self.sampling_t == SML.SAMPLING_METHOD.FULLFACTORIAL:
      sampling = SML.FullFactorial(ns=nsamples, vlim=v, w=np.array([0.1, 0.05]), c=True)
    elif self.sampling_t == SML.SAMPLING_METHOD.LH: 
      sampling = SML.LHS(ns=nsamples, vlim=v)
      self.seed += np.random.randint(0, 10000)
      sampling.options["randomness"] = self.seed
      sampling.options["criterion"] = "center"
    elif self.sampling_t == SML.SAMPLING_METHOD.RS:
      sampling = SML.RS(ns=nsamples, vlim=v)
    elif self.sampling_t == SML.SAMPLING_METHOD.HALTON:
      sampling = SML.halton(ns=nsamples, vlim=v, is_ham=True)
    
    Ps= copy.deepcopy(sampling.generate_samples())
    # self.visualize_samples(Ps[:, 0], Ps[:, 1])
    # if self.xmin is not None:
    #   self.visualize_samples(self.xmin.coordinates[0], self.xmin.coordinates[1])
    return self.map_samples_from_coords_to_points(Ps)
  
  def map_samples_from_coords_to_points(self, samples: np.ndarray) -> List[Point]:
    x: List[Point] = [0.] *self.ns
    for i in range(samples.shape[0]):
      x[i] = Point()
      x[i].coordinates = copy.deepcopy(samples[i])
    return x
      
  def evaluate_phi(self, points: List[Point]):
    """ """
    self.rtime = np.empty((self.nm))
    xs: List[Point] = [Point()] * self.nm
    
    for i in range(self.nm):
      tic = time.perf_counter()
      xs[i] = copy.deepcopy(points)
      xs[i] = copy.deepcopy(self.models[i].evaluate_samples(points))
      if i == 0:
        if self.models[i].xmin < self.xmin:
          self.xmin = copy.deepcopy(self.models[i].xmin)
          self.vicinity_ratio = np.multiply(self.vicinity_ratio, 2.0)
          self.succ_search = True
        else:
          self.vicinity_ratio = np.multiply(self.vicinity_ratio, 0.5)
          self.succ_search = False
      toc = time.perf_counter()
      self.rtime[i] = toc-tic
    return xs
  
  def evaluate_phi_using_model_i(self, points: List[Point], i):
    """ """
    xs = [1]
    tic = time.perf_counter()
    xs = copy.deepcopy(points)
    xs = copy.deepcopy(self.models[i].evaluate_samples(points))
    toc = time.perf_counter()
    self.rtime = toc-tic
    return xs
  
  def visualize_samples(self, x, y):
    


    
    # xx = np.linspace(-2, 2, 300)
    # yy = np.linspace(-2, 2, 300)
    # X, Y = np.meshgrid(xx, yy)
    # if self.iter == 0:
    #   r = lambda xx,yy: 100.0 * (yy - xx ** 2.0) ** 2.0 + (1 - xx) ** 2.0#abs(x*np.sin(x)+0.1*x)+abs(y*np.sin(y)+0.1*y)
      

    #   im = plt.contourf(X, Y, r(X, Y), 500, cmap='viridis_r')
    #   plt.contour(X, Y, r(X, Y), 100, colors=['#F8F8FF'], linewidths=0.3)

    if isinstance(x, np.ndarray):
      plt.scatter(x, y)
      plt.grid(color='b', linestyle='--', linewidth=0.5)
      # plt.show()
    else:
      plt.scatter(x, y, color='k')
      plt.grid(color='b', linestyle='--', linewidth=0.5)
      # plt.show()
    
    # plt.colorbar(im, label="f", orientation="vertical")
    plt.ion()
    plt.show()
    plt.pause(0.1)

    # if self.iter == 0:
    #   plt.ion()
    #   self.figure, self.ax = plt.subplots(figsize=(10, 8))
    #   self.line1, = self.ax.plot(x, y, 'o')
    #   plt.grid(color='b', linestyle='--', linewidth=0.5)
    # else:
    #   self.line1, = self.ax.plot(x, y, 'o')
    #   self.figure.canvas.draw()
    #   self.figure.canvas.flush_events()

  def search_step(self):
    """ """
    if self.iter > 0:
      S: List[Point] = self.generate_sample_points()
      for i in range(self.ns):
        xtemp = copy.deepcopy(self.predict_RAM(S[i]))
        if not self.hashtable.is_duplicate(S[i]):
          self.hashtable.add_to_cache(xtemp)
        # self.calculate_Lambda()
        if self.stop_search or xtemp<self.xmin:
          self.xmin = copy.deepcopy(xtemp)
          self.succ_search = True
          if self.opportunistic:
            break
      
      
        
      
      if self.retrain_surrogates:
        # Evaluate the initial penalty function 
        xs = self.evaluate_phi(S)
        for e in xs: self.xs_hist.append(e)
        # Calculate absolute adequacies: discrepancies between models output and the reference model
        self.calc_absolute_adeq(self.xs_hist)
        # Calculate relative adequacies: discrepancies between models output and the reference model
        self.calc_relative_adeq(self.xs_hist)
        # Build error surrogates
        self.build_Adeq_surrogates()
        # Update adequacies at the current minimizer
        self.update_adeq_at_minimizer()
      
      if self.succ_search:
        self.vicinity_ratio = np.multiply(self.vicinity_ratio, 2.0)
      else:
        self.vicinity_ratio = np.multiply(self.vicinity_ratio, 0.5)
      
      if self.RAF_params.display_overview:
        print(f'iteration= {self.iter}, step_name= search_step, is_success={self.succ_search}, xmin= {self.xmin.coordinates}, bbeval= {self.ns}, psize_init=  {self.optimizer_param.psize_init}, fmin= {self.xmin.f}, hmin= {self.xmin.h}')
        if self.mToUpdate.size != 0 and self.mToRefer.size != 0:
          print(f'Updated models= {self.mToUpdate}, Referral models= {self.mToRefer}')
      
      
  
  def predict_RAM(self, x: Point):
    """ Predicting relative adequacies """
    xout: Point
    self.stop_search = False
    if not isinstance(x, list):
      self.RAM: np.ndarray = np.zeros((self.nm-1, self.nm-1))
      if self.RAM_old.size ==0:
        self.RAM_old = np.zeros((self.nm-1, self.nm-1))
      # Array of models that have predicted adequacies that lie outside the trust region
      # Lambda > Delta
      # The inadequacy surrogate of these models has to be retrained
      self.mToUpdate = np.zeros((0,2))
      # models that we need to relatively refer to when inadequacy surrogates of other models have to be retrained
      self.mToRefer = np.zeros((0,2))
      # Loop over the list of models to create/update the RAM (a square-matrix)
      breakLoop = False
      for i in range(self.nm):
        for j in range(self.nm):
          # Ignore the diagonal elements as the model refers to itself there
          # Ignore the zero index which refer to the absolute reference model
          if i > 0 and j > 0 and self.models[i].sm_RA[j-1] != None:
            # Predicted relative adequacy with model j
            RA_temp = self.models[i].predict_RA(x.coordinates, j-1)
            # Update RAM
            self.RAM[i-1, j-1] = copy.deepcopy(RA_temp)
            self.RAM_old[i-1, j-1] = copy.deepcopy(self.RAM[i-1, j-1])
            # Update the relative adequacy vector of this model at the ks sample point
            self.models[i].RA_min[j-1] = copy.deepcopy(RA_temp)
            # Update Lambda of this model
            self.models[i].Lambda[j-1] = np.abs(np.subtract(self.models[i].RA_min[j-1], self.models[i].RA_min_old[j-1]))
            # Check if Lambda is within the trust region of this model
            if self.models[i].Lambda[j-1] > self.models[i].TRM.Delta[j-1]:
              # If it lies outside the trust region, then this model needs to be updated
              if self.mToUpdate.size == 0:
                self.mToUpdate = np.asarray([[i, j]])
              else:
                self.mToUpdate = np.append(self.mToUpdate, [[i, j]], axis=0)
            else:
              # Otherwise we can refer to it when updating/correcting other models that have less accuracy than that one
              if self.mToRefer.size == 0:
                self.mToRefer = np.asarray([[i, j]])
              else:
                self.mToRefer = np.append(self.mToRefer, [[i, j]], axis=0)
            # xh = self.models[j].evaluate_point(copy.deepcopy(x))
            # xl = self.models[i].evaluate_point(copy.deepcopy(x))
            # self.models[i].TRM.updateRadius(xh, xl, j-1)
            if breakLoop is True:
              break
          if breakLoop is True:
            break

      # If we have models that we need to retrain their errors surroagte, then sort them based on their runtime
      if self.mToUpdate.size != 0:
        sorted_mToUpdate = self.mToUpdate[self.mToUpdate[:,0].argsort()]
      else:
        # Otherwise, use the last model in the model list which has the least computational cost and most inferior outputs, yet can be rectified using trusted predictions of its relative adequacies of other models
        for r in range(len(self.models[-1].RA)):
          if np.sum(self.models[-1].RA[r]) != 0 and self.models[-1].xmin.f != np.inf and self.models[-1].Lambda[r] != np.inf:
            bbout = self.models[-1].bb(x.coordinates)
            x.__eval__(bbout)
            self.switch_to_search = False
            out: Point = self.models[-1].rectify(x, self.models[-1].RA_min[r], r)
            if out < self.models[-1].xmin: 
              self.models[-1].xmin = copy.deepcopy(self.models[-1].rectify(x, self.models[-1].RA_min[r], r))
              self.stop_search = True
            return out
      # If we have models to be used as reference for retraining the error surrogates associated with other models that have more inferior predictability than the reference ones
      if self.mToRefer.size != 0:
        sorted_mToRefer = self.mToRefer[self.mToRefer[:,0].argsort()]
        self.retrain_surrogates: bool = True
      else:
        # If we couldn't find models that we can trust for retraining the error surrogates of other models, then try with predicted absolute adequacies
        for s in range(sorted_mToUpdate.shape[0]):
          k = sorted_mToUpdate[s,1]
          self.models[k].AA_min = self.models[k].predict_AA(x.coordinates)
          self.models[k].Lambda_AA = np.abs(np.subtract(self.models[k].AA_min, self.models[k].AA_min_old))
          
          
          # If approximated absolute adequacies can be trusted, use them, if not we need to reevaluate all models so we can update their adequacy surrogates
          if self.models[k].Lambda_AA > self.models[k].TRM.Delta_AA:
            continue
          else:
            # Evaluate the point using the reference model
            if not x.evaluated:
              x = self.models[0].evaluate_point(x)
            
            xm: List[Point] = copy.deepcopy(self.evaluate_phi_using_model_i([x], k))
            self.models[k].AA = copy.deepcopy(x.f-xm[0].f)
            self.evaluate_phi([x])
            # xh = self.models[0].evaluate_point(copy.deepcopy(x))
            # xl = self.models[k].evaluate_point(copy.deepcopy(x))
            # self.models[i].TRM.updateRadius_AA(xh, xl)
      
      if self.mToRefer.size == 0:
        self.retrain_surrogates: bool = True
        self.switch_to_search = True
        self.stop_search = False
        return x
      # Evaluate all models, including reference model, to train/retrain adequacy surroagtes
      for m in range(sorted_mToUpdate.shape[0]):
        xm: Point = self.models[sorted_mToUpdate[m,1]].evaluate_point(copy.deepcopy(x))
        mh = sorted_mToRefer[0, 1]
        
        # x = self.models[mh].evaluate_point(copy.deepcopy(x))
        
        self.models[sorted_mToUpdate[m,1]].RA_min[mh-1] = x.f-xm.f
        self.models[sorted_mToUpdate[m,1]].Lambda[mh-1] = np.abs(np.subtract(self.models[sorted_mToUpdate[m,1]].RA_min[mh-1], self.models[sorted_mToUpdate[m,1]].RA_min_old[mh-1]))

      self.switch_to_search = True
      out: Point = self.models[sorted_mToUpdate[m,1]].evaluate_point(x)
      if out < self.models[sorted_mToUpdate[m,1]].xmin: 
        self.models[sorted_mToUpdate[m,1]].xmin = copy.deepcopy(out)
        self.stop_search = True
      return out



  def calc_absolute_adeq(self, xs: List[List[Point]]):
    """  """
    
    if isinstance(xs, np.ndarray) or isinstance(xs, list):
      for i in range(1, self.nm):
        self.models[i].AA = np.empty(len(xs[i]))
        for j in range(len(xs[i])):
          if isinstance(xs[i], np.ndarray) or isinstance(xs[i], list):
            self.models[i].AA[j] = abs((xs[0][j].f)-(xs[i][j].f))
          elif self.models[i].AA:
            self.models[i].AA[j] = abs((xs[0][j].f)-(xs[i][j].f))
          else:
            raise Exception(RuntimeError, "Absolute adequacy matrix should be initialized before dumping errors in it!")
          if self.models[i].sm_AA and self.models[i].sm_AA.xt.is_initialized():
            self.models[i].sm_AA.addToTrainingSet(np.array([xs[0][j].coordinates]), np.array([self.models[i].AA[j]]))
    else:
      raise Exception(RuntimeError, "evaluate_absolute_inadeq was called for evaluating a non-list of points!") 
  def calc_relative_adeq(self, xs: List[Point]):
    """  """
    if isinstance(xs, np.ndarray) or isinstance(xs, list):
      for i in range(self.nm):
        if i > 0:
          self.models[i].RA = np.zeros((self.nm-1, len(xs[i])))
          for j in range(self.nm):
            if j > 0 and i != j:
              for k in range(len(xs[j])):
                self.models[i].RA[j-1, k] = (xs[i][k].f)-(xs[j][k].f)
    # """  """
    # if isinstance(xs, np.ndarray) or isinstance(xs, list):
    #   for i in range(self.nm):
    #     if i > 0:
    #       if self.models[i].RA.size == 0:
    #         self.models[i].RA = np.zeros((self.nm-1, self.ns))
    #         for j in range(self.nm):
    #           if j > 0 and i != j:
    #             for k in range(len(xs[j])):
    #               self.models[i].RA[j-1, k] = (xs[i][k].f)-(xs[j][k].f)
    #       else:
            
    #         for j in range(self.nm):
    #           RAtemp: list = []
    #           if j > 0 and i != j:
    #             RAtemp = copy.deepcopy(self.models[i].RA[j-1].tolist())
    #             for k in range(len(xs[j])):
    #               RAtemp.append((xs[i][k].f)-(xs[j][k].f))
    #             self.models[i].RA.resize((self.nm-1,len(RAtemp)), refcheck=False)
    #             self.models[i].RA[j-1] = np.asarray(copy.deepcopy(RAtemp))
                
    

  def build_Adeq_surrogates(self):
    """  """
    for i in range(self.nm):
      if self.models[i].rank < self.nm:
        self.models[i].build_surrogate_errors(self.xs_hist[i])

  def update_absolute_adeq(self, model_index):
    """ TODO: """

  def update_relative_adeq(self, model_index):
    """ TODO: """

  def evaluate(self, x: List[float]):
    
    xtemp: Point = Point()
    xtemp.coordinates = copy.deepcopy(x)
    xtemp.evaluated = 0
    x2 = copy.deepcopy(self.predict_RAM(xtemp))
    # if self.switch_to_search and self.psevals < 2:
    #   self.search_step()
    #   # x2 = copy.deepcopy(self.predict_RAM(xtemp, 0))
    #   # self.switch_to_search = False
    #   self.psevals += 1
    # elif self.switch_to_search:
    #   x2 = copy.deepcopy(self.predict_RAM(xtemp, 0))
    #   self.switch_to_search = False
      
    return [x2.f, [x2.h]]


  def poll(self):
    self.psevals = 0
    fun : Callable = self.evaluate
    eval = {"blackbox": fun}
    param = {"baseline": self.xmin.coordinates,
                "lb": self.prob_params.lb,
                "ub": self.prob_params.ub,
                "var_names": self.prob_params.var_names,
                "scaling": self.prob_params.scaling,
                "post_dir": "./post"}
    options: Dict = {}
    
    for i in [[getattr(self.optimizer_param, attr), attr] for attr in dir(self.optimizer_param) if not attr.startswith("__")]:
      if i[1] != 'optimizer':
        options[i[1]] = i[0]
    if self.succ_search:
      options["psize_init"] = self.optimizer_param.psize_init * 0.5
    else:
      options["psize_init"] = self.optimizer_param.psize_init

    data = {"evaluator": eval, "param": param, "options":options}

    out = {}
    out = OMADS.main(data)
    xtemp: Point = Point()
    xtemp.coordinates = out["xmin"]
    xtemp = copy.deepcopy(self.evaluate_phi_using_model_i([xtemp], 0))[0]
    xtemp.evaluated = 1
    
    if xtemp.f < self.xmin.f and xtemp.h <= self.xmin.h:
      self.xmin = copy.deepcopy(xtemp)
      self.poll_success = True
      self.vicinity_ratio = np.multiply(self.vicinity_ratio, 2.0)
      self.optimizer_param.psize_init = options["psize_init"]
      
    else:
      self.psize = 1.
      self.poll_success = False
      self.vicinity_ratio = np.multiply(self.vicinity_ratio, 0.5)
      self.optimizer_param.psize_init = options["psize_init"]*0.5
      self.switch_to_search
    
    if self.RAF_params.display_overview:
      print(f'iteration= {self.iter}, step_name= poll_step, is_success= {self.poll_success}, xmin= {self.xmin.coordinates}, bbeval= {out["nbb_evals"]}, psize_init=  {options["psize_init"]}, fmin= {self.xmin.f}, hmin= {self.xmin.h}')
      if self.mToUpdate.size != 0 and self.mToRefer.size != 0:
          print(f'Updated models= {self.mToUpdate}, Referral models= {self.mToRefer}')
    
    return out

  
  
  def set_models_to_evaluators(self, ms: List[Callable]):
    self.models = [1] * len(ms)
    
    for i in range(len(ms)):
      self.models[i] = model()
      self.models[i].TRM = DFTRM()
      self.models[i].TRM.Delta = 0.5
      self.models[i].nsamples = self.ns
      self.models[i].rank = len(ms)-i
      self.models[i].bb = copy.deepcopy(ms[i])
      self.models[i].rtime = len(ms)-i 

  def check_stopping_critt(self):
    """ Check if the RAF outer loop can be terminated """
    if self.iter >= self.RAF_params.max_it or self.neval >= self.RAF_params.neval_total or self.optimizer_param.psize_init <self.RAF_params.tol:
      return True
    else:
      return False
  
  def calculate_Lambda(self, ks:int = None):
    """ Calculate the distances among the current predicted error of a model and other models """
    for i in range(self.nm):
      if ks is not None:
        if self.iter == 0:
          self.models[i].Lambda = np.multiply(self.models[i].RA[ks], np.inf) 
        else:
          self.models[i].Lambda = np.abs(np.subtract(self.models[i].RA[ks] - self.models[i].RA_old[ks]))
      else:
        if self.iter == 0:
          self.models[i].Lambda = np.multiply(self.models[i].RA, np.inf) 
        else:
          self.models[i].Lambda = np.abs(np.subtract(self.models[i].RA - self.models[i].RA_old))
  
  def update(self):
    for i in range(self.nm):
      self.models[i].RA_old = copy.deepcopy(self.models[i].RA)
      self.models[i].AA_old = copy.deepcopy(self.models[i].AA)
      self.models[i].RA_min_old = copy.deepcopy(self.models[i].RA_min)
      self.models[i].AA_min_old = copy.deepcopy(self.models[i].AA_min)
      self.models[i].TRM.xold = copy.deepcopy(self.xmin)
      self.models[i].TRM.xold_hat = copy.deepcopy(self.models[i].xmin)
      if self.iter == 0:
        self.models[i].TRM.Delta = [self.Delta_init] * (self.nm-1)
        self.models[i].Lambda = [np.inf] * (self.nm-1)
    self.iter += 1
    self.terminate = self.check_stopping_critt()

  
  def run(self):
    goToSearch: bool = True
    while True:
      if goToSearch:
        self.search_step()
        # Go to the poll step after updating algorithm parameters
        goToSearch = False
      else:
        self.poll()
        if not self.poll_success:
          # Go to the search step when OMAD cannot find any better
          goToSearch = True
      # Update the algorithm parameters
      self.update()
      # Check termination cirteria
      if self.terminate:
        break
  
  def update_adeq_at_minimizer(self):
    xmin: Point = Point()
    xmin = copy.deepcopy(self.models[0].xmin)
    for i in range(self.nm):
      if i != 0:
        self.models[i].RA_min = []
        self.models[i].AA_min = 0.
        self.models[i].AA_min = self.models[i].calculate_error(copy.deepcopy(xmin))
      for j in range(self.nm):
        if i != 0 and j!=0:
          xi = copy.deepcopy(self.models[i].evaluate_point(copy.deepcopy(xmin)))
          self.models[i].xmin = copy.deepcopy(xi)

          xj = copy.deepcopy(self.models[j].evaluate_point(copy.deepcopy(xmin)))          
          self.models[i].RA_min.append(self.models[i].calculate_error(xj))

  def acquire_prior_knowledge(self, ns: int = None):
    """ """
    tic = time.perf_counter()
    # Generate initial samples (trial points)
    # Define the list of models to be managed (selected and rectified) during the optimization process
    # The models can be a list of executables (blackboxes) or callables
    Ms = []
    for i in self.RAF_params.bb_models:
      Ms.append(globals()[i])
    # Set each model in the list to an evaluator (construct evaluators class)
    self.set_models_to_evaluators(Ms)
    # Set number of avialable models
    self.nm = len(self.models)
    # Set minimum point to the initial starting point (baseline)
    self.xmin = Point()
    self.xmin.coordinates = self.prob_params.baseline
    self.evaluate_phi_using_model_i([self.xmin], 0)
  # Set number of the outer loop iterations
    S: List[Point] = self.generate_sample_points(ns)
    self.hashtable = Cache()
    for s in S: self.hashtable.add_to_cache(s)
    # Initialize TR radius
    self.Delta = [0.] * self.nm * self.ns
    self.Delta_init = 1.
    # Initialize the absolute adequacy matrix
    self.AAM = np.empty((self.nm-1))
    # Initialize relative adequacy matrix
    self.RAM = np.empty((self.nm-1, self.nm-1))
    
    S.append(self.xmin)
    # Evaluate the initial penalty function
    xs = self.evaluate_phi(S)
    if self.RAF_params.display_overview:
        print(f'iteration= {self.iter}, step_name= search_step, is_success={self.succ_search}, xmin= {self.xmin.coordinates}, bbeval= {self.ns}, psize_init=  {self.optimizer_param.psize_init}, fmin= {self.xmin.f}, hmin= {self.xmin.h}')
    self.xs_hist = []
    # Calculate absolute adequacies: discrepancies between models output and the reference model
    self.xs_hist = copy.deepcopy(xs)
    self.calc_absolute_adeq(xs)
    # Calculate relative adequacies: discrepancies between models output and the reference model
    self.calc_relative_adeq(xs)
    # Build error surrogates
    self.build_Adeq_surrogates()
    # Update adequacies at the current minimizer
    self.update_adeq_at_minimizer()
    # Calculate Lambda
    # self.calculate_Lambda()
    # Update initialization
    self.update()
    toc = time.perf_counter()

    self.offLineRunTime += toc - tic


    
if __name__ == "__main__":

  """ Parse the parameters files """
  if len(sys.argv) <= 1:
    sys.argv.append("tests/toy/rosen.json")
  if type(sys.argv[1]) is dict:
    data = sys.argv[1]
  elif isinstance(sys.argv[1], str):
    if os.path.exists(os.path.abspath(sys.argv[1])):
      _, file_extension = os.path.splitext(sys.argv[1])
      if file_extension == ".json":
        try:
          with open(sys.argv[1]) as file:
            data = json.load(file)
        except ValueError:
          raise IOError('invalid json file: ' + sys.argv[1])
      else:
          raise IOError(f"The input file {sys.argv[1]} is not a JSON dictionary. "
                        f"Currently, OMADS supports JSON files solely!")
    else:
      raise IOError(f"Couldn't find {sys.argv[1]} file!")
  else:
    raise IOError("The first input argument couldn't be recognized. "
                  "It should be either a dictionary object or a JSON file that holds "
                  "the required input parameters.")
  tic = time.perf_counter()
  # Construct the OMADS-RAF class
  rp = RAF_p(**data["RAF_params"])
  opt = optimizer(**data["optimizer"])
  P = Parameters(**data["prob_params"])
  MM = OMADS_RAF(RAF_params=rp, optimizer_param=opt, prob_params=P)
  MM.vicinity_ratio = np.ones(len(P.lb))
  MM.acquire_prior_knowledge(8)
  
  # Start running the RAF
  MM.ns = 8
  MM.run()
  toc = time.perf_counter()
  print(f'----Run-Summary----')
  print(f'Run completed in {toc-tic} seconds')
  print(f'xmin = {MM.xmin}')
  print(f'hmin ={MM.xmin.h}')
  print(f'fmin ={MM.xmin.f}')
  for i in range(MM.nm):
    if i == 0:
      print(f'Number of evaluations of the reference model is {MM.models[i].nevals}')
    else:
      print(f'Number of evaluations of model # {i} is {MM.models[i].nevals}')
    






  

