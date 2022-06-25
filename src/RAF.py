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
from tkinter.messagebox import NO
from typing import List, Dict, Any, Callable, Protocol, Optional

import numpy as np
from numpy import cos, exp, pi, prod, sin, sqrt, subtract, inf

from OMADS import Point, PreMADS, PostMADS, OrthoMesh, Parameters,  Evaluator, Cache, Directions2n, DefaultOptions
import OMADS
from enum import Enum, auto

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
    self.rho = (self.xold.f-xnew.f) / (self.xold_hat.f-xnew_hat.f)
    if self.rho < self.R1:
      self.Delta[index] *= self.c1
    elif self.rho > self.R2:
      self.Delta[index] *= self.c2

  def updateRadius_AA(self, xnew: Point, xnew_hat: Point):
    """  """
    self.rho = (self.xold.f-xnew.f) / (self.xold_hat.f-xnew_hat.f)
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
  
  def build_surrogate_errors(self, x):
    """ """

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
    time.sleep(0.3)
    return y

def rosen1(x, *argv):
  x = np.asarray(x)
  y = [np.sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0,
  axis=0), [0]]
  time.sleep(0.3)
  return y

def rosen2(x, *argv):
  x = np.asarray(x)
  y = [np.sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0, 
  axis=0) + np.sum(np.cos(x)), [0]]
  time.sleep(0.2)
  return y

def rosen3(x):
  x = np.asarray(x)
  y = [np.sum(np.add(np.sin(x), 100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0), axis=0), [0]]
  time.sleep(0.1)
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
class RAF_data:
  models: List[model] = model
  rtime: np.ndarray = None
  inputs: Parameters = None
  nvars: int = None
  sampling_t: int = SML.SAMPLING_METHOD.LH
  ns: int = 1
  xmin: Point = None
  hashtable: Cache = None
  max_it: int = 100
  tol: float = 1e-12
  psize: float = 1.
  Delta: np.ndarray = np.empty((1,1))
  Lambda: np.ndarray = np.empty((1,1))
  TRM: DFTRM = None
  nm: int = 1
  AAM: np.ndarray = field(default=np.empty((1)))
  RAM: np.ndarray = field(default=np.empty((1)))
  levels: np.ndarray = field(default=np.empty((1)))
  poll_success: bool = False
  terminate: bool = False
  iter: int = 0
  neval: int = 0
  eval_budget: int = 100
  switch_to_search: bool = False
  seed: int = 0
  display_poll: bool = False
  display_overview: bool = False
  RAM_old: np.ndarray = np.empty((0,0))
  xs_hist: List[List] = None
  Delta_init: List = None
  retrain_surrogates: bool = False




@dataclass
class OMADS_RAF(RAF_data):
  
  def generate_sample_points(self, nsamples: int = None) -> List[Point]:
    """ Generate the sample points """
    xlim = []
    v = np.empty((self.nvars, 2))
    for i in range(len(self.inputs.lb)):
      v[i] = [self.inputs.lb[i], self.inputs.ub[i]]
    if nsamples is None:
      nsamples = int((self.nvars+1)*(self.nvars+2)/2)
    
    self.ns = nsamples

    if self.sampling_t == SML.SAMPLING_METHOD.FULLFACTORIAL:
      sampling = SML.FullFactorial(ns=nsamples, vlim=v, w=np.array([0.8, 0.2]), c=True)
    elif self.sampling_t == SML.SAMPLING_METHOD.LH: 
      sampling = SML.LHS(ns=nsamples, vlim=v)
      sampling.options["criterion"] = "ExactSE"
    elif self.sampling_t == SML.SAMPLING_METHOD.RS:
      sampling = SML.RS(ns=nsamples, vlim=v, w=None, c=False)

    return self.map_samples_from_coords_to_points(sampling.generate_samples())
  
  def map_samples_from_coords_to_points(self, samples: np.ndarray) -> List[Point]:
    x: List[Point] = [0.] *self.ns
    for i in range(samples.shape[0]):
      x[i] = Point()
      x[i].coordinates = copy.deepcopy(samples[i])
    return x
      
  def evaluate_phi(self, points: List[Point]):
    """ """
    self.rtime = np.empty((self.nm))
    xs = [1] * self.nm
    
    for i in range(self.nm):
      tic = time.perf_counter()
      xs[i] = copy.deepcopy(points)
      xs[i] = copy.deepcopy(self.models[i].evaluate_samples(points))
      if i == 0:
        self.xmin = copy.deepcopy(self.models[i].xmin)
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

  def search_step(self):
    """ """
    if self.iter > 0:
      S: List[Point] = self.generate_sample_points()

      for i in range(self.ns):
        xtemp = copy.deepcopy(self.predict_RAM(S[i]))
        # self.calculate_Lambda()
        if self.stop_search:
          self.xmin = copy.deepcopy(xtemp)
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
      if self.display_overview:
        print(f'iteration= {self.iter}, step_name= search_step, bbeval= {self.ns}, psize_init=  {self.psize}, fmin= {self.xmin.f}, hmin= {self.xmin.h}')
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
      self.mToUpdate = np.empty((0,2))
      # models that we need to relatively refer to when inadequacy surrogates of other models have to be retrained
      self.mToRefer = np.empty((0,2))
      # Loop over the list of models to create/update the RAM (a square-matrix)
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
                self.mToUpdate = np.asarray([[self.models[i].rtime, i]])
              else:
                self.mToUpdate = np.append(self.mToUpdate, [[self.models[i].rtime, i]], axis=0)
            else:
              # Otherwise we can refer to it when updating/correcting other models that have less accuracy than that one
              if self.mToRefer.size == 0:
                self.mToRefer = np.asarray([[self.models[i].rtime, i]])
              else:
                self.mToRefer = np.append(self.mToRefer, [[self.models[i].rtime, i]], axis=0)
            xh = self.models[j].evaluate_point(copy.deepcopy(x))
            xl = self.models[i].evaluate_point(copy.deepcopy(x))
            self.models[i].TRM.updateRadius(xh, xl, j-1)

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
            xh = self.models[0].evaluate_point(copy.deepcopy(x))
            xl = self.models[k].evaluate_point(copy.deepcopy(x))
            self.models[i].TRM.updateRadius_AA(xh, xl)
      
      if self.mToRefer.size == 0:
        self.retrain_surrogates: bool = True
        self.switch_to_search = True
        self.stop_search = False
        return x
      # Evaluate all models, including reference model, to train/retrain adequacy surroagtes
      for m in range(sorted_mToUpdate.shape[0]):
        xm: Point = self.models[sorted_mToUpdate[m,1]].evaluate_point(copy.deepcopy(x))
        mh = sorted_mToRefer[m, 1]
        
        x = self.models[mh].evaluate_point(copy.deepcopy(x))
        
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
          elif self.models[i].AAM:
            self.models[i].AA[j] = abs((xs[0][j].f)-(xs[i][j].f))
          else:
            raise Exception(RuntimeError, "Absolute adequacy matrix should be initialized before dumping errors in it!")
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
    param = {"baseline": self.models[-1].xmin.coordinates,
                "lb": self.inputs.lb,
                "ub": self.inputs.ub,
                "var_names": self.inputs.var_names,
                "scaling": self.inputs.scaling,
                "post_dir": "./post"}
    options = {"seed": self.seed, "budget": self.eval_budget, "tol": self.tol, "display": self.display_poll, "psize_init": self.psize}

    data = {"evaluator": eval, "param": param, "options":options}

    out = {}
    out = OMADS.main(data)
    xtemp = Point()
    xtemp.coordinates = out["xmin"]
    xtemp.f = out["fmin"]
    xtemp.h = out["hmin"]
    xtemp.evaluated = 1
    if out["fmin"] < self.xmin.f and out["hmin"] <= self.xmin.h:
      self.xmin = copy.deepcopy(xtemp)
      self.poll_success = True
      self.psize = out["psize"]
    else:
      # self.psize = 1.
      self.poll_success = False
    
    if self.display_overview:
      # {"xmin": poll.xmin.coordinates,
      #   "fmin": poll.xmin.f, 
      #   "hmin": poll.xmin.h, 
      #   "nbb_evals" : poll.bb_eval, 
      #   "niterations" : iteration, 
      #   "nb_success": poll.nb_success, 
      #   "psize": poll.mesh.psize, 
      #   "psuccess": poll.mesh.psize_success, 
      #   "pmax": poll.mesh.psize_max}
      print(f'iteration= {self.iter}, step_name= poll_step, bbeval= {out["nbb_evals"]}, psize_init=  {self.psize}, hmin= {out["hmin"]}, fmin= {out["fmin"]}')
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
    if self.iter >= self.max_it or self.neval >= self.eval_budget or self.psize <self.tol:
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
          # Go to the search step when OMAD finds no better
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
    

    
if __name__ == "__main__":
  tic = time.perf_counter()
  # Construct the OMADS-RAF class
  MM = OMADS_RAF()
  # Define the list of models to be managed (selected and rectified) during the optimization process
  # The models can be a list of executables (blackboxes) or callables
  Ms = [sphere1, sphere2, sphere3]
  # Set each model in the list to an evaluator (construct evaluators class)
  MM.set_models_to_evaluators(Ms)
  # Set number of avialable models
  MM.nm = len(MM.models)
  # Set number of the outer loop iterations
  MM.max_it = 10
  # Set design and OMADS parameters
  # Set the poll size tolerance
  MM.tol = 1e-4
  # Set initial poll size
  MM.psize = 1.
  # Construct input parameters
  MM.inputs = Parameters()
  MM.inputs.lb = [-2, -2]
  MM.inputs.ub = [2, 2]
  MM.inputs.baseline = [-1., 1.]
  MM.nvars = 2
  MM.eval_budget = 10000
  MM.seed = 0
  MM.display_poll = True
  MM.display_overview = True
  # Generate initial samples (trial points)
  S: np.ndarray = MM.generate_sample_points(50)
  # Initialize TR radius
  MM.Delta = [0.] * MM.nm * MM.ns
  MM.Delta_init = 1.
  # Initialize the absolute adequacy matrix
  MM.AAM = np.empty((MM.nm-1))
  # Initialize relative adequacy matrix
  MM.RAM = np.empty((MM.nm-1, MM.nm-1))
  # Set minimum point to the initial starting point (baseline)
  MM.xmin = Point()
  MM.xmin.coordinates = MM.inputs.baseline
  # Evaluate the initial penalty function
  xs = MM.evaluate_phi(S)
  MM.xs_hist = []
  # Calculate absolute adequacies: discrepancies between models output and the reference model
  MM.xs_hist = copy.deepcopy(xs)
  MM.calc_absolute_adeq(xs)
  # Calculate relative adequacies: discrepancies between models output and the reference model
  MM.calc_relative_adeq(xs)
  # Build error surrogates
  MM.build_Adeq_surrogates()
  # Update adequacies at the current minimizer
  MM.update_adeq_at_minimizer()
  # Calculate Lambda
  # MM.calculate_Lambda()
  # Update initialization
  MM.update()
  # Start running the RAF
  MM.run()
  toc = time.perf_counter()
  print(f'----Run-Summary----')
  print(f'Run completed in {toc-tic} seconds')
  print(f'xmin = {MM.xmin}')
  print(f'hmin =')
  print(f'fmin =')
  for i in range(MM.nm):
    if i == 0:
      print(f'Number of evaluations of the reference model is {MM.models[i].nevals}')
    else:
      print(f'Number of evaluations of model # {i} is {MM.models[i].nevals}')
    






  

