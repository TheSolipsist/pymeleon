import pymeleon as pym
import pygrank as pg
from mylib_pymeleon import viewer

def ppr_pym(*args):
    signal = viewer(*args) >> pym.parse(pg.GraphSignal)
    return signal >> pg.PageRank()
