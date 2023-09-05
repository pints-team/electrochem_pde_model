import numpy as np

from ..element_global import ElementGlobal
from ...refdom import RefTri


class ElementTriBell(ElementGlobal):
    """The Bell triangle."""

    nodal_dofs = 6
    maxdeg = 5
    dofnames = ['u', 'u_x', 'u_y', 'u_xx', 'u_xy', 'u_yy']
    doflocs = np.array([[0., 0.],
                        [0., 0.],
                        [0., 0.],
                        [0., 0.],
                        [0., 0.],
                        [0., 0.],
                        [1., 0.],
                        [1., 0.],
                        [1., 0.],
                        [1., 0.],
                        [1., 0.],
                        [1., 0.],
                        [0., 1.],
                        [0., 1.],
                        [0., 1.],
                        [0., 1.],
                        [0., 1.],
                        [0., 1.]])
    refdom = RefTri

    def gdof(self, F, w, i):
        if i < 18:
            j = i % 6
            k = int(i / 6)
            if j == 0:
                return F[()](*w['v'][k])
            elif j == 1:
                return F[(0,)](*w['v'][k])
            elif j == 2:
                return F[(1,)](*w['v'][k])
            elif j == 3:
                return F[(0, 0)](*w['v'][k])
            elif j == 4:
                return F[(0, 1)](*w['v'][k])
            elif j == 5:
                return F[(1, 1)](*w['v'][k])
        self._index_error()
