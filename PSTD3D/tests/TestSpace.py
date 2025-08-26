import unittest
from PSTD3D.src.typespace import ReadSpaceParams

class TestSpace(unittest.TestCase):

    def setUp(self):

        space = self
        
        space.dims = 3
        space.Nx = 32
        space.Ny = 32   
        space.Nz = 32
        space.dx = 0.1
        space.dy = 0.1
        space.dz = 0.1
        pass


    def test_ReadSpaceParams(self):
        
        params = ReadSpaceParams('PSTD3D/tests/params.txt')

        self.assertEqual(params['dims'], 3)
        self.assertEqual(params['Nx'], 32)
        self.assertEqual(params['Ny'], 32)
        self.assertEqual(params['Nz'], 32)
        self.assertAlmostEqual(params['dx'], 0.1)
        self.assertAlmostEqual(params['dy'], 0.1)
        self.assertAlmostEqual(params['dz'], 0.1)