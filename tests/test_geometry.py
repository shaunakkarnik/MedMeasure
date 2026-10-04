import unittest
import numpy as np
from medmeasure.geometry import fit_mask


class GeometryChecks(unittest.TestCase):
    def test_anisotropic_circle_is_measured_in_mm(self):
        row,col=np.mgrid[:160,:160]
        mask=((row-80)*.5)**2+((col-80)*1.2)**2 <= 20**2
        fits,count=fit_mask(mask,[.5,1.2])
        self.assertEqual(count,1)
        self.assertEqual(len(fits),1)
        self.assertAlmostEqual(fits[0]['major_mm'],40,delta=2)
        self.assertAlmostEqual(fits[0]['minor_mm'],40,delta=2)
        wrong,_=fit_mask(mask,[1.2,.5])
        self.assertGreater(wrong[0]['major_mm'],80)

    def test_empty_and_tiny_masks(self):
        mask=np.zeros((20,20),dtype=bool)
        self.assertEqual(fit_mask(mask,[1,1]),([],0))
        mask[10,10]=True
        self.assertEqual(fit_mask(mask,[1,1]),([],1))

    def test_multiple_components_are_counted_even_when_unmeasurable(self):
        row,col=np.mgrid[:80,:80]
        mask=(row-40)**2+(col-40)**2 < 100
        mask[0,0]=True
        fits,count=fit_mask(mask,[1,1])
        self.assertEqual(count,2)
        self.assertEqual(len(fits),1)

    def test_bad_spacing_rejected(self):
        with self.assertRaises(ValueError):
            fit_mask(np.zeros((10,10)),[0,1])


if __name__=='__main__':
    unittest.main()
