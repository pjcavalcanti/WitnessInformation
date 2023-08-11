from unittest import TestCase

import torch
from criterion_utils import *


class TestIsPositiveSemidefinite(TestCase):
    def test_is_positive_semidefinite_hermitian_only(self):
        for _ in range(1000):
            dim = int(torch.randint(1, 6, (1,)))
            m = torch.randn(dim, dim, dtype=torch.complex128) + 1j * torch.randn(
                dim, dim, dtype=torch.complex128
            )
            m = m + m.conj().T
            eigvals = torch.linalg.eigvalsh(m)
            min_eigval = eigvals.min()
            is_positive = min_eigval >= 0 or torch.isclose(
                min_eigval, torch.tensor(0.0).type(min_eigval.dtype)
            )
            self.assertTrue(is_positive_semidefinite(m) == is_positive)

    def test_is_positive_semidefinite_only_positive(self):
        for _ in range(1000):
            dim = int(torch.randint(1, 6, (1,)))
            m = torch.randn(dim, dim, dtype=torch.complex128) + 1j * torch.randn(
                dim, dim, dtype=torch.complex128
            )
            m = m @ m.conj().T
            self.assertTrue(is_positive_semidefinite(m))


class TestMutualInformationValues(TestCase):
    def test_mutual_information_values_ent(self):
        x = torch.tensor([0, 0.1, 0.1, 0.2])
        ent = torch.tensor([0, 0, 1, 1])
        
        p0, p1, p2 = torch.tensor(1 / 4), torch.tensor(1 / 2), torch.tensor(1 / 4)
        psep, pent = torch.tensor(1 / 2), torch.tensor(1 / 2)
        p0sep, p0ent, p1sep, p1ent, p2sep, p2ent = (
            torch.tensor(1 / 4),
            torch.tensor(0),
            torch.tensor(1 / 4),
            torch.tensor(1 / 4),
            torch.tensor(0),
            torch.tensor(1 / 4),
        )
        
        hx = -p0 * torch.log(p0) - p1 * torch.log(p1) - p2 * torch.log(p2)
        hent = -pent * torch.log(pent) - psep * torch.log(psep)  # type: ignore
        hxent = - p0sep * torch.log(p0sep) - p1ent * torch.log(p1ent) - p2ent * torch.log(p2ent) - p1sep * torch.log(p1sep)  # type: ignore
        
        mi_manual = hx + hent - hxent
        mi = mutual_information_values_ent(x, ent)
        
        self.assertTrue(torch.isclose(mi, mi_manual))
        
    def test_mutual_information_sign_ent(self):
        x = torch.tensor([0, -0.1, -0.1, -0.2])
        signs = torch.tensor([1, -1, -1, -1])
        ent = torch.tensor([0, 0, 1, 1])
        
        pp, pm = torch.tensor(1 / 4), torch.tensor(3 / 4)
        pent, psep = torch.tensor(1 / 2), torch.tensor(1 / 2)
        ppent, pment, ppsep, pmsep = torch.tensor(0), torch.tensor(1/2), torch.tensor(1 / 4), torch.tensor(1 / 4)
        
        hs = -pp * torch.log(pp) - pm * torch.log(pm)
        hent = -pent * torch.log(pent) - psep * torch.log(psep)
        hsent = - pment * torch.log(pment) - ppsep * torch.log(ppsep) - pmsep * torch.log(pmsep)
        
        mi = mutual_information_sign_ent(x, ent)
        mi_manual = hs + hent - hsent
        
        self.assertTrue(torch.isclose(mi, mi_manual))
