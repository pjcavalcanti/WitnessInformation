from unittest import TestCase

from random_generation_utils import *
torch.manual_seed(0)

class TestId(TestCase):
    def test_id_2(self):
        dim = 2
        expected = torch.tensor([[1, 0], [0, 1]], dtype=torch.complex128)
        actual = id(dim)
        self.assertTrue(torch.allclose(expected, actual), f"id fail for dim={dim}")

    def test_id_3(self):
        dim = 3
        actual = id(dim)
        expected = torch.tensor(
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.complex128
        )
        self.assertTrue(torch.allclose(expected, actual), f"id fail for dim={dim}")


class TestBasisElement(TestCase):
    def test_basis_element_2_0(self):
        dim = 2
        basis_element_index = 0
        actual = basis_element(dim, basis_element_index)
        expected = torch.tensor([[1], [0]], dtype=torch.complex128)
        self.assertTrue(
            torch.allclose(expected, actual),
            f"basis_element fail for dim={dim}, basis_element_index={basis_element_index}",
        )

    def test_basis_element_2_1(self):
        dim = 2
        basis_element_index = 1
        actual = basis_element(dim, basis_element_index)
        expected = torch.tensor([[0], [1]], dtype=torch.complex128)
        self.assertTrue(
            torch.allclose(expected, actual),
            f"basis_element fail for dim={dim}, basis_element_index={basis_element_index}",
        )

    def test_basis_element_3_0(self):
        dim = 3
        basis_element_index = 0
        actual = basis_element(dim, basis_element_index)
        expected = torch.tensor([[1], [0], [0]], dtype=torch.complex128)
        self.assertTrue(
            torch.allclose(expected, actual),
            f"basis_element fail for dim={dim}, basis_element_index={basis_element_index}",
        )

    def test_basis_element_3_2(self):
        dim = 3
        basis_element_index = 2
        actual = basis_element(dim, basis_element_index)
        expected = torch.tensor([[0], [0], [1]], dtype=torch.complex128)
        self.assertTrue(
            torch.allclose(expected, actual),
            f"basis_element fail for dim={dim}, basis_element_index={basis_element_index}",
        )


class TestRandomUnitary(TestCase):
    def test_random_unitary(self):
        for _ in range(100):
            dim = int(torch.randint(2, 4, (1,)).item())
            u = random_unitary(dim)
            id = torch.eye(dim).type(torch.complex128)
            self.assertTrue(
                u.shape == (dim, dim), f"random_unitary wrong dimension for dim={dim}"
            )
            self.assertTrue(
                torch.allclose(u @ u.conj().T, id),
                f"random_unitary not unitary for dim={dim}",
            )
            self.assertTrue(
                torch.allclose(u.conj().T @ u, id),
                f"random_unitary not unitary for dim={dim}",
            )


class TestRandomRhoPure(TestCase):
    def test_random_rho_pure(self):
        for _ in range(100):
            n1, n2 = int(torch.randint(2, 4, (1,)).item()), int(
                torch.randint(2, 4, (1,)).item()
            )
            rho = rand_rho_pure(n1, n2)
            self.assertTrue(
                rho.shape == (n1 * n2, n1 * n2),
                f"random_rho_pure not correct shape for n1={n1}, n2={n2}",
            )
            self.assertTrue(
                torch.allclose(rho, rho.conj().T),
                f"random_rho_pure not hermitian for n1={n1}, n2={n2}",
            )
            self.assertTrue(
                torch.allclose(
                    torch.trace(rho), torch.tensor(1, dtype=torch.complex128)
                ),
                f"random_rho_pure not unit trace for n1={n1}, n2={n2}",
            )
            eigvals = torch.linalg.eigvalsh(rho)
            print(eigvals)
            min_eigval = torch.min(eigvals)
            self.assertTrue(
                torch.allclose(min_eigval, torch.tensor(0, dtype=torch.float64))
                or min_eigval > 0,
                f"random_rho_pure not positive semidefinite for n1={n1}, n2={n2}",
            )
            self.assertTrue(
                torch.allclose(
                    (rho @ rho).trace(), torch.tensor(1, dtype=torch.complex128)
                ),
                f"random_rho_pure not pure for n1={n1}, n2={n2}",
            )


class TestRandRhoPurification(TestCase):
    def test_rand_rho_purification(self):
        for _ in range(100):
            dim1, dim2 = int(torch.randint(2, 4, (1,)).item()), int(
                torch.randint(2, 4, (1,)).item()
            )
            rho_purification = rand_rho_purification(dim1, dim2)
            self.assertTrue(
                rho_purification.shape == ((dim1 * dim2) ** 2, (dim1 * dim2) ** 2),
                f"rand_rho_purification not correct shape for dim1={dim1}, dim2={dim2}",
            )
            self.assertTrue(
                torch.allclose(rho_purification, rho_purification.conj().T),
                f"rand_rho_purification not hermitian for dim1={dim1}, dim2={dim2}",
            )
            self.assertTrue(
                torch.allclose(
                    rho_purification.trace(), torch.tensor(1, dtype=torch.complex128)
                ),
                f"rand_rho_purification not unit trace for dim1={dim1}, dim2={dim2}",
            )
            eigvals = torch.linalg.eigvalsh(rho_purification)
            min_eigval = torch.min(eigvals)
            self.assertTrue(
                torch.allclose(min_eigval, torch.tensor(0, dtype=torch.float64))
                or min_eigval > 0,
                f"rand_rho_purification not positive semidefinite for dim1={dim1}, dim2={dim2}",
            )
            self.assertTrue(
                torch.allclose(
                    (rho_purification @ rho_purification).trace(),
                    torch.tensor(1, dtype=torch.complex128),
                ),
                f"rand_rho_purification not pure for dim1={dim1}, dim2={dim2}",
            )


class TestPartialTrace(TestCase):
    def test_partial_trace_2_2(self):
        dim1, dim2 = 2, 2
        for _ in range(100):
            m = torch.randn(
                dim1 * dim2, dim1 * dim2, dtype=torch.complex128
            ) + 1j * torch.randn(dim1 * dim2, dim1 * dim2, dtype=torch.complex128)
            kraus_operators = partial_trace_kraus_operators(dim1, dim2)
            actual = partial_trace(m, dim1, kraus_operators)
            expected = torch.tensor(
                [
                    [m[0, 0] + m[1, 1], m[0, 2] + m[1, 3]],
                    [m[2, 0] + m[3, 1], m[2, 2] + m[3, 3]],
                ],
                dtype=torch.complex128,
            )
            self.assertTrue(
                torch.allclose(expected, actual),
                f"partial_trace fail for dim1={dim1}, dim2={dim2}",
            )

    def test_partial_trace_2_3(self):
        dim1, dim2 = 2, 3
        for _ in range(100):
            m = torch.randn(
                dim1 * dim2, dim1 * dim2, dtype=torch.complex128
            ) + 1j * torch.randn(dim1 * dim2, dim1 * dim2, dtype=torch.complex128)
            kraus_operators = partial_trace_kraus_operators(dim1, dim2)
            actual = partial_trace(m, dim1, kraus_operators)
            expected = torch.tensor(
                [
                    [m[0, 0] + m[1, 1] + m[2, 2], m[0, 3] + m[1, 4] + m[2, 5]],
                    [m[3, 0] + m[4, 1] + m[5, 2], m[3, 3] + m[4, 4] + m[5, 5]],
                ],
                dtype=torch.complex128,
            )
            self.assertTrue(
                torch.allclose(expected, actual),
                f"partial_trace fail for dim1={dim1}, dim2={dim2}",
            )


class TestPartialTranspose(TestCase):
    def test_partial_transpose_2_2(self):
        dim1, dim2 = 2, 2
        for _ in range(100):
            m = torch.randn(
                dim1 * dim2, dim1 * dim2, dtype=torch.complex128
            ) + 1j * torch.randn(dim1 * dim2, dim1 * dim2, dtype=torch.complex128)
            actual = partial_transpose(m, dim1, dim2, system=2)
            expected = m.clone()
            expected[1, 0] = m[0, 1]
            expected[0, 1] = m[1, 0]
            expected[0, 3] = m[1, 2]
            expected[1, 2] = m[0, 3]
            expected[2, 1] = m[3, 0]
            expected[3, 0] = m[2, 1]
            expected[2, 3] = m[3, 2]
            expected[3, 2] = m[2, 3]
            self.assertTrue(
                torch.allclose(expected, actual),
                f"partial_transpose fail for dim1={dim1}, dim2={dim2}",
            )

    def test_partial_transpose_2_3(self):
        dim1, dim2 = 2, 3
        for _ in range(100):
            m = torch.randn(
                dim1 * dim2, dim1 * dim2, dtype=torch.complex128
            ) + 1j * torch.randn(dim1 * dim2, dim1 * dim2, dtype=torch.complex128)
            actual = partial_transpose(m, dim1, dim2, system=2)
            expected = m.clone()

            expected[0, 1] = m[1, 0]
            expected[0, 2] = m[2, 0]
            expected[1, 2] = m[2, 1]
            expected[1, 0] = m[0, 1]
            expected[2, 0] = m[0, 2]
            expected[2, 1] = m[1, 2]

            expected[0, 4] = m[1, 3]
            expected[0, 5] = m[2, 3]
            expected[1, 5] = m[2, 4]
            expected[1, 3] = m[0, 4]
            expected[2, 3] = m[0, 5]
            expected[2, 4] = m[1, 5]

            expected[3, 1] = m[4, 0]
            expected[3, 2] = m[5, 0]
            expected[4, 2] = m[5, 1]
            expected[4, 0] = m[3, 1]
            expected[5, 0] = m[3, 2]
            expected[5, 1] = m[4, 2]

            expected[3, 4] = m[4, 3]
            expected[3, 5] = m[5, 3]
            expected[4, 5] = m[5, 4]
            expected[4, 3] = m[3, 4]
            expected[5, 3] = m[3, 5]
            expected[5, 4] = m[4, 5]

            self.assertTrue(
                torch.allclose(expected, actual),
                f"partial_transpose fail for dim1={dim1}, dim2={dim2}",
            )


class TestRandomRho(TestCase):
    def test_rand_rho(self):
        for _ in range(100):
            dim1, dim2 = int(torch.randint(2, 4, (1,))), int(torch.randint(2, 4, (1,)))
            rho = rand_rho(dim1, dim2, kraus=None)
            self.assertTrue(
                rho.shape == (dim1 * dim2, dim1 * dim2),
                f"rand_rho wrong dimension for dim1={dim1}, dim2={dim2}",
            )
            self.assertTrue(
                torch.allclose(rho, rho.conj().T),
                f"rand_rho not hermitian for dim1={dim1}, dim2={dim2}",
            )
            self.assertTrue(
                torch.allclose(
                    torch.trace(rho), torch.tensor(1.0, dtype=torch.complex128)
                ),
                f"rand_rho not trace 1 for dim1={dim1}, dim2={dim2}",
            )
            eigvals = torch.linalg.eigvalsh(rho)
            min_eigval = torch.min(eigvals)
            self.assertTrue(
                torch.allclose(min_eigval, torch.tensor(0.0, dtype=torch.float64))
                or min_eigval > 0,
                f"rand_rho not positive semidefinite for dim1={dim1}, dim2={dim2}",
            )
            purity = torch.trace(torch.matmul(rho, rho))
            self.assertTrue(
                torch.allclose(purity.imag, torch.tensor(0.0, dtype=torch.float64)),
                f"purity not real for dim1={dim1}, dim2={dim2}",
            )
            self.assertTrue(
                (purity.real <= 1 and purity.real > 0)
                or torch.allclose(purity.real, torch.tensor(0.0, dtype=torch.float64)),
                f"rand_rho invalid purity for dim1={dim1}, dim2={dim2}",
            )

class TestRandomFunctional(TestCase):
    def test_random_functional(self):
        for _ in range(100):
            dim1, dim2 = int(torch.randint(2, 4, (1,))), int(torch.randint(2, 4, (1,)))
            m = random_functional(dim1, dim2)
            self.assertTrue(m.shape == (dim1 * dim2, dim1 * dim2), f"random_functional wrong dimension for dim1={dim1}, dim2={dim2}")
            self.assertTrue(torch.allclose(m, m.conj().T), f"random_functional not hermitian for dim1={dim1}, dim2={dim2}")
            norm = torch.linalg.norm(m)
            self.assertTrue(torch.allclose(norm, torch.tensor(1.0, dtype=torch.float64)), f"random_functional not norm 1 for dim1={dim1}, dim2={dim2}")

class TestRandomWitnessPartialTranpose(TestCase):
    def test_random_witness_from_partial_transpose(self):
        for _ in range(100):
            dim1, dim2 = int(torch.randint(2, 4, (1,))), int(torch.randint(2, 4, (1,)))
            w = random_witness_from_partial_transpose(dim1, dim2)
            self.assertTrue(torch.allclose(w, w.conj().T), f"random_witness_from_partial_transpose not hermitian for dim1={dim1}, dim2={dim2}")
            norm = torch.linalg.norm(w)
            self.assertTrue(torch.allclose(norm, torch.tensor(1.0, dtype=torch.float64)), f"random_witness_from_partial_transpose not norm 1 for dim1={dim1}, dim2={dim2}")

class TestRandomWitnessFromFamily(TestCase):
    def test_random_witness_from_family(self):
        some_is_not_positive = False
        dim1, dim2 = 2, 2
        for _ in range(100):
            w = random_witness_from_family(dim1, dim2)
            self.assertTrue(torch.allclose(w, w.conj().T), f"random_witness_from_family not hermitian for dim1={dim1}, dim2={dim2}")
            eigvals = torch.linalg.eigvalsh(w)
            print(eigvals)
            min_eigval = torch.min(eigvals)
            if min_eigval < 0:
                some_is_not_positive = True
            max_eigval = torch.max(eigvals)
            self.assertTrue(max_eigval >= 0, f"random_witness_from_family does not have positive eigenvalue for dim1={dim1}, dim2={dim2}")
        self.assertTrue(some_is_not_positive, f"no random_witness_from_family has negative eigenvalue for dim1={dim1}, dim2={dim2}")
