import torch

# Define the device (use GPU if available)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def basis_element(dim, i):
    """
    Create a basis vector in a complex Hilbert space of dimension 'dim' with 1 in the i-th position.

    Args:
    dim (int): The dimension of the basis.
    i (int): The position of the nonzero entry in the basis vector.
    
    Returns:
    torch.Tensor: The i-th basis vector with shape (dim, 1).
    """
    keti = torch.zeros((dim, 1)).type(torch.complex128).to(device)
    keti[i] = 1
    return keti

def id(dim):
    """
    Create an identity matrix of dimension 'dim'.

    Args:
    dim (int): The dimension of the identity matrix.
    
    Returns:
    torch.Tensor: The identity matrix of dimension 'dim'.
    """
    return (torch.eye(dim) + 0j).type(torch.complex128)

def random_unitary(dim):
    """
    Create a random unitary matrix of dimension 'dim'.

    Args:
    dim (int): The dimension of the unitary matrix.
    
    Returns:
    torch.Tensor: The random unitary matrix of dimension 'dim'.
    """
    m = (torch.randn(dim, dim) + 1j * torch.randn(dim, dim)).type(torch.complex128).to(device)
    q, r = torch.linalg.qr(m)
    return q @ torch.diag_embed(torch.diag(r) / torch.abs(torch.diag(r)))

def rand_rho_pure(dim1, dim2):
    """
    Create a random pure density matrix for a bipartite system.

    Args:
    dim1, dim2 (int): The dimensions of the two subsystems.
    
    Returns:
    torch.Tensor: The random pure density matrix.
    """
    dim = dim1 * dim2
    psi = basis_element(dim, 0)
    u = random_unitary(dim)
    psi = u @ psi
    return psi @ psi.conj().T

def rand_rho_purification(dim1, dim2):
    """
    Create a random purified density matrix for a bipartite system.

    Args:
    dim1, dim2 (int): The dimensions of the two subsystems.
    
    Returns:
    torch.Tensor: The random purified density matrix.
    """
    dim = dim1 * dim2
    psi = basis_element(dim ** 2, 0)
    u = random_unitary(dim ** 2)
    psi = u @ psi
    return psi @ psi.conj().T

def partial_trace_kraus_operator(dim1,dim2, k):
    """
    Create the k-th Kraus operator for the partial trace.

    Args:
    dim1, dim2 (int): The dimensions of the two subsystems.
    k (int): The index of the Kraus operator.
    
    Returns:
    torch.Tensor: The k-th Kraus operator.
    """
    id = torch.eye(dim1).type(torch.complex128).to(device)
    Ek = torch.kron(id,basis_element(dim2, k))
    return Ek

def partial_trace_kraus_operators(dim1,dim2):
    """
    Create all Kraus operators for the partial trace.

    Args:
    dim1, dim2 (int): The dimensions of the two subsystems.
    
    Returns:
    list[torch.Tensor]: The list of all Kraus operators.
    """
    return [partial_trace_kraus_operator(dim1,dim2, k) for k in range(dim2)]

def partial_trace(rho, dim1, kraus_operators):
    """
    Perform the partial trace of a density matrix over the second system.

    Args:
    rho (torch.Tensor): The density matrix.
    dim1 (int): The dimension of the subsystem to keep.
    kraus_operators (list[torch.Tensor]): The Kraus operators for the partial trace.
    
    Returns:
    torch.Tensor: The partial trace of the density matrix.
    """
    pt = torch.zeros((dim1, dim1)).type(torch.complex128).to(device)
    for Ek in kraus_operators:
        pt += Ek.conj().T @ rho @ Ek
    return pt

def partial_transpose(matrix, d1, d2, system=2):
    """
    Perform the partial transpose of a matrix.

    Args:
    matrix (torch.Tensor): The matrix to transpose.
    d1, d2 (int): The dimensions of the two subsystems.
    system (int): The index of the subsystem to transpose (1 or 2).
    
    Returns:
    torch.Tensor: The partial transpose of the matrix.
    """
    tensor = matrix.reshape(d1, d2, d1, d2)
    if system == 1:
        transposed_tensor = torch.permute(tensor, (2, 1, 0, 3))
    elif system == 2:
        transposed_tensor = torch.permute(tensor, (0, 3, 2, 1))
    else:
        raise ValueError("Invalid value for system. Must be 1 or 2.")
    transposed_matrix = transposed_tensor.reshape(d1 * d2, d1 * d2)
    return transposed_matrix

def rand_rho(dim1, dim2, kraus=None):
    """
    Generate a random density matrix for a bipartite system using purification.

    Args:
    dim1, dim2 (int): The dimensions of the two subsystems.
    kraus (list[torch.Tensor], optional): The Kraus operators for the partial trace. 
    
    Returns:
    torch.Tensor: The random density matrix.
    """
    rho_pure = rand_rho_purification(dim1, dim2)
    if kraus is None:
        kraus = partial_trace_kraus_operators(dim1 * dim2, dim1 * dim2)
    return partial_trace(rho_pure, dim1 * dim2, kraus)

def random_functional(dim1, dim2):
    """
    Generate a random Hermitian matrix (functional) with unit trace.

    Args:
    dim1, dim2 (int): The dimensions of the two subsystems.
    
    Returns:
    torch.Tensor: The random Hermitian matrix with unit trace.
    """
    m = (torch.randn((dim1 * dim2, dim1 * dim2)).type(torch.complex128) + 1j * torch.randn((dim1 * dim2, dim1 * dim2)).type(torch.complex128)).to(device)
    m = m + m.conj().T
    return m / (torch.trace(m @ m.conj().T) ** 0.5)

def random_witness_from_partial_transpose(dim1, dim2):
    """
    Generate a random entanglement witness from the partial transpose of a pure state.

    Args:
    dim1, dim2 (int): The dimensions of the two subsystems.
    
    Returns:
    torch.Tensor: The random entanglement witness.
    """
    pure_rho = rand_rho_pure(dim1, dim2)
    return partial_transpose(pure_rho, dim1, dim2, system=2)

def random_witness_from_family(dim1, dim2):
    """
    Generate a random entanglement witness from a family of witnesses for qubits.

    Args:
    dim1, dim2 (int): The dimensions of the two subsystems.

    Returns:
    torch.Tensor: The random entanglement witness.

    Raises:
    AssertionError: If dimensions aren't equal to 2 (the function is implemented only for qubits).
    """
    assert dim1 == dim2 and dim1 == 2, "Witness from family only implemented for qubits."
    
    w = torch.zeros((dim1 * dim2, dim1 * dim2)).type(torch.complex128).to(device)
    params = 2 * torch.rand(3) - 1
    alpha, beta, gamma = params[0], params[1], params[2]
    gamma = gamma * torch.sqrt((alpha**2 + beta**2)/2)
    
    alpha = alpha.type(torch.complex128)
    beta = beta.type(torch.complex128)
    gamma = gamma.type(torch.complex128)
    w[0,0] = 1 + gamma
    w[1,1] = 1 - gamma
    w[2,2] = 1 - gamma
    w[3,3] = 1 + gamma
    w[3,0] = alpha + beta
    w[0,3] = alpha + beta
    w[1,2] = alpha - beta
    w[2,1] = alpha - beta
    return w / 4
