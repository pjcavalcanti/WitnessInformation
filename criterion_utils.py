import torch
from random_generation_utils import device, partial_transpose


def is_positive_semidefinite(matrix):
    """
    Checks if the given matrix is positive semidefinite.
    
    Args:
    matrix (torch.Tensor): The matrix to check.
    
    Returns:
    bool: True if the matrix is positive semidefinite, False otherwise.
    """
    if not torch.allclose(matrix, matrix.conj().T):
        return False

    min_eig = torch.linalg.eigvalsh(matrix).min()
    return min_eig >= 0 or torch.isclose(min_eig, torch.tensor(0.0).type(min_eig.dtype))


def is_entangled_ppt(rho, dim1, dim2):
    """
    Checks if the given density matrix is entangled using the Peres-Horodecki criterion (PPT).
    
    Args:
    rho (torch.Tensor): The density matrix to check.
    dim1 (int): The dimension of the first subsystem.
    dim2 (int): The dimension of the second subsystem.
    
    Returns:
    bool: True if the matrix is entangled, False otherwise.
    """
    rho_t = partial_transpose(rho, dim1, dim2, system=2)
    return not is_positive_semidefinite(rho_t)


def mutual_information_values_ent(values_vector, entangled_vector):
    """
    Calculates the mutual information between the values vector and the entangled vector.
    
    Args:
    values_vector (torch.Tensor): The vector of values.
    entangled_vector (torch.Tensor): The vector of entanglement labels.
    
    Returns:
    torch.Tensor: The mutual information.
    """
    events = torch.stack((values_vector, entangled_vector), dim=1)
    counts, _ = torch.histogramdd(events, bins=[100, 2])
    
    pJoint = counts / counts.sum()
    pValue = pJoint.sum(dim=1)
    pEntangled = pJoint.sum(dim=0)
    
    hValue = - torch.special.xlogy(pValue, pValue).sum()
    hEntangled = - torch.special.xlogy(pEntangled, pEntangled).sum()
    hJoint = - torch.special.xlogy(pJoint, pJoint).sum()

    return hValue + hEntangled - hJoint


def mutual_information_sign_ent(values_vector, entangled_vector):
    """
    Calculates the mutual information between the signs of the values vector and the entangled vector.
    
    Args:
    values_vector (torch.Tensor): The vector of values.
    entangled_vector (torch.Tensor): The vector of entanglement measures.
    
    Returns:
    torch.Tensor: The mutual information.
    """
    signs_vector = torch.sign(values_vector)
    events = torch.stack((signs_vector, entangled_vector), dim=1)
    counts, _ = torch.histogramdd(events, bins=[2, 2])
    
    pJoint = counts / counts.sum()
    pSign = pJoint.sum(dim=1)
    pEntangled = pJoint.sum(dim=0)
    
    hSign = - torch.special.xlogy(pSign, pSign).sum()
    hEntangled = - torch.special.xlogy(pEntangled, pEntangled).sum()
    hJoint = - torch.special.xlogy(pJoint, pJoint).sum()

    return hSign + hEntangled - hJoint
