import os
import time
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import csv

os.environ['QT_QPA_PLATFORM'] = 'offscreen'

from random_generation_utils import (
    device,
    rand_rho,
    random_functional,
    random_witness_from_partial_transpose,
    random_witness_from_family,
)
from criterion_utils import (
    is_entangled_ppt,
    mutual_information_sign_ent,
    mutual_information_values_ent,
)

torch.manual_seed(0)


def generate_states(dim1=2, dim2=2, n_rhos=1000):
    """Generates quantum states using random rho.
    Returns a list of generated states."""

    return [rand_rho(dim1, dim2) for _ in tqdm(range(n_rhos), desc="Generating states")]


def generate_entanglement_labels(sample_states, dim1, dim2):
    """Generates entanglement labels for the states.
    Returns a tensor of entanglement labels."""

    n_rhos = len(sample_states)
    is_entangled = torch.zeros((n_rhos,), dtype=torch.float64).to(device)
    for i in tqdm(range(n_rhos), desc="Generating entanglement labels"):
        rho = sample_states[i]
        is_entangled[i] = torch.tensor(is_entangled_ppt(rho, dim1, dim2), dtype=torch.float64)#.clone().detach().to(device)  # type: ignore
    return is_entangled


def compute_trace_values(n_witnesses, states, dim1, dim2, get_witness_from_family):
    """Computes trace values for each witness.
    Returns a list of trace values for each witness."""

    trace_values_list = []
    for _ in tqdm(range(n_witnesses)):
        witness = get_witness_from_family(dim1, dim2).view(1, -1).to(device).conj()
        value = (witness @ states).squeeze().real
        trace_values_list.append(value)
    return trace_values_list


def compute_information(
    n_witnesses, states, is_entangled, dim1, dim2, get_witness_from_family
):
    """Computes mutual information for entanglement.
    Returns lists of fine and coarse grained information."""

    info_fine_grained = []
    info_coarse_grained = []
    
    for _ in tqdm(range(n_witnesses)):
        witness = get_witness_from_family(dim1, dim2).view(1, -1).to(device).conj()
        trace_values = (witness @ states).squeeze().real
        info_fine_grained.append(mutual_information_values_ent(trace_values, is_entangled).item())
        info_coarse_grained.append(mutual_information_sign_ent(trace_values, is_entangled).item())
        
    return info_fine_grained, info_coarse_grained


def plot_histograms(info_fine_grained, info_coarse_grained, filename):
    """Plots histograms of fine and coarse grained information.
    Not used in the paper, but can be used to visualize the data."""

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(f"{filename}")
    ax[0].set_title("Trace Value information")
    ax[0].xaxis.set_label_text("Mutual Information")
    ax[0].set_xlim([0, 0.1])

    ax[1].set_title("Trace Sign information")
    ax[1].xaxis.set_label_text("Mutual Information")
    ax[1].set_xlim([0, 0.012])

    ax[0].hist(info_fine_grained, bins=100)
    ax[1].hist(info_coarse_grained, bins=100)
    if not os.path.exists("histograms"):
        os.makedirs("histograms")
    plt.savefig(f"histograms/{filename}")
    plt.close(fig)

def simulate_fixed_parameters(
    n_witnesses,
    states,
    is_entangled,
    dim1,
    dim2,
    get_witness_from_family,
    filename="histograms.png",
):
    """Performs the simulation by generating states,
    computing mutual information, and plotting histograms."""

    info_fine_grained, info_coarse_grained = compute_information(
        n_witnesses, states, is_entangled, dim1, dim2, get_witness_from_family
    )
    # Optional, not needed for the paper:
    # plot_histograms(info_fine_grained, info_coarse_grained, filename)

    # Save each individual mutual information value into separate CSV files
    if not os.path.exists("data"):
        os.makedirs("data")

    with open(
        f"data/{filename.replace('.png', '_fine_grained.csv')}", "w", newline=""
    ) as file:
        writer = csv.writer(file)
        # writer.writerow(['Info_Fine_Grained'])
        for fine in info_fine_grained:
            writer.writerow([fine])

    with open(
        f"data/{filename.replace('.png', '_coarse_grained.csv')}", "w", newline=""
    ) as file:
        writer = csv.writer(file)
        # writer.writerow(['Info_Coarse_Grained'])
        for coarse in info_coarse_grained:
            writer.writerow([coarse])


def save_example_joint_events_distribution_data(is_entangled, states):
    """Specific for the qubit-qubit case"""
    witness = torch.tensor([[
        1/4, 0, 0, 1/2,
        0, 1/4, 0, 0,
        0, 0, 1/4, 0,
        1/2, 0, 0, 1/4,
        ]], dtype=torch.complex128).view(1, -1).to(device).conj()
    
    trace_values = (witness @ states).squeeze().real
    
    with open("data/example_joint_events.csv", "w") as file:
        writer = csv.writer(file)
        for i in range(len(trace_values)):
            writer.writerow([trace_values[i].item(), is_entangled[i].item()])
        
def save_entanglement_events(is_entangled, n1, n2):
    """Saves the entangled distribution for a given dimension"""
    with open(f"data/entanglement_events_{n1}_{n2}.csv", "w") as file:
        writer = csv.writer(file)
        for i in range(len(is_entangled)):
            writer.writerow([is_entangled[i].item()])
    
def simulate_all_parameters():
    """Performs the simulation for all combinations of
    dimensions, witness generators, and powers."""

    n_witnesses = 100000
    n_states = 100000
    dimensions_system1 = [2, 3]
    dimensions_system2 = [2]
    functional_generators = [
        random_functional,
        random_witness_from_partial_transpose,
        random_witness_from_family,
    ]
    powers = [1]
    
    # Keep track of total number of simulations
    total_simulations = len(dimensions_system1) * len(dimensions_system2) * len(functional_generators) * len(powers)
    simulation_count = 1
    start_time = time.time()

    # Iterate over all combinations of parameters:
    
    # System dimensions
    for n1 in dimensions_system1:
        for n2 in dimensions_system2:
            if n1 == n2 == 3:
                continue
            
            # Generate states and entanglement labels
            print(f"\n\nRunning simulation for {n1}x{n2} systems.")
            states = (
                torch.stack(generate_states(n1, n2, n_states))
                .to(device)
                .view(n_states, -1, 1)
            )
            is_entangled = generate_entanglement_labels(states, n1, n2)
            
            
            # Save sample joint probability distribution data for n1 = n2 = 2
            if (n1 == n2 == 2):
                save_example_joint_events_distribution_data(is_entangled, states)
            # Save entangled distribution data
            save_entanglement_events(is_entangled, n1, n2)

            # Observable Families
            for functional_generator in functional_generators:
                # Possibilities for higher moments
                for power in powers:
                    if (
                        functional_generator.__name__ == "random_witness_from_family"
                        and not (n1 == n2 == 2)
                    ):
                        continue

                    # Print partial elapsed time
                    elapsed_time = time.time() - start_time
                    hours, remainder = divmod(elapsed_time, 3600)
                    minutes, seconds = divmod(remainder, 60)
                    print(
                        f"\nRunning simulation {simulation_count} out of {total_simulations}. Elapsed time: {int(hours):02}:{int(minutes):02}:{seconds:02.0f}."
                    )

                    # Run simulation for given moment
                    def observable_generator_given_power(dim1, dim2):
                        w = functional_generator(dim1, dim2)
                        out = w.clone()
                        for _ in range(power - 1):
                            out = out @ w
                        return out

                    simulate_fixed_parameters(
                        n_witnesses,
                        states,
                        is_entangled,
                        n1,
                        n2,
                        observable_generator_given_power,
                        f"histograms_{n1}_{n2}_{functional_generator.__name__}_{power}th_momentum.png",
                    )
                    simulation_count += 1
                    
    # Print total elapsed time
    elapsed_time = time.time() - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"\nSimulation finished. Elapsed time: {int(hours):02}:{int(minutes):02}:{seconds:02.0f}.")
    
simulate_all_parameters()   