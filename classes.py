import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional, Callable
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
import seaborn as sns
from tqdm import tqdm
from math import ceil
from matplotlib.ticker import FuncFormatter

class System(ABC):
    """
    Base class for statistical mechanical analysis of system with N degress of freedom. Inherited by all models.
    """
    
    def __init__(self, N: int):
        self.N = N
        self.coupling_matrix = np.zeros((N,N))
        self.current_state = np.zeros(N, dtype=int)
        self.state_type = None
        
    @abstractmethod
    def energy(self, states: np.array, couplings: np.array) -> np.array:
        """Calculate total energy(Hamiltonian) of system for given state(s)"""
        pass
    
    @abstractmethod
    def boltzmann_factor(self, states: np.array, couplings: np.array, beta: float=1.0) -> np.array:
        """Calculate boltzmann factor e^{-H(S,J)B} for input state(s) S at inverse temp beta"""
        pass

    @abstractmethod
    def partition_function(self, couplings: np.array, beta: float=1.0):
        pass
    
    @abstractmethod
    def boltzmann_dist(self, states: np.array, couplings: np.array, beta: float=1.0) -> np.array:
        pass
    
    @abstractmethod
    def free_energy(self, couplings: np.array, beta: float=1.0):
        pass
    
    @abstractmethod
    def local_energy_change(self, site: int, new_state: int) -> float:
        """ Calculate energy change if site changes to to new_state"""
        pass

    @abstractmethod
    def find_ground_states(self) -> np.array:
        """ Find ground state configurations"""
        pass

    @abstractmethod
    def find_ground_state_energy(self) -> float:
        """ Find ground state configurations"""
        pass
    
    def set_coupling(self, i:int, j:int, strength:float):
        """ Set coupling between sites i and j, assuming symmetry"""
        self.coupling_matrix[i,j] = strength
        self.coupling_matrix[j,i] = strength
        
    def set_couplings_from_dict(self, couplings: Dict[Tuple[int, int], float]):
        """Set multiple couplings from a dictionary of (i,j): J_ij pairs"""
        for (i, j), strength in couplings.items():
            self.set_coupling(i, j, strength)
        
    def get_neighboors(self, site: int) -> List[int]:
        """ Return list of a site's neighbors """
        return np.where(self.coupling_matrix[site] != 0)[0].tolist()
    
    def print_current_properties(self):
        print("Current state: ", self.current_state)
        print("Current energy: ", self.energy(self.current_state, self.coupling_matrix))
        print("Current magnetization: ", self.magnetization())
        print("Current state Boltzmann factor: ", self.boltzmann_factor())

    def print_ground_state_val(self):
        print("Number of possible state configurations: ",len(self.find_all_configs()))
        print("Ground state configurations: ", self.find_ground_states())
        print("Ground state energy: ", self.find_ground_state_energy())

    
    def print_all_config_properties(self):
        all_configs = self.find_all_configs()
        all_boltzmann = self.boltzmann_factor(states=all_configs)
        all_energies = self.energy(states=all_configs)
        all_probabilities = self.boltzmann_dist(states=all_configs)
        
        
        for i, config in enumerate(all_configs):
            print(f"{config}\n Boltzmann factor: {all_boltzmann[i]}\n Energy: {all_energies[i]}\n Probability: {all_probabilities[i]}\n")
    
        print("Partition function/sum: ", self.partition_function())
        print("Free energy: ", self.free_energy())
    

    def to_networkx(self) -> nx.Graph:
        """ Convert to NetworkX graph"""
        G = nx.Graph()
        G.add_nodes_from(range(self.N))
        
        # add edges with weights
        for i in range(self.N):
            for j in range(i+1, self.N):
                if self.coupling_matrix[i,j] != 0:
                    G.add_edge(i,j, weight=self.coupling_matrix[i,j])
        
        # add node states as attributes
        nx.set_node_attributes(G, dict(enumerate(self.current_state)), 'state')
        return G

"""
======= Ising Model definition =======
"""
class IsingModel(System):
    """
    Generic model class for Ising type systems with +- 1 variable states. Standard Ising model or SK model
    depending on the Coupling matrix input.

    """
    def __init__(self, N: int, rng: np.random.Generator = None, coupling_matrix: np.array = None, h: float = 0.0, verbose: bool=True):
        super().__init__(N)
        self.coupling_matrix = coupling_matrix
        self.h = h 
        self.verbose = verbose

        # Create random number generator if a predefined one isnt input
        if rng is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = rng

        # Cache for configurations and energies
        self._all_configs = None
        self._all_energies = None
        self._energy_data_computed = False
        
        # initialize system with random +1/-1 spins
        self.current_state = rng.choice([+1,-1], size=N)
        self.state_type = 'ising'

    def _compute_all_energies(self):
        """ Private method to compute and cache configs and energies """
        if not self._energy_data_computed:
            if self.verbose is True:
                print(f"Computing all {2**self.N} configurations and energies for N={self.N} system...")
            self._all_configs = self.find_all_configs()
            self._all_energies = self.energy(self._all_configs)
            self._energy_data_computed = True
            if self.verbose is True:
                print("Energy Computation completed and stored")

    def get_all_configs_and_energies(self):
        """ 
        Get all configurations and their corresponding energies.
        This method computes them if they are not already cached.

        returns: all_configs, all_energies
        """
        if not self._energy_data_computed:
            self._compute_all_energies()
        return self._all_configs, self._all_energies
        
    def energy(self, states: np.array = None, couplings: np.array = None) -> np.array:
        """ Calculate Ising hamiltonian of states input as an array of arrays or a single array for one state, with a given coupling"""
        """ H = -1/2 ∑<i,j> J_ij*s_i*s_j - h*∑i s_i """
        if states is None:
            states = self.current_state
        
        if couplings is None:
            couplings = self.coupling_matrix
            
        # Handle single state input(1D array) by reshaping to 2D
        if states.ndim == 1:
            states = states.reshape(1,-1)
            squeeze_output = True # squeeze back to float output
        else:
            squeeze_output = False
        
        # Interaction energy calculating using vectorized functions
        # For each state S: -0.5 * sum_ij(J_ij * s_i * s_j)
        interaction_energy = -0.5 * np.sum((states @ couplings) * states, axis=1)
        
        # Vectorized field energy calculation
        # For each state S: -h * sum_i(s_i)
        field_energy = -self.h * np.sum(states, axis=1)
        
        # Total Energies
        total_energies = interaction_energy + field_energy
        
        # If input was 1D, return scalar; otherwise return array
        if squeeze_output:
            return total_energies[0]
        else:
            return total_energies
        
    def find_ground_states(self) -> np.array:
        """ Find ground state (lowest energy) configurations"""
        all_configs, all_energies = self.get_all_configs_and_energies()
        min_energy = np.min(all_energies)
        ground_state_indices = np.where(all_energies == min_energy)[0]

        return all_configs[ground_state_indices]
    
    def find_ground_state_energy(self) -> float:
        """ find the ground state (minimum) energy of the system """

        _, all_energies = self.get_all_configs_and_energies()

        return float(np.min(all_energies))

    def boltzmann_factor(self, states: np.array = None, couplings: np.array = None, beta: float=1.0) -> np.array:
        """ Calculate Boltzmann factor of input states with couplings J (default: coupling_matrix) and inverse temp beta:"""
        """ exp(-beta * H(S,J)"""
        # Calculate energies with energy function
        energies = self.energy(states, couplings)
        # Calculate Boltzmann factors
        boltzmann_factors = np.exp(-energies * beta)
        
        return boltzmann_factors
    
    def partition_function(self, couplings: np.array = None, beta: float=1.0) -> float:
        """ Returns the sum of boltzmann factors over all possible state configuations of system with given couplings at a specific beta"""
        """Z(J) = ∑S exp(-beta * H(S,J) """
        
        all_configs = self.find_all_configs()
        all_boltzmann = self.boltzmann_factor(states=all_configs, couplings=couplings, beta=beta)
        return np.sum(all_boltzmann)
    
    def boltzmann_dist(self, states: np.array = None, couplings: np.array = None, beta: float=1.0) -> np.array:
        """ Function to find boltzmann distribution values (probability system is in certain state configurations S at given beta and coupling) by default, finds it for all possible configurations of system"""
        """ P(S) = 1/Z * exp(-beta * H(S,J) """
        
        if states is None:
            states = self.current_state
        if couplings is None:
            couplings = self.coupling_matrix
        
        Z = self.partition_function(couplings=couplings, beta=beta)
        boltzmann_factors = self.boltzmann_factor(states=states, couplings=couplings, beta=beta)
        return boltzmann_factors / Z
        
    def free_energy(self, couplings: np.array = None, beta: float=1.0) -> float:
        """ Calculate free energy of system """
        """ F = -ln(Z) / beta """
        if couplings is None:
            couplings = self.coupling_matrix
            
        Z = self.partition_function(couplings=couplings, beta=beta)
        return -np.log(Z) / beta
        
    
        
    def find_all_configs(self):
        """ Finds all possible state configurations (complete phase space) for system. Returns a np array of np arrays with entries +1/-1 """
        num_configs = 2**self.N
        indices = np.arange(num_configs)
        configs = np.zeros((num_configs, self.N), dtype=int)
        for bit in range(self.N):
            configs[:, self.N-1-bit] = (indices >> bit) & 1
        # Convert from binary to -1/+1 representation
        return 2 * configs - 1
        
    
    def local_energy_change(self, site: int, new_state: int) -> float:
        """ Energy Change for flipping spin at site"""
        current_state = self.current_state[site]
        if new_state == current_state:
            return 0.0
        
        delta_spin = new_state - current_state
        interaction_change = -delta_spin * np.dot(self.coupling_matrix[site, :], self.current_state)
        field_change = -self.h * delta_spin
        
        return interaction_change + field_change
        
    def flip_spin(self, site: int):
        """Flips spin at a given site"""
        self.current_state[site] *= -1
        
    def magnetization(self) -> float:
        """ Calculate total magnetization in current state of system """
        return np.sum(self.current_state)
    
    def magnetization_per_site(self) -> float:
        """Calculate magnitization/N in current state of system """
        return self.magnetization() / self.N
    
    def thermodynamic_magnetization(self, couplings: np.array = None, beta: float = 1.0, per_site: bool = True) -> float:
        """ Calculate a thermodynamic equilibirum expectation value of absolute magnetization as weighted by probability of each configuration"""
        """ |M| = Σ_S |M(S)| * P(S)"""
        if couplings is None:
            couplings = self.coupling_matrix

        all_configs = self.find_all_configs()
        all_magnetizations = np.sum(all_configs, axis=1)
        probabilities = self.boltzmann_dist(states=all_configs, couplings=couplings, beta=beta)
        expected_abs_magnetization = np.dot(np.abs(all_magnetizations), probabilities)

        if per_site:
            return expected_abs_magnetization / self.N
        else:
            return expected_abs_magnetization

""" 
======= Topology Builder =======
"""

class TopologyBuilder:
    """ 
    Helper class to build common lattice topologies and initialize random coupling matrices for SK models.
    """
    
    @staticmethod
    def square_lattice(L: int, J: float=1.0, periodic: bool=True) -> np.ndarray:
        """ Create square lattice coupling matrix with uniform coupling J"""
        N = L * L
        coupling = np.zeros((N,N))
        
        for i in range(L):
            for j in range(L):
                site = i * L + j

                # Right neigbor
                if j < L - 1:
                    coupling[site, site+1] = J
                elif periodic:
                    coupling[site, i*L] = J

                # Down neighbor
                if i < L - 1:
                    coupling[site,site+L] = J
                elif periodic:
                     coupling[site, j] = J
                    
        # make symmetric
        coupling = coupling + coupling.T
        return coupling
    
    @staticmethod
    def triangle() -> Tuple[np.ndarray, int]:
        """ Create unfrustrated triangle """
        N = 3
        coupling = np.array([
            [0,1,1],
            [1,0,1],
            [1,1,0]
        ])
        
        return coupling, N


    @staticmethod
    def frustrated_triangle() -> Tuple[np.ndarray, int]:
        """ Create a frustrated triangle with 3 antiferromagnetic bonds"""
        N = 3
        coupling = np.array([
            [0,-1,-1],
            [-1,0,-1],
            [-1,-1,0]
        ])
        
        return coupling, N
    
    @staticmethod
    def one_bond_frustrated_triangle() -> Tuple[np.ndarray, int]:
        """ Create a frustrated triangle with 1 antiferromagnetic bond, 2 ferromagnetic bonds"""
        N = 3
        coupling = np.array([
            [0,-1,1],
            [-1,0,1],
            [1,1,0]
        ])
        
        return coupling, N
    
    @staticmethod
    def chain(N: int, J: float = 1.0, periodic: bool=True) -> np.ndarray:
        """ Create 1D chain coupling matrix """
        coupling = np.zeros((N, N))
        
        for i in range(N - 1):
            coupling[i,i+1] = J
            
        if periodic and N>2:
            coupling[0, N-1] = J
            
        coupling = coupling + coupling.T
        return coupling
    
    @staticmethod
    def random_frustrated_square(L: int, J_ferro: float=1.0, J_anti: float=-1.0, frustration_fraction: float = 0.1, periodic: bool = True) -> np.ndarray:
        """ Create square lattice with random frustrated bonds"""
        # start w/ ferromagnetic square lattice
        coupling = TopologyBuilder.square_lattice(L,J_ferro, periodic=periodic)
            
        # Randomly flip some bonds to antiferromagnetic
        N = L * L
        for i in range(N):
            for j in range(i+1, N):
                if coupling[i,j] != 0: # If theres already a bond
                    if np.random.random() < frustration_fraction:
                        coupling[i,j] = J_anti
                        coupling[j,i] = J_anti
                            
        return coupling
        
    @staticmethod
    def sk_random_network(N: int, rng: np.random.Generator = None):
        """Generate random symmetric Gaussian coupling matrix with variance 1/N"""
        # Create a rng if non is input
        if rng is None:
            rng = np.random.default_rng()

        # Standard normal has variance 1, multiply by sqrt(1/N) = 1/sqrt(N)
        coupling_upper = rng.standard_normal((N,N)) / np.sqrt(N)
        coupling_matrix = (coupling_upper + coupling_upper.T)/2
        # No self interaction
        np.fill_diagonal(coupling_matrix, 0.0)
        
        return coupling_matrix
    
    @staticmethod
    def random_regular_graph(N: int, k: int, rng: np.random.Generator = None) -> np.ndarray:
        """ Generates a random regular graph topolgy with +/-1 couplings, where N nodes are equal degree (k)"""
        if k%2 != 0 or N*k % 2 != 0:
            raise ValueError("N*k and k must be even to form a regular graph.")
        if k>=N or k<0:
            raise ValueError("Degree k must be between 0 an N-1")
        
        if rng is None:
            rng = np.random.default_rng()

        seed_value = int(rng.integers(0,100000))
        G = nx.random_regular_graph(n=N, d=k, seed=seed_value)
        coupling_matrix = np.zeros((N,N))

        for u,v in G.edges():
            J_uv =rng.choice([-1.0,1.0])
            coupling_matrix[u,v] = J_uv
            coupling_matrix[v,u] = J_uv

        return coupling_matrix

"""
======= Monte Carlo Sampling =======
"""
class MonteCarlo:
    """
    Class for doing Monte Carlo simulations using the Metropolis-Hastings algorithm. 
    For ising systems, proposed move is a state flip.
    """
    def __init__(self, system: System, rng: np.random.Generator = None):
        self.system = system
        if rng is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = rng

    def metropolis_step(self, beta: float = 1.0) -> bool:
        """ Single metropolis monte carlo step. Random site choosen, state flip proposed. 
            Move accepted or rejected baced on metropolis acceptance criteria, ensuring system
            evolves towards thermal equilibrium.
            Returns True if move accepted, False if not
        """
        # 1. Random select site
        site = self.rng.integers(0,self.system.N)

        # 2. Propose spin flip
        current_state = self.system.current_state[site]
        new_state = -current_state

        # 3. Calculate the change in energy of flip
        delta_E = self.system.local_energy_change(site, new_state)

        # 4. Apply Metropolis accpetance criterion:
        # if delta_E < 0 : always accept move
        # if delta_ > 0 : accept with probability exp(-beta * delta_E)
        if delta_E < 0 or self.rng.random() < np.exp(-beta * delta_E):
            self.system.flip_spin(site)
            return True
        else:
            return False

    def metropolis_sweep(self, beta: float = 1.0, n_sweeps: int = 1000, equilibration_sweeps: int = 100) -> dict:
        """ Perform multiple Metropolis sweeps and collect statistics after an equilibration/burnin period (getting system to thermal equilibrium)
            A 'sweep' consists of N individual metropolis steps, where N is num of sites in system, giving 
            each site on average one chance to be update per sweep.

            Returns a distionary containing arrays of measured observables after each sweep, such as energy and magnetization.
        """
        # Burn In phase
        for _ in range(equilibration_sweeps):
            for _ in range(self.system.N):
                self.metropolis_step(beta)
        
        # Measurement Phase
        energies = []
        magnetizations = []
        for _ in range(n_sweeps):
            for _ in range(self.system.N):
                self.metropolis_step(beta)

            # record quantities
            energies.append(self.system.energy())
            magnetizations.append(self.system.magnetization())

        return {
            'energies': np.array(energies),
            'magnetizations': np.array(magnetizations)
        }
    
    


"""
======= Classes for analysis =======
"""

class FrustrationAnalyzer:
    """
    For examining how frustration effects properties 
    of random frustrated square lattices
    """
    
    def __init__(self, L: int =4, J_ferro: float = 1.0, J_anti: float = -1.0, rng_seed: int = 111):
        """ Initialize with lattice parameters """
        self.L = L
        self.N = L*L
        self.J_ferro = J_ferro
        self.J_anti = J_anti
        self.rng = np.random.default_rng(rng_seed)
        
    def parameter_sweep(self, frustration_fractions: np.ndarray, temperatures: np.ndarray, n_realizations: int = 10) -> dict:
        """Parameter sweep over frustration fractions and temperatures"""
        
        n_ff = len(frustration_fractions)
        n_temps = len(temperatures)
        
        results = {
            'frustration_fractions': frustration_fractions,
            'temperatures': temperatures,
            'free_energies': np.zeros((n_ff, n_temps, n_realizations)),
            'free_energy_means': np.zeros((n_ff, n_temps)),
            'free_energy_sds': np.zeros((n_ff,n_temps)),
            'magnetizations': np.zeros((n_ff, n_temps, n_realizations)),
            'magnetization_means': np.zeros((n_ff,n_temps)),
            'magnetization_sds': np.zeros((n_ff,n_temps))
        }
        
        total_iter = n_ff * n_temps * n_realizations
        with tqdm(total=total_iter, desc="Computing") as pbar:
            for i, ff in enumerate(frustration_fractions):
                for j, temp in enumerate(temperatures):
                    beta = 1.0 / temp
                    
                    free_energies_temp = []
                    magnetizations_temp = []
                    
                    for k in range(n_realizations):
                        # Generate random frustrated lattice
                        coupling_matrix = TopologyBuilder.random_frustrated_square(L=self.L, J_ferro=self.J_ferro, J_anti=self.J_anti, frustration_fraction=ff)
                        
                        # Create Ising Model
                        model = IsingModel(self.N, rng=self.rng, coupling_matrix = coupling_matrix)
                        
                        # Calculate free energy and magnetization
                        free_energy = model.free_energy(beta=beta)
                        magnetization = model.thermodynamic_magnetization(beta=beta)
                        
                        free_energies_temp.append(free_energy)
                        magnetizations_temp.append(magnetization)
                        results['free_energies'][i,j,k] = free_energy
                        results['magnetizations'][i,j,k] = magnetization
                        
                        pbar.update(1)
                        
                    # Calculate statistics
                    results['free_energy_means'][i,j] = np.mean(free_energies_temp)
                    results['free_energy_sds'][i,j] = np.std(free_energies_temp)
                    results['magnetization_means'][i,j] = np.mean(magnetizations_temp)
                    results['magnetization_sds'][i,j] = np.std(magnetizations_temp)
                    
        return results
    
class BasicEnergyAnalyzer:
    """ 
    Analyzer for basic analysis of the Hamiltonian's behavior and closely realated quantities like
    the partition function and Boltzmann Distribution in IsingModels.
    """
    def __init__(self, model: IsingModel, temperatures: np.array = np.array([1.0, 5.0, 10.0, 20.0])):
        self.model = model
        self.temperatures = temperatures

    def get_coupling_samples(self, num_realizations: int = 1, topology: str = 'random_frustrated_square', **kwargs) -> list[np.ndarray]:
        """ Return a sample of randomly drawn couplings for the given model """
        rng = np.random.default_rng()
        if topology == 'random_frustrated_square':
            return [TopologyBuilder.random_frustrated_square(**kwargs) for _ in range(num_realizations)]
        elif topology == 'sk_random_network':
            return [TopologyBuilder.sk_random_network(N=self.model.N, rng=rng, **kwargs) for _ in range(num_realizations)]
        elif topology == 'random_regular_graph':
            return [TopologyBuilder.random_regular_graph(N=self.model.N, rng=rng, **kwargs) for _ in range(num_realizations)]
        else:
            raise ValueError("Not implemented or deterministic topology entered.")
    
    def analyze_energy_distribution(self) -> dict:
        """ Analyze hamiltonian energy levels across all possible configurations of the system """
        
        # get all possible configurations and energies from model object
        all_configs, all_energies = self.model.get_all_configs_and_energies()
        return {
            'all_configs': all_configs,
            'all_energies': all_energies
        }

    def analyze_boltzmann_distribution(self, temperature: float, ordered: bool) -> dict: 
        """ Analysizes Boltzmann distribution over all configurations at given temp"""
        beta = 1.0 / temperature
        all_configs, all_energies = self.model.get_all_configs_and_energies()
        probabilities = self.model.boltzmann_dist(states=all_configs, beta=beta)
        config_indices = np.arange(len(all_configs))

        if ordered:
            sorted_indices = np.argsort(all_energies)
            probabilities = probabilities[sorted_indices]

        return {
            'config_indices': config_indices,
            'probabilities': probabilities
        }
    
    def analyze_free_energy(self, temperature: float, ordered: bool, num_realizations: int = 1, topology: str = 'random_frustrated_square', **kwargs) -> dict:
        """ Analyzes free energy over all couplings at given temperature """
        beta = 1.0 / temperature
        all_couplings = self.get_coupling_samples(num_realizations=num_realizations, topology=topology, **kwargs) 
        free_energies = np.array([self.model.free_energy(couplings=c, beta=beta) for c in all_couplings])
        coupling_indices = np.arange(len(all_couplings))

        if ordered:
            sorted_indices = np.argsort(free_energies)
            free_energies = free_energies[sorted_indices]


        return {
            'coupling_indices': coupling_indices,
            'free_energies': free_energies
        }


"""
======= Modular Plotter class =======
"""

class Plotter:
    """
    Dedicated class to handle plotting and visualization of analysis results
    Create an instance specifying subplot grid, then call plotting methods to populate the axes
    """
    def __init__(self, rows: int=1, cols: int=1, figsize: tuple=(12,8), **kwargs):
        self.fig, self.axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False, **kwargs)

    def show(self):
        self.fig.tight_layout()
        plt.show()

    def _apply_styles(self, ax: plt.Axes, title: str, title_font: dict, xlabel: str, ylabel: str, label_font: dict, tick_font: dict):
        """Helper function to apply common styles to an axis."""
        ax.set_title(title, **(title_font or {}))
        ax.set_xlabel(xlabel, **(label_font or {}))
        ax.set_ylabel(ylabel, **(label_font or {}))
        
        if tick_font:
            tick_kwargs = tick_font.copy()
            if 'size' in tick_kwargs:
                tick_kwargs['labelsize'] = tick_kwargs.pop('size')
            ax.tick_params(axis='both', **tick_kwargs)

    def plot_network_graph(self, system: System, layout: str='lattice', figsize: tuple = (10,8), title: str='', ax=None):
        """ Displays nice NetworkX graph representation of the system """
        plt.sca(ax)
        G = system.to_networkx()

        # Choose layout
        if layout == 'lattice':
            grid_size = int(np.ceil(np.sqrt(system.N)))
            pos = {i: (i % grid_size, grid_size - 1 - (i // grid_size)) for i in range(system.N)}
        elif layout == 'spiral':
            pos = nx.spiral_layout(G)
        elif layout == 'spring':
            pos = nx.spring_layout(G)
        elif layout == 'multipartite':
            pos = nx.multipartite_layout(G)
        elif layout == 'arf':
            pos = nx.arf_layout(G)
        else:
            raise ValueError('Unknown layout type. Please use ["lattice", "spiral", "spring", "multipartite", "arf"].')
        
        # set title
        ax.set_title(title)

        # Node Properties
        node_colors = ['green' if state == +1 else 'lightblue' for state in system.current_state]
        node_labels = {i: f'{i+1}\n({system.current_state[i]:+d})' for i in range(system.N)}

        # Edge Properties
        edges = G.edges()
        pos_edges = [(u,v) for u,v in edges if G[u][v]['weight'] > 0]
        neg_edges = [(u,v) for u,v in edges if G[u][v]['weight'] < 0]
        pos_weights = [G[u][v]['weight'] for u,v in pos_edges]
        neg_weights = [abs(G[u][v]['weight']) for u,v in neg_edges]

        # Draw graph components
        nx.draw_networkx_nodes(G,
                               pos,
                               ax=ax,
                               node_color=node_colors,
                               node_size=2500,
                               edgecolors='black')
        nx.draw_networkx_labels(G,
                                pos,
                                labels=node_labels,
                                font_size=18,
                                font_weight='bold')
        
        min_width, max_width = 1.0, 5.0
        if pos_weights:
            scaled_pos = min_width + (max_width - min_width) * (np.array(pos_weights) - min(pos_weights)) / (max(pos_weights) - min(pos_weights)) if len(set(pos_weights)) > 1 else [2.0] * len(pos_weights)
            nx.draw_networkx_edges(G,
                                   pos,
                                   edgelist=pos_edges,
                                   width=scaled_pos,
                                   edge_color='black',
                                   alpha=1.0)
        if neg_weights:
            scaled_neg = min_width + (max_width - min_width) * (np.array(neg_weights) - min(neg_weights)) / (max(neg_weights) - min(neg_weights)) if len(set(neg_weights)) > 1 else [2.0] * len(neg_weights)
            nx.draw_networkx_edges(G,
                                   pos,
                                   edgelist=neg_edges,
                                   width=scaled_neg,
                                   edge_color='red',
                                   alpha=1.0)
        
    # ---- BasicEnergyAnalyzer Plots ----
    def plot_energy_distribution(self, ax: plt.Axes, analyzer: BasicEnergyAnalyzer, title: str = 'Energy Distribution', title_font: dict = None, label_font: dict = None, tick_font: dict = None):
        plt.sca(ax)
        results = analyzer.analyze_energy_distribution()
        unique_energies, counts = np.unique(results['all_energies'], return_counts=True)
        bar_width = np.min(np.diff(np.sort(unique_energies))) * 0.2 if len(unique_energies) > 1 else 0.8
        ax.bar(unique_energies, counts, width=bar_width, alpha=0.7, color='blue', edgecolor='black')
        
        self._apply_styles(ax, title, title_font, 'Hamiltonian Energy Level', 'Number of Configuration', label_font, tick_font)
        ax.grid(True, alpha=0.5)

    def plot_energy_landscape(self, ax: plt.Axes, analyzer: BasicEnergyAnalyzer, ordered: bool = True, title: str = 'Energy Landscape', title_font: dict = None, label_font: dict = None, tick_font: dict = None):
        plt.sca(ax)
        results = analyzer.analyze_energy_distribution()
        energies = results['all_energies']
        xlabel = 'Configuration Index'
        if ordered:
            energies = np.sort(energies)
            xlabel += ' (Sorted by Energy)'
        
        ax.plot(np.arange(len(energies)), energies, '-o', color='purple', alpha=0.6, markersize=8, linewidth = 3)
        self._apply_styles(ax, title, title_font, xlabel, 'Energy (H)', label_font, tick_font)
        ax.grid(True, alpha=0.5)


    def plot_boltzmann_distribution(self, ax: plt.Axes, analyzer: BasicEnergyAnalyzer, ordered: bool = True, log_scale: bool = False, title: str = 'Boltzmann Distribution', title_font: dict = None, label_font: dict = None, tick_font: dict = None, legend_font: dict = None):
        plt.sca(ax)
        colors = plt.cm.viridis(np.linspace(0, 1, len(analyzer.temperatures)))
        for temp, color in zip(analyzer.temperatures, colors):
            results = analyzer.analyze_boltzmann_distribution(temperature=temp, ordered=ordered)
            ax.plot(results['config_indices'], results['probabilities'], '-o', color=color, alpha=0.6, markersize=8, linewidth = 3, label=f'T={temp:.2f}')
        
        xlabel = 'Configuration Index'
        if ordered:
            xlabel += ' (Sorted by Energy)'
        
        self._apply_styles(ax, title, title_font, xlabel, 'Boltzmann Probability', label_font, tick_font)
        
        ax.legend(**(legend_font or {}))
        if log_scale:
            ax.set_yscale('log')
        ax.grid(True, which="both", alpha=0.5)

     # ---- FrustrationAnalyzer Plots ----

    def plot_property_vs_frustration(self, ax: plt.Axes, results: dict, prop: str = 'free_energy'):
        """Plots a property (e.g., free energy) as a function of frustration fraction."""
        plt.sca(ax)
        ff, temps = results['frustration_fractions'], results['temperatures']
        means, sds = results[f'{prop}_means'], results[f'{prop}_sds']
        ylabel = prop.replace('_', ' ').title()
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(temps)))
        for j, (temp, color) in enumerate(zip(temps, colors)):
            ax.errorbar(ff, means[:, j], yerr=sds[:, j], label=f'T={temp:.2f}', color=color, marker='o', capsize=3)
        ax.set(xlabel='Frustration Fraction', ylabel=ylabel, title=f'{ylabel} vs Frustration')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def plot_property_vs_temperature(self, ax: plt.Axes, results: dict, prop: str = 'magnetization'):
        """Plots a property (e.g., magnetization) as a function of temperature."""
        plt.sca(ax)
        ff, temps = results['frustration_fractions'], results['temperatures']
        means, sds = results[f'{prop}_means'], results[f'{prop}_sds']
        ylabel = prop.replace('_', ' ').title()

        colors = plt.cm.plasma(np.linspace(0, 1, len(ff)))
        for i, (frust, color) in enumerate(zip(ff, colors)):
            ax.errorbar(temps, means[i, :], yerr=sds[i, :], label=f'ff={frust:.2f}', color=color, marker='s', capsize=3)
        ax.set(xlabel='Temperature', ylabel=ylabel, title=f'{ylabel} vs Temperature')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    def plot_phase_diagram(self, ax: plt.Axes, results: dict, prop: str = 'free_energy'):
        """Plots a 2D phase diagram of a property vs. temperature and frustration."""
        plt.sca(ax)
        ff, temps = results['frustration_fractions'], results['temperatures']
        data = results[f'{prop}_means']
        label = prop.replace('_', ' ').title()
        
        im = ax.imshow(data, aspect='auto', origin='lower', extent=[temps[0], temps[-1], ff[0], ff[-1]])
        ax.set(xlabel='Temperature', ylabel='Frustration Fraction', title=f'{label} Phase Diagram')
        self.fig.colorbar(im, ax=ax, label=label)

    # ---- MonteCarlo Simulation Plots ----
    def plot_metropolis_evolution(self, ax_energy: plt.Axes, ax_magnetization: plt.Axes, simulation_results: list, temperatures: list, label_font: dict = None, tick_font: dict = None, legend_font: dict = None):
        """
        Plots the energy and magnetization evolution from a Monte Carlo simulation.
        
        Args:
            ax_energy (plt.Axes): The subplot for the energy plot.
            ax_magnetization (plt.Axes): The subplot for the magnetization plot.
            simulation_results (list): A list of dictionaries, where each dict contains 'energies' and 'magnetizations' arrays for a run.
            temperatures (list): A list of temperatures corresponding to each simulation run.
        """
        plt.sca(ax_energy)
        colors = plt.cm.plasma(np.linspace(0, 1, len(temperatures)))
        
        for i, temp in enumerate(temperatures):
            results = simulation_results[i]
            n_sweeps = len(results['energies'])
            x_vals = np.arange(n_sweeps)
            
            # Plot energy time series
            ax_energy.plot(x_vals, results['energies'], color=colors[i], label=f'T={temp:.2f}')
            
            # Plot magnetization time series (absolute value per site)
            abs_mag_per_site = np.abs(results['magnetizations']) / results['N']
            ax_magnetization.plot(x_vals, abs_mag_per_site, color=colors[i], label=f'T={temp:.2f}')

        # Style the energy plot
        self._apply_styles(ax_energy, '', {}, '', 'Energy', label_font, tick_font)
        ax_energy.grid(True, alpha=0.5)
        ax_energy.legend(**(legend_font or {}), loc='center right')

        # Style the magnetization plot
        self._apply_styles(ax_magnetization, '', {}, 'Sweep #', 'Absolute Magnetization / N', label_font, tick_font)
        ax_magnetization.grid(True, alpha=0.5)
        ax_magnetization.legend(**(legend_font or {}), loc='center right')
        
