import torch
import numpy as np
from typing import Optional, Dict, List, Tuple
from .BiNet import BiNet
from .nodes_layer import nodes_layer
from .exclusive_metadata import exclusive_metadata
from .inclusive_metadata import inclusive_metadata

class ExpMaxGPU:
    """
    GPU-accelerated Expectation Maximization class for MMSBM.
    
    This class handles the initialization and execution of the EM algorithm
    using PyTorch tensors on GPU for improved performance.
    
    Attributes
    ----------
    device : str
        Device to run computations on ('cuda' or 'cpu')
    BiNet : BiNet
        The bipartite network object
    theta_a : torch.Tensor
        Membership matrix for layer a (GPU tensor)
    theta_b : torch.Tensor
        Membership matrix for layer b (GPU tensor)
    pkl : torch.Tensor
        Probability matrix between groups (GPU tensor)
    omega : torch.Tensor
        Expectation parameters matrix (GPU tensor)
    qka_exclusive : Dict[str, torch.Tensor]
        Dictionary of probability matrices for exclusive metadata (GPU tensors)
    q_k_tau_inclusive : Dict[str, torch.Tensor]
        Dictionary of probability matrices for inclusive metadata (GPU tensors)
    zeta_inclusive : Dict[str, torch.Tensor]
        Dictionary of membership factors for inclusive metadata (GPU tensors)
    training_links : torch.Tensor
        Training links as GPU tensor
    training_labels : torch.Tensor
        Training labels as GPU tensor
    observed_nodes_a : torch.Tensor
        Observed nodes in layer a as GPU tensor
    observed_nodes_b : torch.Tensor
        Observed nodes in layer b as GPU tensor
    non_observed_nodes_a : torch.Tensor
        Non-observed nodes in layer a as GPU tensor
    non_observed_nodes_b : torch.Tensor
        Non-observed nodes in layer b as GPU tensor
    """
    
    def __init__(self, BiNet: BiNet, device: str = 'cuda'):
        """
        Initialize the GPU-accelerated EM algorithm.
        
        Parameters
        ----------
        BiNet : BiNet
            The bipartite network object
        device : str, default='cuda'
            Device to run computations on ('cuda' or 'cpu')
        """
        self.device = device
        self.BiNet = BiNet
        
        # Initialize tensors as None - they will be set during init_EM
        self.theta_a = None
        self.theta_b = None
        self.pkl = None
        self.omega = None
        self.qka_exclusive = {}
        self.q_k_tau_inclusive = {}
        self.zeta_inclusive = {}
        self.training_links = None
        self.training_labels = None
        self.observed_nodes_a = None
        self.observed_nodes_b = None
        self.non_observed_nodes_a = None
        self.non_observed_nodes_b = None
        
        # Cache for tensor conversions
        self._tensor_cache = {}
    
    def _to_tensor(self, x, key: str = None) -> torch.Tensor:
        """
        Convert numpy array to torch tensor and cache it.
        
        Parameters
        ----------
        x : np.ndarray or torch.Tensor
            Input array or tensor
        key : str, optional
            Cache key for the tensor
            
        Returns
        -------
        torch.Tensor
            Tensor on the specified device
        """
        if key and key in self._tensor_cache:
            return self._tensor_cache[key]
        
        if isinstance(x, np.ndarray):
            tensor = torch.from_numpy(x).to(self.device)
        elif isinstance(x, torch.Tensor):
            tensor = x.to(self.device)
        else:
            tensor = torch.tensor(x, device=self.device)
        
        if key:
            self._tensor_cache[key] = tensor
        
        return tensor
    
    def _to_numpy(self, x: torch.Tensor) -> np.ndarray:
        """Convert torch tensor to numpy array."""
        return x.cpu().numpy()
    
    def init_EM(self, training: Optional[np.ndarray] = None):
        """
        Initialize the EM algorithm with GPU tensors.
        
        Parameters
        ----------
        training : np.ndarray, optional
            Training data. If None, uses all data.
        """
        # Set training data
        if training is not None:
            self.training_links = self._to_tensor(training[:, :2], 'training_links')
            self.training_labels = self._to_tensor(training[:, 2], 'training_labels')
        else:
            self.training_links = self._to_tensor(self.BiNet.df[['source_id', 'target_id']].values, 'training_links')
            self.training_labels = self._to_tensor(self.BiNet.df['labels_id'].values, 'training_labels')
        
        # Set observed and non-observed nodes
        self.observed_nodes_a = self._to_tensor(self.BiNet.observed_nodes_a, 'observed_nodes_a')
        self.observed_nodes_b = self._to_tensor(self.BiNet.observed_nodes_b, 'observed_nodes_b')
        self.non_observed_nodes_a = self._to_tensor(self.BiNet.non_observed_nodes_a, 'non_observed_nodes_a')
        self.non_observed_nodes_b = self._to_tensor(self.BiNet.non_observed_nodes_b, 'non_observed_nodes_b')
        
        # Initialize membership matrices
        self.theta_a = torch.rand(self.BiNet.nodes_a.N_nodes, self.BiNet.nodes_a.K, device=self.device)
        self.theta_b = torch.rand(self.BiNet.nodes_b.N_nodes, self.BiNet.nodes_b.K, device=self.device)
        
        # Normalize membership matrices
        self.theta_a /= self.theta_a.sum(dim=1, keepdim=True)
        self.theta_b /= self.theta_b.sum(dim=1, keepdim=True)
        
        # Initialize probability matrix
        self.pkl = torch.rand(self.BiNet.nodes_a.K, self.BiNet.nodes_b.K, self.BiNet.N_labels, device=self.device)
        self.pkl /= self.pkl.sum(dim=2, keepdim=True)
        
        # Initialize omega matrix
        self._init_omega()
        
        # Initialize metadata tensors
        self._init_metadata_tensors()
    
    def _init_omega(self):
        """Initialize the omega matrix."""
        Na = self.BiNet.nodes_a.N_nodes
        Nb = self.BiNet.nodes_b.N_nodes
        Ka = self.BiNet.nodes_a.K
        Kb = self.BiNet.nodes_b.K
        
        self.omega = torch.zeros((Na, Nb, Ka, Kb), device=self.device)
        
        # Compute initial omega values
        for i, j, label in zip(self.training_links[:, 0], self.training_links[:, 1], self.training_labels):
            i, j, label = int(i), int(j), int(label)
            self.omega[i, j] = self.pkl[:, :, label] * torch.outer(self.theta_a[i], self.theta_b[j])
            self.omega[i, j] /= (self.omega[i, j].sum() + 1e-16)
    
    def _init_metadata_tensors(self):
        """Initialize metadata-related tensors."""
        # Initialize exclusive metadata tensors
        for meta_name, meta in self.BiNet.nodes_a.meta_exclusives.items():
            K = self.BiNet.nodes_a.K
            N_att = meta.N_att
            self.qka_exclusive[meta_name] = torch.rand(K, N_att, device=self.device)
            self.qka_exclusive[meta_name] /= self.qka_exclusive[meta_name].sum(dim=1, keepdim=True)
        
        for meta_name, meta in self.BiNet.nodes_b.meta_exclusives.items():
            K = self.BiNet.nodes_b.K
            N_att = meta.N_att
            self.qka_exclusive[meta_name] = torch.rand(K, N_att, device=self.device)
            self.qka_exclusive[meta_name] /= self.qka_exclusive[meta_name].sum(dim=1, keepdim=True)
        
        # Initialize inclusive metadata tensors
        for meta_name, meta in self.BiNet.nodes_a.meta_inclusives.items():
            K = self.BiNet.nodes_a.K
            Tau = meta.Tau
            self.q_k_tau_inclusive[meta_name] = torch.rand(K, Tau, device=self.device)
            self.q_k_tau_inclusive[meta_name] /= self.q_k_tau_inclusive[meta_name].sum(dim=1, keepdim=True)
            self.zeta_inclusive[meta_name] = torch.rand(meta.N_att, Tau, device=self.device)
            self.zeta_inclusive[meta_name] /= self.zeta_inclusive[meta_name].sum(dim=1, keepdim=True)
        
        for meta_name, meta in self.BiNet.nodes_b.meta_inclusives.items():
            K = self.BiNet.nodes_b.K
            Tau = meta.Tau
            self.q_k_tau_inclusive[meta_name] = torch.rand(K, Tau, device=self.device)
            self.q_k_tau_inclusive[meta_name] /= self.q_k_tau_inclusive[meta_name].sum(dim=1, keepdim=True)
            self.zeta_inclusive[meta_name] = torch.rand(meta.N_att, Tau, device=self.device)
            self.zeta_inclusive[meta_name] /= self.zeta_inclusive[meta_name].sum(dim=1, keepdim=True)
    
    def EM_step(self, N_steps: int = 1):
        """
        Perform one or more EM steps.
        
        Parameters
        ----------
        N_steps : int, default=1
            Number of EM steps to perform
        """
        for _ in range(N_steps):
            # E-step: Update omega
            self._update_omega()
            
            # M-step: Update parameters
            self._update_theta()
            self._update_pkl()
            self._update_metadata_parameters()
    
    def _update_omega(self):
        """Update the omega matrix (E-step)."""
        for i, j, label in zip(self.training_links[:, 0], self.training_links[:, 1], self.training_labels):
            i, j, label = int(i), int(j), int(label)
            self.omega[i, j] = self.pkl[:, :, label] * torch.outer(self.theta_a[i], self.theta_b[j])
            self.omega[i, j] /= (self.omega[i, j].sum() + 1e-16)
    
    def _update_theta(self):
        """Update membership matrices (M-step)."""
        # Update theta_a
        new_theta_a = torch.zeros_like(self.theta_a)
        new_theta_a[self.observed_nodes_a] = torch.sum(self.omega[self.observed_nodes_a.unsqueeze(1), self.observed_nodes_b], dim=(1, 2))
        
        # Add metadata contributions for layer a
        for meta_name, meta in self.BiNet.nodes_a.meta_exclusives.items():
            meta_omega = self._compute_meta_omega_exclusive(meta, self.BiNet.nodes_a)
            new_theta_a += torch.sum(meta_omega, dim=1) * meta.lambda_val
        
        for meta_name, meta in self.BiNet.nodes_a.meta_inclusives.items():
            meta_omega = self._compute_meta_omega_inclusive(meta, self.BiNet.nodes_a)
            new_theta_a += torch.sum(meta_omega, dim=(1, 2)) * meta.lambda_val
        
        # Normalize
        new_theta_a /= (new_theta_a.sum(dim=1, keepdim=True) + 1e-16)
        
        # Handle cold starts
        if not self.BiNet.nodes_a._has_metas and len(self.non_observed_nodes_a) > 0:
            means = torch.mean(new_theta_a[self.observed_nodes_a], dim=0)
            new_theta_a[self.non_observed_nodes_a] = means
        
        self.theta_a = new_theta_a
        
        # Update theta_b (similar process)
        new_theta_b = torch.zeros_like(self.theta_b)
        new_theta_b[self.observed_nodes_b] = torch.sum(self.omega[self.observed_nodes_a, self.observed_nodes_b.unsqueeze(1)], dim=(0, 2))
        
        # Add metadata contributions for layer b
        for meta_name, meta in self.BiNet.nodes_b.meta_exclusives.items():
            meta_omega = self._compute_meta_omega_exclusive(meta, self.BiNet.nodes_b)
            new_theta_b += torch.sum(meta_omega, dim=1) * meta.lambda_val
        
        for meta_name, meta in self.BiNet.nodes_b.meta_inclusives.items():
            meta_omega = self._compute_meta_omega_inclusive(meta, self.BiNet.nodes_b)
            new_theta_b += torch.sum(meta_omega, dim=(1, 2)) * meta.lambda_val
        
        # Normalize
        new_theta_b /= (new_theta_b.sum(dim=1, keepdim=True) + 1e-16)
        
        # Handle cold starts
        if not self.BiNet.nodes_b._has_metas and len(self.non_observed_nodes_b) > 0:
            means = torch.mean(new_theta_b[self.observed_nodes_b], dim=0)
            new_theta_b[self.non_observed_nodes_b] = means
        
        self.theta_b = new_theta_b
    
    def _update_pkl(self):
        """Update probability matrix pkl (M-step)."""
        Ka = self.BiNet.nodes_a.K
        Kb = self.BiNet.nodes_b.K
        N_labels = self.BiNet.N_labels
        
        new_pkl = torch.zeros((Ka, Kb, N_labels), device=self.device)
        
        # Compute sum over observed links for each label
        for label in range(N_labels):
            mask = (self.training_labels == label)
            if mask.sum() > 0:
                new_pkl[:, :, label] = torch.sum(self.omega[self.training_links[mask, 0], self.training_links[mask, 1]], dim=0)
        
        # Normalize
        suma = torch.sum(new_pkl, dim=2, keepdim=True)
        new_pkl /= (suma + 1e-16)
        
        self.pkl = new_pkl
    
    def _update_metadata_parameters(self):
        """Update metadata-related parameters (M-step)."""
        # Update exclusive metadata parameters
        for meta_name, meta in self.BiNet.nodes_a.meta_exclusives.items():
            self._update_qka_exclusive(meta, self.BiNet.nodes_a)
        
        for meta_name, meta in self.BiNet.nodes_b.meta_exclusives.items():
            self._update_qka_exclusive(meta, self.BiNet.nodes_b)
        
        # Update inclusive metadata parameters
        for meta_name, meta in self.BiNet.nodes_a.meta_inclusives.items():
            self._update_inclusive_metadata(meta, self.BiNet.nodes_a)
        
        for meta_name, meta in self.BiNet.nodes_b.meta_inclusives.items():
            self._update_inclusive_metadata(meta, self.BiNet.nodes_b)
    
    def _compute_meta_omega_exclusive(self, meta, nodes_layer):
        """Compute omega matrix for exclusive metadata."""
        N_nodes = nodes_layer.N_nodes
        N_att = meta.N_att
        K = nodes_layer.K
        
        omega = torch.zeros((N_nodes, N_att, K), device=self.device)
        
        for i, a in meta.links:
            i, a = int(i), int(a)
            omega[i, a] = self.theta_a[i] * self.qka_exclusive[meta.meta_name][:, a]
            omega[i, a] /= (omega[i, a].sum() + 1e-16)
        
        return omega
    
    def _compute_meta_omega_inclusive(self, meta, nodes_layer):
        """Compute omega matrix for inclusive metadata."""
        # This is a simplified version - you may need to adapt based on your specific implementation
        N_nodes = nodes_layer.N_nodes
        N_att = meta.N_att
        K = nodes_layer.K
        Tau = meta.Tau
        
        omega = torch.zeros((N_nodes, N_att, K, Tau), device=self.device)
        
        # Implementation depends on your specific inclusive metadata structure
        # This is a placeholder - you'll need to implement based on your needs
        
        return omega
    
    def _update_qka_exclusive(self, meta, nodes_layer):
        """Update qka parameters for exclusive metadata."""
        K = nodes_layer.K
        N_att = meta.N_att
        
        new_qka = torch.zeros((K, N_att), device=self.device)
        
        for k in range(K):
            for a in range(N_att):
                mask = (meta.links[:, 1] == a)
                if mask.sum() > 0:
                    new_qka[k, a] = torch.sum(self.theta_a[meta.links[mask, 0], k])
        
        # Normalize
        suma = torch.sum(new_qka, dim=1, keepdim=True)
        new_qka /= (suma + 1e-16)
        
        self.qka_exclusive[meta.meta_name] = new_qka
    
    def _update_inclusive_metadata(self, meta, nodes_layer):
        """Update inclusive metadata parameters."""
        # This is a placeholder - implement based on your specific needs
        pass
    
    def converges(self, tolerance: float = 1e-6) -> bool:
        """
        Check if the EM algorithm has converged.
        
        Parameters
        ----------
        tolerance : float, default=1e-6
            Convergence tolerance
            
        Returns
        -------
        bool
            True if converged, False otherwise
        """
        # This is a simplified convergence check
        # You may want to implement a more sophisticated convergence criterion
        return True  # Placeholder
    
    def get_accuracy(self, test_links: Optional[np.ndarray] = None) -> float:
        """
        Compute accuracy of predictions.
        
        Parameters
        ----------
        test_links : np.ndarray, optional
            Test links. If None, uses training links.
            
        Returns
        -------
        float
            Accuracy score
        """
        if test_links is not None:
            links = self._to_tensor(test_links[:, :2])
            labels = self._to_tensor(test_links[:, 2])
        else:
            links = self.training_links
            labels = self.training_labels
        
        # Compute predictions
        predictions = self.get_predicted_labels(links)
        
        # Compute accuracy
        correct = (predictions == labels).sum()
        total = len(labels)
        
        return float(correct / total)
    
    def get_predicted_labels(self, links: torch.Tensor) -> torch.Tensor:
        """
        Get predicted labels for given links.
        
        Parameters
        ----------
        links : torch.Tensor
            Links to predict labels for
            
        Returns
        -------
        torch.Tensor
            Predicted labels
        """
        predictions = torch.zeros(len(links), device=self.device)
        
        for idx, (i, j) in enumerate(links):
            i, j = int(i), int(j)
            probs = torch.sum(self.theta_a[i].unsqueeze(1) * self.theta_b[j].unsqueeze(0) * self.pkl, dim=(0, 1))
            predictions[idx] = torch.argmax(probs)
        
        return predictions
    
    def save_parameters(self, filepath: str):
        """
        Save EM parameters to file.
        
        Parameters
        ----------
        filepath : str
            Path to save parameters
        """
        torch.save({
            'theta_a': self.theta_a,
            'theta_b': self.theta_b,
            'pkl': self.pkl,
            'qka_exclusive': self.qka_exclusive,
            'q_k_tau_inclusive': self.q_k_tau_inclusive,
            'zeta_inclusive': self.zeta_inclusive
        }, filepath)
    
    def load_parameters(self, filepath: str):
        """
        Load EM parameters from file.
        
        Parameters
        ----------
        filepath : str
            Path to load parameters from
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.theta_a = checkpoint['theta_a']
        self.theta_b = checkpoint['theta_b']
        self.pkl = checkpoint['pkl']
        self.qka_exclusive = checkpoint['qka_exclusive']
        self.q_k_tau_inclusive = checkpoint['q_k_tau_inclusive']
        self.zeta_inclusive = checkpoint['zeta_inclusive'] 