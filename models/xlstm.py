import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class xLSTMCell(nn.Module):
    """
    Extended LSTM Cell implementation with advanced gating mechanisms.
    
    This implementation extends traditional LSTM with:
    1. Time-aware gating mechanism
    2. Extended memory representation
    3. Hierarchical update mechanism
    4. Gaussian noise injection for regularization
    5. Adaptive forget gate coupling
    """
    def __init__(self, input_size, hidden_size, bias=True, dropout=0.0, k_lipschitz=2.0):
        super(xLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.k_lipschitz = k_lipschitz
        
        # Traditional LSTM parameters
        self.W_ih = nn.Linear(input_size, 4 * hidden_size, bias=bias)
        self.W_hh = nn.Linear(hidden_size, 4 * hidden_size, bias=bias)
        
        # Time gate parameters
        self.W_time = nn.Linear(1, 3 * hidden_size, bias=bias)
        
        # Extended memory parameters
        self.W_im = nn.Linear(input_size, hidden_size, bias=bias)
        self.W_hm = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.W_tm = nn.Linear(1, hidden_size, bias=bias)
        
        # Hierarchical update parameters
        self.W_hc = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.W_zh = nn.Linear(hidden_size, hidden_size, bias=bias)
        
        # Oscillation parameters as learnable
        self.tau = nn.Parameter(torch.Tensor(hidden_size))
        self.r = nn.Parameter(torch.Tensor(hidden_size))
        self.s = nn.Parameter(torch.Tensor(hidden_size))
        
        # Layer normalization for better stability
        self.ln_cell = nn.LayerNorm(hidden_size)
        self.ln_hidden = nn.LayerNorm(hidden_size)
        self.ln_memory = nn.LayerNorm(hidden_size)
        
        # Attention mechanism to focus on relevant parts of the input
        self.attn = nn.Linear(input_size + hidden_size, 1)
        
        # Coupling factor for adaptive forget gate
        self.coupling = nn.Parameter(torch.Tensor(hidden_size))
        
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters with carefully chosen values"""
        # Initialize oscillation parameters
        nn.init.uniform_(self.tau, 10, 1000)  # Periods between 10 and 1000 timesteps
        nn.init.uniform_(self.r, 0, 1)  # Random shifts
        nn.init.uniform_(self.s, 0.05, 0.2)  # Active ratio between 5% and 20%
        
        # Xavier/Glorot initialization for better gradient flow
        for name, param in self.named_parameters():
            if 'weight' in name and 'ln' not in name and param.dim() > 1:
                nn.init.xavier_uniform_(param, gain=1.0)
            elif 'bias' in name and 'ln' not in name:
                nn.init.zeros_(param)
        
        # Initialize coupling factor
        nn.init.constant_(self.coupling, 0.5)
        
        # Enforce Lipschitz constant constraint for stability
        self._enforce_lipschitz()
    
    def _enforce_lipschitz(self):
        """Enforce Lipschitz constant for weight matrices to improve stability"""
        with torch.no_grad():
            for name, param in self.named_parameters():
                if 'weight' in name and 'ln' not in name and param.dim() > 1:
                    # Spectral normalization-like approach
                    u, s, v = torch.svd(param)
                    param.copy_(u @ torch.diag(torch.clamp(s, max=self.k_lipschitz)) @ v.t())
    
    def forward(self, input, time, hidden=None, noise_scale=0.01):
        """
        Forward pass for the xLSTM cell with advanced gating and regularization.
        
        Args:
            input: Input tensor of shape (batch_size, input_size)
            time: Time tensor of shape (batch_size, 1)
            hidden: Tuple of (h, c, m) where h, c, and m are tensors of shape (batch_size, hidden_size)
                   If None, initialized with zeros
            noise_scale: Scale of Gaussian noise injection for regularization
        
        Returns:
            h_new, c_new, m_new: New hidden state, cell state, and memory state
        """
        batch_size = input.size(0)
        
        # Initialize hidden state if not provided
        if hidden is None:
            h = torch.zeros(batch_size, self.hidden_size, device=input.device)
            c = torch.zeros(batch_size, self.hidden_size, device=input.device)
            m = torch.zeros(batch_size, self.hidden_size, device=input.device)
        else:
            h, c, m = hidden
        
        # Attention mechanism to focus on relevant parts of input
        attn_input = torch.cat([input, h], dim=1)
        attn_weights = torch.sigmoid(self.attn(attn_input))
        
        # Apply attention to input
        attended_input = input * attn_weights
        
        # Apply dropout for regularization
        if self.training and self.dropout > 0:
            attended_input = F.dropout(attended_input, p=self.dropout)
        
        # Traditional LSTM gates computation with attended input
        gates = self.W_ih(attended_input) + self.W_hh(h)
        i, f, g, o = gates.chunk(4, dim=1)
        
        # Time-aware modulation
        time_gates = self.W_time(time)
        t_i, t_f, t_o = time_gates.chunk(3, dim=1)
        
        # Apply sigmoid activation to gates
        i = torch.sigmoid(i + t_i)  # Input gate
        f = torch.sigmoid(f + t_f + self.coupling * h)  # Forget gate with coupling
        g = torch.tanh(g)  # Cell update
        o = torch.sigmoid(o + t_o)  # Output gate
        
        # LSTM update
        c_next = f * c + i * g
        
        # Extended memory update with time awareness
        m_update = torch.tanh(self.W_im(input) + self.W_hm(h) + self.W_tm(time))
        m_next = torch.sigmoid(self.W_hc(c_next)) * m + (1 - torch.sigmoid(self.W_hc(c_next))) * m_update
        
        # Apply layer normalization for stability
        c_next = self.ln_cell(c_next)
        m_next = self.ln_memory(m_next)
        
        # Oscillation mechanism based on time
        phi = (((time - self.r) % self.tau) / self.tau).expand_as(h)
        k = 0.5 * (1 + torch.cos(2 * math.pi * phi / self.s))
        k = torch.clamp(k, 0, 1)
        
        # Hierarchical hidden state update with memory influence
        z = torch.sigmoid(self.W_zh(m_next))
        h_lstm = o * torch.tanh(c_next) 
        h_next = z * (k * h_lstm + (1 - k) * h) + (1 - z) * torch.tanh(m_next)
        
        # Apply layer normalization
        h_next = self.ln_hidden(h_next)
        
        # Add controlled Gaussian noise for regularization
        if self.training:
            h_next = h_next + torch.randn_like(h_next) * noise_scale
        
        return h_next, c_next, m_next


class xLSTM(nn.Module):
    """
    xLSTM layer that stacks multiple xLSTMCell layers with residual connections.
    """
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, 
                 batch_first=True, dropout=0.0, residual=True):
        super(xLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.residual = residual
        
        # Create a list of xLSTM cells
        self.cell_list = nn.ModuleList()
        for i in range(num_layers):
            layer_input_size = input_size if i == 0 else hidden_size
            self.cell_list.append(xLSTMCell(
                input_size=layer_input_size, 
                hidden_size=hidden_size, 
                bias=bias,
                dropout=dropout if i < num_layers - 1 else 0.0,
                k_lipschitz=2.0 / num_layers  # Scale Lipschitz constant by depth
            ))
        
        # Input projection for residual connections
        if residual and input_size != hidden_size:
            self.input_proj = nn.Linear(input_size, hidden_size)
        else:
            self.input_proj = None
    
    def forward(self, input, time, hidden=None, noise_scale=0.01):
        """
        Forward pass for the xLSTM.
        
        Args:
            input: Input tensor of shape (batch_size, seq_len, input_size) if batch_first=True
                  else (seq_len, batch_size, input_size)
            time: Time tensor of shape (batch_size, seq_len, 1) if batch_first=True
                  else (seq_len, batch_size, 1)
            hidden: Tuple of (h_0, c_0, m_0) where each is a tensor of shape 
                   (num_layers, batch_size, hidden_size)
                   If None, initialized with zeros
        
        Returns:
            output: Output tensor of shape (batch_size, seq_len, hidden_size) if batch_first=True
                   else (seq_len, batch_size, hidden_size)
            h_n, c_n, m_n: Hidden state, cell state, and memory state for the last step
        """
        # Adjust dimensions if not batch_first
        if not self.batch_first:
            input = input.transpose(0, 1)
            time = time.transpose(0, 1)
        
        batch_size = input.size(0)
        seq_len = input.size(1)
        
        # Initialize hidden state if not provided
        if hidden is None:
            h = [torch.zeros(batch_size, self.hidden_size, device=input.device) for _ in range(self.num_layers)]
            c = [torch.zeros(batch_size, self.hidden_size, device=input.device) for _ in range(self.num_layers)]
            m = [torch.zeros(batch_size, self.hidden_size, device=input.device) for _ in range(self.num_layers)]
        else:
            h, c, m = hidden
            h = [h[i] for i in range(self.num_layers)]
            c = [c[i] for i in range(self.num_layers)]
            m = [m[i] for i in range(self.num_layers)]
        
        # Project input for residual connection if needed
        if self.input_proj is not None:
            input_proj = self.input_proj(input)
        
        # Process each time step
        outputs = []
        for t in range(seq_len):
            # Forward through each layer
            input_t = input[:, t, :]
            time_t = time[:, t, :]
            
            # Decaying noise scale for better training
            step_noise = noise_scale * (1.0 - t / seq_len) if self.training else 0.0
            
            # Process through layers
            for layer in range(self.num_layers):
                # Apply noise to input
                if layer == 0 and self.training:
                    input_t = input_t + torch.randn_like(input_t) * step_noise * 0.1
                
                # Process through cell
                h[layer], c[layer], m[layer] = self.cell_list[layer](
                    input_t, time_t, (h[layer], c[layer], m[layer]), step_noise
                )
                
                # Prepare input for next layer
                input_t = h[layer]
                
                # Apply residual connection
                if self.residual and layer > 0:
                    h[layer] = h[layer] + h[layer-1]
            
            # Add output with residual connection from input if applicable
            if self.residual and self.input_proj is not None:
                outputs.append(h[-1] + input_proj[:, t, :])
            else:
                outputs.append(h[-1])
        
        # Stack outputs
        outputs = torch.stack(outputs, dim=1)
        
        # Stack h, c, and m for return
        h_n = torch.stack(h, dim=0)
        c_n = torch.stack(c, dim=0)
        m_n = torch.stack(m, dim=0)
        
        # Adjust dimensions if not batch_first
        if not self.batch_first:
            outputs = outputs.transpose(0, 1)
        
        return outputs, (h_n, c_n, m_n)


class MultiHeadTemporalAttention(nn.Module):
    """
    Multi-head self-attention mechanism specialized for temporal data.
    """
    def __init__(self, hidden_size, num_heads=4, dropout=0.1):
        super(MultiHeadTemporalAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        assert self.head_dim * num_heads == hidden_size, "hidden_size must be divisible by num_heads"
        
        # Projections for Q, K, V
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
        # Time encoding
        self.time_embed = nn.Linear(1, hidden_size)
        
        # Relative position encoding
        self.max_seq_len = 200  # Maximum expected sequence length
        self.pos_embed = nn.Parameter(torch.Tensor(2 * self.max_seq_len - 1, self.head_dim))
        nn.init.xavier_normal_(self.pos_embed)
        
        # Normalization and dropout
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout)
        )
        
    def _relative_position_index(self, seq_len):
        """Calculate relative position indices for positional encoding"""
        pos_indices = torch.arange(seq_len, device=self.pos_embed.device)
        rel_indices = pos_indices.unsqueeze(1) - pos_indices.unsqueeze(0)
        # Shift to [0, 2*max_seq_len-1]
        rel_indices += self.max_seq_len - 1
        # Clip to valid range
        rel_indices = torch.clamp(rel_indices, 0, 2 * self.max_seq_len - 2)
        return rel_indices
        
    def forward(self, x, time=None):
        """
        Apply temporal attention to the sequence.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size)
            time: Optional time tensor of shape (batch_size, seq_len, 1)
        
        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_size)
        """
        # Save residual
        residual = x
        
        batch_size, seq_len, hidden_size = x.size()
        
        # Apply layer normalization
        x = self.layer_norm1(x)
        
        # Incorporate time information if provided
        if time is not None:
            time_features = self.time_embed(time)
            x = x + time_features
        
        # Multi-head attention projections
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Transpose for attention computation
        q = q.transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        k = k.transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        v = v.transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Add relative positional encoding
        rel_pos_indices = self._relative_position_index(seq_len)
        rel_pos_embedding = self.pos_embed[rel_pos_indices]
        
        # Add positional bias to attention scores
        pos_scores = torch.matmul(q.unsqueeze(-2), rel_pos_embedding.transpose(-2, -1)).squeeze(-2)
        scores = scores + pos_scores
        
        # Causal masking for autoregressive processing
        mask = torch.triu(torch.ones(seq_len, seq_len, device=scores.device), diagonal=1)
        mask = mask.unsqueeze(0).unsqueeze(0)  # Add batch and head dims
        scores = scores.masked_fill(mask.bool(), -1e9)
        
        # Apply softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention weights to values
        context = torch.matmul(attn_weights, v)  # (batch_size, num_heads, seq_len, head_dim)
        context = context.transpose(1, 2).contiguous()  # (batch_size, seq_len, num_heads, head_dim)
        context = context.view(batch_size, seq_len, hidden_size)
        
        # Apply output projection and dropout
        output = self.out_proj(context)
        output = self.dropout(output)
        
        # Add residual connection
        output = output + residual
        
        # Feed-forward network with residual connection
        residual = output
        output = self.layer_norm2(output)
        output = self.ffn(output) + residual
        
        return output


class UncertaintyModule(nn.Module):
    """
    Module to estimate uncertainty in predictions.
    """
    def __init__(self, input_size, output_size):
        super(UncertaintyModule, self).__init__()
        
        # Network for mean estimation
        self.mean_net = nn.Sequential(
            nn.Linear(input_size, input_size // 2),
            nn.GELU(),
            nn.Linear(input_size // 2, output_size)
        )
        
        # Network for variance estimation (aleatoric uncertainty)
        self.var_net = nn.Sequential(
            nn.Linear(input_size, input_size // 2),
            nn.GELU(),
            nn.Linear(input_size // 2, output_size),
            nn.Softplus()  # Ensure positive variance
        )
    
    def forward(self, x):
        """
        Estimate mean and variance.
        
        Args:
            x: Input features of shape (batch_size, input_size)
        
        Returns:
            mean: Predicted mean values
            var: Predicted variance (uncertainty)
        """
        mean = self.mean_net(x)
        var = self.var_net(x) + 1e-6  # Add small constant for numerical stability
        
        return mean, var


class CryptoXLSTM(nn.Module):
    """
    Enhanced cryptocurrency price prediction model using xLSTM.
    Includes advanced attention mechanisms, uncertainty estimation, and deep regularization.
    """
    def __init__(self, input_size, hidden_size, num_layers=3, dropout=0.3, 
                 l2_reg=1e-5, uncertainty=True):
        super(CryptoXLSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.l2_reg = l2_reg
        self.uncertainty = uncertainty
        
        # Input projection with normalization
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.input_norm = nn.LayerNorm(hidden_size)
        self.input_dropout = nn.Dropout(dropout)
        
        # xLSTM layers with residual connections
        self.xlstm = xLSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            residual=True
        )
        
        # Multi-head temporal attention layers
        self.temporal_attention = nn.ModuleList([
            MultiHeadTemporalAttention(hidden_size, num_heads=8, dropout=dropout/2)
            for _ in range(2)
        ])
        
        # Feature extraction layers with residual connections
        self.feature_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size * 2, hidden_size),
                nn.LayerNorm(hidden_size)
            ) for _ in range(2)
        ])
        
        # Output modules
        if uncertainty:
            # Uncertainty-aware output
            self.uncertainty_module = UncertaintyModule(hidden_size, 14)  # 7 days * 2 values (open, close)
        else:
            # Standard output
            self.output_layers = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.LayerNorm(hidden_size // 2),
                nn.GELU(),
                nn.Dropout(dropout / 2),
                nn.Linear(hidden_size // 2, 14)  # 7 days * 2 values (open, close)
            )
        
        # Variational dropout for further regularization
        self.variational_dropout = nn.Dropout2d(dropout / 3)
        
        # Stochastic depth/dropout for advanced regularization
        self.stochastic_depth_prob = 0.1
        self.register_buffer('drop_path', torch.zeros(1))
        
        # Weight initialization
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using carefully chosen schemes for different components"""
        for name, param in self.named_parameters():
            if 'weight' in name and 'norm' not in name and 'xlstm' not in name:
                if 'temporal_attention' in name:
                    # Use scaled initialization for attention
                    nn.init.xavier_normal_(param, gain=0.6)
                elif len(param.shape) >= 2:
                    # Xavier/Glorot for regular layers
                    nn.init.xavier_normal_(param, gain=1.0)
                elif len(param.shape) == 1:
                    # Initialize biases with small positive values
                    nn.init.constant_(param, 0.01)
    
    def stochastic_depth(self, x, training=None):
        """Apply stochastic depth - randomly drop entire layers"""
        if training is None:
            training = self.training
            
        if not training or torch.rand(1).item() > self.stochastic_depth_prob:
            return x
        
        return torch.zeros_like(x)
    
    def forward(self, x, time):
        """
        Forward pass with enhanced processing.
        
        Args:
            x: Input features of shape (batch_size, seq_len, input_size)
            time: Time tensor of shape (batch_size, seq_len, 1)
        
        Returns:
            predictions: If uncertainty=False, predicted prices of shape (batch_size, 14)
                        If uncertainty=True, tuple of (mean, variance) each of shape (batch_size, 14)
        """
        # Apply input projection and normalization
        x = self.input_proj(x)
        x = self.input_norm(x)
        x = self.input_dropout(x)
        
        # Apply variational dropout channel-wise
        x = x.unsqueeze(3)  # Add an extra dimension for 2D dropout
        x = self.variational_dropout(x)
        x = x.squeeze(3)  # Remove the extra dimension
        
        # Run through xLSTM
        output, _ = self.xlstm(x, time)
        
        # Apply multi-head temporal attention layers with residual connections
        for attn_layer in self.temporal_attention:
            # Stochastic depth for regularization
            if self.training and torch.rand(1).item() < self.stochastic_depth_prob:
                continue
                
            # Apply attention
            residual = output
            attended = attn_layer(output, time)
            output = residual + attended
        
        # Take the last timestep output
        last_output = output[:, -1, :]
        
        # Apply feature extraction with residual connections
        feature_output = last_output
        for i, layer in enumerate(self.feature_layers):
            # Stochastic depth - randomly skip some layers during training
            if self.training and torch.rand(1).item() < self.stochastic_depth_prob:
                continue
                
            residual = feature_output
            feature_output = layer(feature_output)
            
            # Apply residual connection with layer-specific scaling
            scale = 1.0 - (i * 0.1)  # Gradually reduce residual contribution
            feature_output = feature_output + scale * residual
        
        # Apply output layers
        if self.uncertainty:
            # Return mean and variance for uncertainty-aware predictions
            mean, var = self.uncertainty_module(feature_output)
            return mean, var
        else:
            # Return standard predictions
            predictions = self.output_layers(feature_output)
            return predictions
    
    def get_l2_regularization_loss(self):
        """Calculate L2 regularization loss with parameter-specific scaling"""
        l2_loss = 0.0
        for name, param in self.named_parameters():
            # Apply stronger regularization to larger matrices
            if 'weight' in name and param.dim() > 1:
                scale = min(1.0, 10.0 / param.numel()) 
                l2_loss += scale * torch.norm(param, 2)
            else:
                l2_loss += 0.1 * torch.norm(param, 2)  # Less regularization for biases and 1D params
                
        return self.l2_reg * l2_loss
        
    def sample_predictions(self, x, time, num_samples=10):
        """
        Generate multiple prediction samples for uncertainty estimation.
        
        Args:
            x: Input features of shape (batch_size, seq_len, input_size)
            time: Time tensor of shape (batch_size, seq_len, 1)
            num_samples: Number of prediction samples to generate
            
        Returns:
            samples: Multiple prediction samples of shape (batch_size, num_samples, 14)
            mean: Mean prediction of shape (batch_size, 14)
            std: Standard deviation of predictions (uncertainty) of shape (batch_size, 14)
        """
        if not self.uncertainty:
            raise ValueError("This method requires uncertainty=True")
            
        samples = []
        
        # Generate multiple samples with dropout enabled
        self.train()  # Enable dropout
        for _ in range(num_samples):
            mean, var = self(x, time)
            # Sample from Gaussian distribution
            eps = torch.randn_like(mean)
            sample = mean + torch.sqrt(var) * eps
            samples.append(sample)
            
        self.eval()  # Restore evaluation mode
        
        # Stack samples
        samples = torch.stack(samples, dim=1)  # (batch_size, num_samples, 14)
        
        # Calculate mean and std
        mean = samples.mean(dim=1)  # (batch_size, 14)
        std = samples.std(dim=1)  # (batch_size, 14)
        
        return samples, mean, std