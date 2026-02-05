### BA-solver Loss & inference implementation. Done ###

import torch
import math
from loss import mean_flat

def mean_flat(tensor):
    return tensor.mean(dim=list(range(1, tensor.ndim)))

class BALoss:
    def __init__(self, delta_lambda=50.0, dropout_prob=0.1, bidirectional_prob=0.5, chain_steps=8):
        self.delta_lambda = delta_lambda
        self.dropout_prob = dropout_prob
        self.bidirectional_prob = bidirectional_prob
        self.chain_steps = chain_steps  # K: chain length

        # Gauss–Legendre (K=3) Constants
        sqrt_06 = math.sqrt(0.6)
        
        # Time offsets coefficients (0 to 1)
        self.c1 = 0.5 * (1.0 - sqrt_06)
        self.c2 = 0.5  # Midpoint
        self.c3 = 0.5 * (1.0 + sqrt_06)

        # Normalized weights (sum to 1.0)
        self.w1 = 5.0 / 18.0
        self.w2 = 8.0 / 18.0
        self.w3 = 5.0 / 18.0

    def sample_schedule(self, x_data):
        B = x_data.shape[0]
        device, dtype = x_data.device, x_data.dtype

        t = torch.rand((B,), device=device, dtype=dtype)
        x_noise = torch.randn_like(x_data)

        t_view = t.view(B, *([1] * (x_data.ndim - 1)))
        xt = (1.0 - t_view) * x_data + t_view * x_noise
        
        return t, xt

    def sample_h(self, t, B):
        t32 = t.to(torch.float32)

        # A. Determine the direction (positive or negative step)
        if self.bidirectional_prob > 0:
            go_backward = torch.rand((B,), device=t.device) > self.bidirectional_prob
        else:
            go_backward = torch.ones((B,), device=t.device, dtype=torch.bool)

        # B. Calculation for maximum step size
        # Backward (h > 0): t -> 0, max_dist = t
        # Forward  (h < 0): t -> 1, max_dist = 1-t
        max_dist = torch.where(go_backward, t32, 1.0 - t32)
        max_dist = torch.clamp(max_dist, min=1e-5)

        # C. Truncated exponential distribution
        u = torch.rand_like(t32)
        m = torch.expm1(-self.delta_lambda * max_dist)
        delta32 = -torch.log1p(u * m) / self.delta_lambda
        delta32 = torch.minimum(delta32, max_dist)
        h32 = torch.where(go_backward, delta32, -delta32)
        h = h32.to(t.dtype)
        
        return h

    def __call__(self, base_model, side_model, x_data, model_kwargs=None):
        if model_kwargs is None:
            model_kwargs = {}
        y = model_kwargs.get("y")
        target_dtype = next(base_model.parameters()).dtype

        # 1. CFG Dropout
        if y is not None and self.dropout_prob > 0:
            drop = torch.rand(y.shape[0], device=y.device) < self.dropout_prob
            y = torch.where(drop, torch.tensor(1000, device=y.device), y)

        # 2. Initial States
        t, xt = self.sample_schedule(x_data)
        B = xt.shape[0]
        xt_curr = xt.to(target_dtype)
        t_curr = t.to(target_dtype)

        # ==========================================================
        # 3. Initial Backbone Computation (NFE #1)
        # ==========================================================
        with torch.no_grad():
            out_base = base_model(xt_curr, t_curr, y)
            if isinstance(out_base, tuple): v_base_curr = out_base[0]
            else: v_base_curr = out_base

        total_loss = 0.0

        # ==========================================================
        # 4. Chain Loop (K times)
        # ==========================================================
        for step_idx in range(self.chain_steps):
            
            # A. Chain step h
            h = self.sample_h(t_curr, B)

            # B. Preparing nodes for quadrature
            tau1 = self.c1 * h
            tau2 = self.c2 * h
            tau3 = self.c3 * h

            # ==========================================================
            # C. Calculation of the next state (xt_next, t_next) by Teacher model (base model)
            # ==========================================================
            with torch.no_grad():
                # SideNet for intermediate velocities
                v1 = v_base_curr + side_model(xt_curr, v_base_curr, t_curr, tau1, y)
                v2 = v_base_curr + side_model(xt_curr, v_base_curr, t_curr, tau2, y)
                v3 = v_base_curr + side_model(xt_curr, v_base_curr, t_curr, tau3, y)

                v_avg = self.w1 * v1 + self.w2 * v2 + self.w3 * v3
                h_view = h.view(-1, *([1] * (xt.ndim - 1)))
                xt_next = xt_curr.float() - v_avg.float() * h_view.float()
                t_next = t_curr.float() - h.float()
                t_next = torch.clamp(t_next, 0.0, 1.0)

                xt_next_in = xt_next.to(target_dtype) 
                t_next_in = t_next.to(target_dtype)

                # ==========================================================
                # D. Calculation for target velocity (NFE #2, #3 ... #K+1)
                # Key point: this target velocity also serves as the input of next loop
                # ==========================================================
                out_base_next = base_model(xt_next_in, t_next_in, y)
                if isinstance(out_base_next, tuple): v_base_next = out_base_next[0]
                else: v_base_next = out_base_next
            
            # ==========================================================
            # E. Student Matching (SideNet)
            # ==========================================================
            # Student predicts v to match v_base_next based on (xt_curr, v_base_curr) and h
            
            # SideNet predictions
            pred_offset = side_model(xt_curr, v_base_curr.detach(), t_curr, h, y)
            pred_v = v_base_curr.detach() + pred_offset

            # Target velocity
            target_v = v_base_next.detach()

            loss_step = mean_flat((pred_v - target_v) ** 2).mean()
            total_loss += loss_step

            # ==========================================================
            # F. Updating for the next loop
            # ==========================================================
            xt_curr = xt_next_in
            t_curr = t_next_in
            v_base_curr = v_base_next # Reuse
        
        final_loss = total_loss / self.chain_steps
        return final_loss, final_loss, torch.tensor(0.0)

def BA_bidirectional_anchor_sampler(
    base_model,
    side_model,
    latents,
    y,
    num_steps=20,
    cfg_scale=1.5,
    cfg_interval_start=1.0,
    cfg_interval_end=0.0,
    # - 0.3333 = Simpson's 3/8 
    # - 0.2764 = Gauss-Lobatto 
    # - 0.2500 = Chebyshev
    node_alpha=0.2763932, 
):
    """
    BA Sampler with Generalized 4-Point Quadrature.
    Supports CFG Interval (Start & End) for REPA-style alignment.
    """
    device, dtype = latents.device, latents.dtype
    B = latents.shape[0]
    target_dtype = next(base_model.parameters()).dtype

    # ============================================================
    # 0. Quadrature Weights Preparation (Generalized Newton-Cotes)
    # ============================================================
    s = 1.0 - 2.0 * node_alpha
    w_in = 1.0 / (3.0 * (1.0 - s**2))
    w_out = 0.5 - w_in
    weights = torch.tensor([w_out, w_in, w_in, w_out], device=device, dtype=target_dtype)
    
    if w_out < 0.01:
        print(f"Warning: node_alpha={node_alpha} is too close to edge, anchor weights are unstable ({w_out:.4f})")

    t_steps = torch.linspace(1.0, 0.0, num_steps + 1, device=device, dtype=dtype)
    x = latents

    if cfg_scale > 1.0:
        y_null = torch.full_like(y, 1000)
        y_in = torch.cat([y, y_null], dim=0)
    else:
        y_in = y

    group_size = 2 * B if cfg_scale > 1.0 else B
    side_batch_size = 3 * group_size 
    h_vec_buffer = torch.empty((side_batch_size,), device=device, dtype=target_dtype)

    def apply_cfg(v_chunk, t_val):
        """
        Determine whether to apply CFG based on the current timestep `t_val`.
        CFG is applied only when `t_val` falls within the interval (cfg_interval_end, cfg_interval_start].
        i.e., cfg_interval_end < t_val <= cfg_interval_start.
        """
        t_scalar = t_val.item() if isinstance(t_val, torch.Tensor) else t_val
        
        is_in_interval = (t_scalar > cfg_interval_end) and (t_scalar <= cfg_interval_start)
        current_scale = cfg_scale if is_in_interval else 1.0
        
        if cfg_scale > 1.0:
            v_c, v_u = v_chunk.chunk(2, dim=0)
            if current_scale != 1.0:
                return v_u + current_scale * (v_c - v_u)
            else:
                return v_c 
        return v_chunk

    # ============================================================
    # 1. Initialize v_start
    # ============================================================
    t_first = t_steps[0]
    if cfg_scale > 1.0:
        x_in = torch.cat([x, x], dim=0)
        t_in = torch.full((2 * B,), t_first, device=device, dtype=dtype)
    else:
        x_in = x
        t_in = torch.full((B,), t_first, device=device, dtype=dtype)

    with torch.no_grad():
        out_base = base_model(x_in.to(target_dtype), t_in.to(target_dtype), y_in)
        v_base_raw = out_base[0] if isinstance(out_base, tuple) else out_base
    
    v_curr_raw = v_base_raw 

    # ============================================================
    # Main loop
    # ============================================================
    for i in range(num_steps):
        t_curr = t_steps[i]
        t_next = t_steps[i + 1]
        h_step = t_next - t_curr 
        h_abs = abs(h_step)

        # --------------------------------------------------------
        # A. v_start preparation
        # --------------------------------------------------------
        v_start = apply_cfg(v_curr_raw, t_curr)
        
        if cfg_scale > 1.0:
            x_in = torch.cat([x, x], dim=0)
            t_in = torch.full((2 * B,), t_curr, device=device, dtype=dtype)
        else:
            x_in = x
            t_in = torch.full((B,), t_curr, device=device, dtype=dtype)

        # --------------------------------------------------------
        # B. Forward Probe
        # --------------------------------------------------------
        h_vec_buffer[0 : group_size]              = node_alpha * h_abs
        h_vec_buffer[group_size : 2*group_size]   = (1.0 - node_alpha) * h_abs
        h_vec_buffer[2*group_size : 3*group_size] = (1.0) * h_abs

        x_comb = torch.cat([x_in] * 3, dim=0)
        v_comb = torch.cat([v_curr_raw] * 3, dim=0)
        t_comb = torch.cat([t_in] * 3, dim=0)
        y_comb = torch.cat([y_in] * 3, dim=0) if cfg_scale > 1.0 else torch.cat([y] * 3, dim=0)

        with torch.no_grad():
            delta_v = side_model(x_comb.to(target_dtype), v_comb, t_comb.to(target_dtype), h_vec_buffer, y_comb)
            v_comb.add_(delta_v)
            
            v_chunk_alpha, v_chunk_1_alpha_fwd, v_chunk_1 = v_comb.chunk(3, dim=0)

            t_alpha = t_curr - node_alpha * h_abs
            t_1_alpha = t_curr - (1.0 - node_alpha) * h_abs
            t_end_pred = t_curr - h_abs

            v_alpha_fwd = apply_cfg(v_chunk_alpha, t_alpha)
            v_1_alpha_fwd = apply_cfg(v_chunk_1_alpha_fwd, t_1_alpha) 
            v_end_pred = apply_cfg(v_chunk_1, t_end_pred)

        # --------------------------------------------------------
        # C. Predictor Update (Pure Forward), Single-Anchor
        # --------------------------------------------------------
        weighted_v_pred = (
            weights[0] * v_start + 
            weights[1] * v_alpha_fwd + 
            weights[2] * v_1_alpha_fwd + 
            weights[3] * v_end_pred
        )
        
        delta_x_pred = weighted_v_pred * h_abs
        x_pred = x - delta_x_pred

        if i == num_steps - 1:
            return x_pred 

        # --------------------------------------------------------
        # D. Backbone Real v_end (Anchor)
        # --------------------------------------------------------
        if cfg_scale > 1.0:
            x_next_in = torch.cat([x_pred, x_pred], dim=0)
            t_next_in = torch.full((2 * B,), t_next, device=device, dtype=dtype)
        else:
            x_next_in = x_pred
            t_next_in = torch.full((B,), t_next, device=device, dtype=dtype)

        with torch.no_grad():
            out_end = base_model(x_next_in.to(target_dtype), t_next_in.to(target_dtype), y_in)
            v_end_raw = out_end[0] if isinstance(out_end, tuple) else out_end
            
            v_curr_raw = v_end_raw 
            v_end_real = apply_cfg(v_end_raw, t_next)

        # --------------------------------------------------------
        # E. Backward Probe for Refine
        # --------------------------------------------------------
        h_vec_buffer[0 : group_size] = (-node_alpha) * h_abs 

        if cfg_scale > 1.0:
            v_next_in = v_end_raw
            x_bwd_in = x_next_in 
            t_bwd_in = t_next_in 
            y_bwd_in = y_in      
        else:
            v_next_in = v_end_raw
            x_bwd_in = x_next_in
            t_bwd_in = t_next_in
            y_bwd_in = y_in

        with torch.no_grad():
            delta_v_bwd = side_model(
                x_bwd_in.to(target_dtype), 
                v_next_in, 
                t_bwd_in.to(target_dtype), 
                h_vec_buffer[:group_size], 
                y_bwd_in
            )
            v_1_alpha_raw_bwd = v_next_in + delta_v_bwd
            t_1_alpha_bwd = t_next + node_alpha * h_abs 

            v_1_alpha_bwd = apply_cfg(v_1_alpha_raw_bwd, t_1_alpha_bwd)

        # --------------------------------------------------------
        # F. Corrector Update (Hybrid Fusion)
        # --------------------------------------------------------
        weighted_v_corr = (
            weights[0] * v_start + 
            weights[1] * v_alpha_fwd + 
            weights[2] * v_1_alpha_bwd + 
            weights[3] * v_end_real
        )
        
        delta_x_final = weighted_v_corr * h_abs
        x = x - delta_x_final

    return x

def BA_single_anchor_sampler(
    base_model,
    side_model,
    latents,
    y,
    num_steps=20,
    nodes=2,
    cfg_scale=1.5,
    cfg_interval_start=1.0, 
    cfg_interval_end=0.0,
):
    """
    Single Anchor
    Single-Anchor Sampler with Multi-Node Probe.
    """
    device, dtype = latents.device, latents.dtype
    B = latents.shape[0]
    K = max(1, nodes)

    target_dtype = next(base_model.parameters()).dtype

    # Gauss–Legendre nodes
    if K == 1:
        xg, wg = torch.tensor([0.0]), torch.tensor([2.0])
    elif K == 2:
        xg = torch.tensor([-0.57735027, 0.57735027])
        wg = torch.tensor([1.0, 1.0])
    elif K == 3:
        xg = torch.tensor([-0.77459667, 0.0, 0.77459667])
        wg = torch.tensor([0.55555556, 0.88888889, 0.55555556])
    elif K == 4:
        xg = torch.tensor([-0.86113631, -0.33998104, 0.33998104, 0.86113631])
        wg = torch.tensor([0.34785485, 0.65214515, 0.65214515, 0.34785485])
    elif K == 5:
        xg = torch.tensor([-0.90617985, -0.53846931, 0.0, 0.53846931, 0.90617985])
        wg = torch.tensor([0.23692689, 0.47862867, 0.56888889, 0.47862867, 0.23692689])
    else:
        raise NotImplementedError

    xg, wg = xg.to(device, dtype), wg.to(device, dtype)
    nodes_unit = 0.5 * (xg + 1.0)
    weights = 0.5 * wg

    t_steps = torch.linspace(1.0, 0.0, num_steps + 1, device=device, dtype=dtype)
    x = latents

    if cfg_scale > 1.0:
        y_null = torch.full_like(y, 1000)

    for i in range(num_steps):
        t_curr = t_steps[i]
        t_next = t_steps[i + 1]
        h_step = t_next - t_curr 
        h_abs = abs(h_step)

        h_nodes = nodes_unit * h_abs
        w_nodes = weights * h_abs

        x_rep = x.repeat_interleave(K, dim=0) 
        t_rep = torch.full((B * K,), t_curr, device=device, dtype=target_dtype)
        
        # Pre-prepare y_rep for SideNet
        y_rep = y.repeat_interleave(K, dim=0)
        
        h_rep = h_nodes.repeat(B) 

        t_val = t_curr.item() if isinstance(t_curr, torch.Tensor) else t_curr
        use_cfg_current_step = (cfg_scale > 1.0) and \
                               (t_val > cfg_interval_end) and \
                               (t_val <= cfg_interval_start)

        # 1. Base Model Pass
        with torch.no_grad():
            if use_cfg_current_step:
                # ---------------- CFG Branch (Double Batch) ----------------
                x_base_in = torch.cat([x, x], dim=0)
                t_base_in = torch.full((2 * B,), t_curr, device=device, dtype=dtype)
                y_base_in = torch.cat([y, y_null], dim=0)
                
                # Base model forward
                out_base = base_model(x_base_in.to(target_dtype), t_base_in.to(target_dtype), y_base_in)
                if isinstance(out_base, tuple): v_base_raw = out_base[0]
                else: v_base_raw = out_base
                
                v_base_cond, v_base_uncond = v_base_raw.chunk(2, dim=0)
                
                v_base_cond_k = v_base_cond.repeat_interleave(K, dim=0)
                v_base_uncond_k = v_base_uncond.repeat_interleave(K, dim=0)
                
                # SideNet Inputs
                v_base_k_in = torch.cat([v_base_cond_k, v_base_uncond_k], dim=0)
                h_k_in = torch.cat([h_rep, h_rep], dim=0).to(target_dtype)
                x_k_in = torch.cat([x_rep, x_rep], dim=0).to(target_dtype)
                t_k_in = torch.cat([t_rep, t_rep], dim=0).to(target_dtype)
                
                # y input handling
                y_base_k_in = y_base_in.repeat_interleave(K, dim=0)

                # SideNet Forward
                delta_v = side_model(x_k_in, v_base_k_in, t_k_in, h_k_in, y_base_k_in)
                
                v_total = v_base_k_in + delta_v
                v_c, v_u = v_total.chunk(2, dim=0)
                
                v = v_u + cfg_scale * (v_c - v_u)
                
            else:
                # ---------------- Non-CFG Branch (Single Batch) ----------------
                # This runs when cfg_scale=1.0 OR when t is outside the CFG interval
                out_base = base_model(x.to(target_dtype), torch.full((B,), t_curr, device=device, dtype=target_dtype), y)
                if isinstance(out_base, tuple): v_base_raw = out_base[0]
                else: v_base_raw = out_base
                
                v_base_k = v_base_raw.repeat_interleave(K, dim=0)
                
                # SideNet Forward (Single Batch)
                delta_v = side_model(x_rep.to(target_dtype), v_base_k, t_rep, h_rep.to(target_dtype), y_rep)
                v = v_base_k + delta_v

        v = v.view(B, K, *x.shape[1:])
        w_view = w_nodes.view(1, K, *([1] * (x.ndim - 1)))
        delta_x = (v * w_view).sum(dim=1)

        x = x - delta_x

    return x