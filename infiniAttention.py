from .modeling_qwen_transformers import apply_rotary_pos_emb


class InfiniAttention(Attention):
    def __init__(
        self,
        config: Qwen2MoeConfig,
        layer_idx: Optional[int] = None,
    ):
        super().__init__(config, layer_idx)

        self.beta = nn.Parameter(torch.randn(1))
        
        self.M = None
        self.z = None
        

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:


        # Initialize memory and normalization term 
        if self.M is None:
            self.M = torch.zeros(self.num_heads, self.head_dim, self.head_dim).to(hidden_states.device)
            self.z = torch.zeros(self.num_heads, self.head_dim).to(hidden_states.device)

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )


        memory_output = self._retrieve_from_memory(query_states)
        debug_print("Memory Output Shape:", memory_output.shape)
        # Update memory with current segment's key and value states
        self._update_memory(key_states, value_states)
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)


        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
        )

        combined_output = self._long_term_injection(attn_output, memory_output)

        # Prepare output for this segment
        combined_output = combined_output.transpose(1, 2).contiguous()
        combined_output = combined_output.view(bsz, q_len, self.hidden_size)
        final_output = self.o_proj(combined_output)

        return final_output, None, None

    def _retrieve_from_memory(self, Q):
        # Retrieve context from compressive memory using linear attention (Eq. 3)
        M_s_1 = torch.matmul(F.elu(Q) + 1, self.M)
        Z_s_1 = torch.matmul(F.elu(Q) + 1, self.z.unsqueeze(-1)) + 1e-8
        A_mem = M_s_1 / Z_s_1
        return A_mem

    def _update_memory(self, K, V, use_delta=False):
        if use_delta:
            V_retrieved = torch.matmul(F.elu(K).transpose(-2, -1) + 1, self.M) / (torch.matmul(F.elu(K).transpose(-2, -1) + 1, self.z.unsqueeze(-1)) + 1e-8)
            updated_M = self.M + torch.matmul(F.elu(K).transpose(-2, -1) + 1, V - V_retrieved)
        else:
            updated_M = self.M + torch.matmul(F.elu(K).transpose(-2, -1) + 1, V)
        
        updated_z = self.z + (F.elu(K) + 1).sum(dim=-2)
        self.M = updated_M.detach()
        self.z = updated_z.detach()
    def _long_term_injection(self, A_dot, A_mem):
        beta = torch.sigmoid(self.beta)
        A = beta * A_mem + (1 - beta) * A_dot
        return A
