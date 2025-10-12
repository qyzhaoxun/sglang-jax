from typing import Tuple

import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.tree_util import register_pytree_node_class

from sgl_jax.srt.layers.attention.base_attn_backend import AttentionBackend
from sgl_jax.srt.layers.radix_attention import AttentionType, RadixAttention
from sgl_jax.srt.managers.schedule_batch import ModelWorkerBatch
from sgl_jax.srt.mem_cache.memory_pool import merge_kv
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sgl_jax.srt.utils.jax_utils import is_tpu_runtime


@jax.jit
def _kv_layer_sum(buf: jax.Array) -> jax.Array:
    # Per-layer KV (fused) checksum (or replace with sum(abs(buf)))
    return jnp.sum(buf)


@register_pytree_node_class
class NativeAttention(AttentionBackend):
    """Native Attention layer for variable-length sequences using ForwardBatch."""

    def __init__(
        self,
        num_attn_heads,
        num_kv_heads,
    ):
        self.num_heads = num_attn_heads
        if num_kv_heads is not None:
            self.num_kv_heads = num_kv_heads
        else:
            self.num_kv_heads = num_attn_heads
        # self.rngs = rngs

    def tree_flatten(self):
        children = ()
        aux_data = {"num_heads": self.num_heads, "num_kv_heads": self.num_kv_heads}
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(
            num_attn_heads=aux_data["num_heads"], num_kv_heads=aux_data["num_kv_heads"]
        )

    def get_forward_metadata(self, batch: ModelWorkerBatch):
        """Init the metadata for a forward pass and return it."""
        return None

    def __call__(
        self,
        q: jax.Array,
        k: jax.Array,
        v: jax.Array,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
    ):
        """
        Args:
            q, k, v: Input tensors of shape [total_tokens, hidden_size]
            forward_batch: ForwardBatch object containing seq_lens and batch_size
            is_causal: Whether to apply causal masking
        Returns:
            Tuple of (output tensor of shape [total_tokens, hidden_size], k, v)
        """
        k_buffer, v_buffer, kv_fused = self._get_and_update_kv_cache(
            k, v, forward_batch, layer.layer_id
        )

        if layer.scaling is None:
            scale = 1.0 / jnp.sqrt(layer.head_dim)
        else:
            scale = layer.scaling

        is_causal = True
        if (
            forward_batch.forward_mode == ForwardMode.DECODE
            or layer.attn_type == AttentionType.ENCODER_ONLY
        ):
            is_causal = False

        attn_output = forward_attention(
            q,
            k_buffer,
            v_buffer,
            forward_batch.seq_lens,
            forward_batch.cache_loc,
            forward_batch.extend_prefix_lens,
            forward_batch.extend_seq_lens,
            layer.q_head_num,
            layer.kv_head_num,
            scale,
            is_causal,
            forward_batch.forward_mode,
        )

        # Return full fused KV buffer for this layer so that caller can persist it outside JIT
        return attn_output, kv_fused

    def _get_and_update_kv_cache(
        self,
        k: jax.Array,
        v: jax.Array,
        forward_batch: ForwardBatch,
        layer_id: int,
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        """
        Get the kv cache from the forward batch.
        """
        layer_idx = layer_id - forward_batch.token_to_kv_pool.start_layer
        before_sum = _kv_layer_sum(forward_batch.token_to_kv_pool.kv_buffer[layer_idx])
        jax.debug.print("[KV-LEG:BEFORE] layer={} kv_sum={}", layer_id, before_sum)

        if is_tpu_runtime():
            if forward_batch.forward_mode == ForwardMode.EXTEND:
                forward_batch.token_to_kv_pool.set_kv_buffer(
                    layer_id, forward_batch.out_cache_loc, k, v, is_decode=False
                )
            else:
                forward_batch.token_to_kv_pool.set_kv_buffer(
                    layer_id, forward_batch.out_cache_loc, k, v, is_decode=True
                )
            # Use fused layer directly from pool; derive K/V views without extra merge
            fused_layer = forward_batch.token_to_kv_pool.get_fused_kv_buffer(layer_id)
            k, v = fused_layer[:, ::2, :], fused_layer[:, 1::2, :]
            fused_return = fused_layer
        else:
            try:
                loc = forward_batch.out_cache_loc
                lmin = jnp.min(loc) if loc.size > 0 else jnp.array(-1, jnp.int32)
                lmax = jnp.max(loc) if loc.size > 0 else jnp.array(-1, jnp.int32)
                loc_len = loc.shape[0]
                # Avoid dynamic arange: use fixed 128 with safe indexing
                idx128 = jnp.arange(128, dtype=jnp.int32)
                valid128 = idx128 < loc_len
                safe_idx128 = jnp.where(valid128, idx128, 0)
                loc_head128 = jnp.take(loc, safe_idx128) if loc_len > 0 else loc
                neg_cnt = jnp.sum(loc < 0)
                zero_cnt = jnp.sum(loc == 0)
                jax.debug.print(
                    "[KV-PATH] [KV-WRITE] NATIVE set_kv_buffer_legacy mode={m} layer={ly} loc={loc} loc_min={lmin} loc_max={lmax} neg_cnt={neg} zero_cnt={z} loc_len={ln} loc_head128={lh} k_shape={ks} v_shape={vs}",
                    m=int(forward_batch.forward_mode),
                    ly=layer_id,
                    loc=loc,
                    lmin=lmin,
                    lmax=lmax,
                    neg=neg_cnt,
                    z=zero_cnt,
                    ln=loc_len,
                    lh=loc_head128,
                    ks=k.shape,
                    vs=v.shape,
                )
            except Exception:
                pass
            updated_layer = forward_batch.token_to_kv_pool.set_kv_buffer_legacy(
                layer_id, forward_batch.out_cache_loc, k, v
            )
            # Functional style: treat updated_layer as authoritative fused buffer for this layer in this step
            # Derive K/V views for attention computation from fused buffer directly
            k = updated_layer[:, ::2, :]
            v = updated_layer[:, 1::2, :]
            # Return fused buffer directly for persistence outside JIT
            fused_return = updated_layer
        return k, v, fused_return

    @staticmethod
    def get_max_running_reqests(max_context_len: int, page_size: int) -> int:
        # native attention backend do not care the max running requests
        return 4096


# @partial(jax.jit, static_argnames=["num_heads", "num_kv_heads", "is_causal", "mode"])
def forward_attention(
    q: jax.Array,
    k_cache: jax.Array,
    v_cache: jax.Array,
    seq_lengths: jax.Array,
    loc: jax.Array,
    extend_prefix_lens: jax.Array,
    extend_seq_lens: jax.Array,
    num_heads,
    num_kv_heads,
    scale=None,
    is_causal=True,
    mode=ForwardMode.DECODE,
):
    """
    Forward pass using native JAX implementation with block-diagonal attention.
    This avoids padding while maintaining efficient matrix operations.

    Args:
        q: input token in decode mode, shape(batch_size, hidden_size), each batch has one token
        k_cache: prefix cache of key, shape(seq_len, hidden_size)
        v_cache: prefix cache of value, shape(seq_len, hidden_size)
        seq_lengths: sequence lengths of each batch
        loc: location of the key/value cache
        extend_prefix_lens: prefix lengths of each batch in extend mode
        extend_seq_lens: sequence lengths of each batch in extend mode
        num_heads: number of query heads
        num_kv_heads: number of key/value heads
        scale: scale for the attention weights
        seq_mask: boolean mask of shape [batch_size, total_prefix_len]

    Returns:
        Output tensor of shape[batch_size, hidden_size]
    """
    # [FA-ARGS] Consolidated arg diagnostics: full loc/seq/extend and shapes/dtypes/modes
    ep = (
        extend_prefix_lens
        if extend_prefix_lens is not None
        else jnp.array([], dtype=jnp.int32)
    )
    es = (
        extend_seq_lens
        if extend_seq_lens is not None
        else jnp.array([], dtype=jnp.int32)
    )

    loc_len = loc.shape[0]
    # Avoid dynamic arange: use fixed 128 with safe indexing
    idx128 = jnp.arange(128, dtype=jnp.int32)
    valid128 = idx128 < loc_len
    safe_idx128 = jnp.where(valid128, idx128, 0)
    loc_head128 = jnp.take(loc, safe_idx128) if loc_len > 0 else loc
    neg_cnt = jnp.sum(loc < 0)
    zero_cnt = jnp.sum(loc == 0)
    nonzero_mask = loc != 0
    any_nz = jnp.any(nonzero_mask)
    # Tail zero segment start: equals the number of non-zero elements (assume tail is all zeros)
    tail_zero_start = jnp.where(
        any_nz,
        jnp.sum(nonzero_mask.astype(jnp.int32)),
        jnp.array(0, jnp.int32),
    )
    tail_zero_len = jnp.maximum(0, loc_len - tail_zero_start)

    jax.debug.print(
        "[FA-ARGS] mode={md} is_causal={ic} | q.shape={qs} k.shape={ks} v.shape={vs} | heads=(q={nh},kv={nkv}) | seq_lengths(len={slen})={sl} sum={ss} | loc(len={llen}) head128={lh} neg_cnt={neg} zero_cnt={z} tail_zero_start={ts} tail_zero_len={tl} | extend_prefix_lens(len={eplen})={epv} | extend_seq_lens(len={eslen})={esv}",
        md=jnp.asarray(mode, jnp.int32),
        ic=is_causal,
        qs=q.shape,
        ks=k_cache.shape,
        vs=v_cache.shape,
        nh=num_heads,
        nkv=num_kv_heads,
        sl=seq_lengths,
        slen=seq_lengths.shape[0],
        ss=jnp.sum(seq_lengths),
        llen=loc_len,
        lh=loc_head128,
        neg=neg_cnt,
        z=zero_cnt,
        ts=tail_zero_start,
        tl=tail_zero_len,
        epv=ep,
        eplen=ep.shape[0],
        esv=es,
        eslen=es.shape[0],
    )

    cache_size = k_cache.shape[0]
    safe_loc = jnp.where(loc > 0, loc, cache_size)
    k_cache = jnp.take(k_cache, safe_loc, axis=0, mode="fill", fill_value=0)
    v_cache = jnp.take(v_cache, safe_loc, axis=0, mode="fill", fill_value=0)

    # Handle both 2D and 3D input formats for q
    if len(q.shape) == 2:
        # Traditional format: [num_tokens, hidden_size]
        num_tokens, hidden_size = q.shape
        head_dim = hidden_size // num_heads
        q_heads = q.reshape(num_tokens, num_heads, head_dim)
    else:
        # Already in multi-head format: [num_tokens, num_heads, head_dim]
        num_tokens, num_heads_input, head_dim = q.shape
        assert (
            num_heads_input == num_heads
        ), f"Expected {num_heads} heads, got {num_heads_input}"
        hidden_size = num_heads * head_dim  # Calculate hidden_size for proper reshaping
        q_heads = q

    # KV cache from get_kv_buffer is already in multi-head format: [cache_size, num_kv_heads, head_dim]
    k_heads = k_cache
    v_heads = v_cache

    # Transpose for efficient matrix operations
    # q: shape of (num_heads, num_tokens, head_dim)
    # k, v: shape of (total_prefix_len, num_heads, head_dim)
    if num_kv_heads != num_heads:
        # For GQA attention, we need to copy k and v heads to match the number of query heads
        num_copies = num_heads // num_kv_heads
        # Use repeat to copy k and v heads
        # [total_prefix_len, num_kv_heads, head_dim] -> [total_prefix_len, num_heads, head_dim]
        k_heads = jnp.repeat(k_heads, num_copies, axis=1)
        v_heads = jnp.repeat(v_heads, num_copies, axis=1)

    q_t = jnp.transpose(q_heads, (1, 0, 2))
    k_t = jnp.transpose(k_heads, (1, 0, 2))
    v_t = jnp.transpose(v_heads, (1, 0, 2))

    if scale is None:
        scale = 1.0 / jnp.sqrt(head_dim)
    attn_logits = jnp.einsum("hqd,hkd->hqk", q_t, k_t) * scale
    neg_inf = jnp.asarray(jnp.finfo(attn_logits.dtype).min, attn_logits.dtype)
    is_valid = loc > 0
    attn_logits = jnp.where(is_valid[jnp.newaxis, jnp.newaxis, :], attn_logits, neg_inf)

    key_valid_from_loc = (loc >= 1) & (loc < cache_size)
    jax.debug.print(
        "[CHK] mode={} K_from_loc={} K_from_seq_lengths={}",
        mode,
        key_valid_from_loc.sum(),
        jnp.sum(seq_lengths),
    )

    if mode == ForwardMode.EXTEND:
        attn_logits = _apply_extend_mask(
            attn_logits, seq_lengths, extend_prefix_lens, extend_seq_lens, is_causal
        )
    else:
        attn_logits = _apply_decode_mask(attn_logits, seq_lengths)

    # Softmax
    attn_weights = jax.nn.softmax(attn_logits, axis=-1)
    # Debug: whether NaNs appear after softmax
    jax.debug.print("[Attn] softmax_finite={f}", f=jnp.isfinite(attn_weights).all())

    attn_output = jnp.matmul(attn_weights, v_t)
    attn_output = jnp.transpose(attn_output, (1, 0, 2))
    return attn_output.reshape(num_tokens, hidden_size)


def _apply_extend_mask(
    attn_weights: jax.Array,
    seq_lengths: jax.Array,
    extend_prefix_lens: jax.Array,
    extend_seq_lens: jax.Array,
    is_causal: bool = True,
):
    """
    Applies a block-diagonal and optionally a causal mask in a unified,
    efficient way, correctly handling padding.
    """
    batch_size = seq_lengths.shape[0]
    _, query_len, key_len = attn_weights.shape

    # --- Create validity masks to handle padding ---
    q_valid_mask = jnp.arange(query_len) < jnp.sum(extend_seq_lens)
    k_valid_mask = jnp.arange(key_len) < jnp.sum(seq_lengths)

    # --- 1. Generate Batch IDs (Optimized) ---
    q_starts = jnp.cumsum(extend_seq_lens, dtype=jnp.int32) - extend_seq_lens
    q_batch_indicators = jnp.zeros(query_len, dtype=jnp.int32).at[q_starts].set(1)
    q_batch_ids = jnp.cumsum(q_batch_indicators, dtype=jnp.int32) - 1

    full_seq_lens = seq_lengths
    k_starts = jnp.cumsum(full_seq_lens, dtype=jnp.int32) - full_seq_lens
    k_batch_indicators = jnp.zeros(key_len, dtype=jnp.int32).at[k_starts].set(1)
    k_batch_ids = jnp.cumsum(k_batch_indicators, dtype=jnp.int32) - 1

    # --- 2. Create block-diagonal mask ---
    final_mask = q_batch_ids[:, None] == k_batch_ids[None, :]

    # --- 3. Optionally add causal mask ---
    if is_causal:
        q_starts_per_pos = q_starts[q_batch_ids]
        q_relative_positions = jnp.arange(query_len, dtype=jnp.int32) - q_starts_per_pos
        prefix_lens_per_pos = extend_prefix_lens[q_batch_ids]
        q_actual_positions = prefix_lens_per_pos + q_relative_positions

        k_starts_per_pos = k_starts[k_batch_ids]
        k_relative_positions = jnp.arange(key_len, dtype=jnp.int32) - k_starts_per_pos

        causal_mask = q_actual_positions[:, None] >= k_relative_positions[None, :]
        final_mask = final_mask & causal_mask

    # --- 4. Apply the final combined mask ---
    # Combine with validity masks to handle padding
    final_mask = final_mask & q_valid_mask[:, None] & k_valid_mask[None, :]

    mask_value = jnp.finfo(attn_weights.dtype).min
    final_mask = final_mask[None, :, :]
    return jnp.where(final_mask, attn_weights, mask_value)


def _apply_decode_mask(attn_weights: jax.Array, seq_lengths: jax.Array):
    """Create a sequence mask that ensures tokens only attend within their sequence."""
    _, query_len, key_len = attn_weights.shape
    num_seqs = len(seq_lengths)

    def create_decode_sequence_mask():
        total_prefix_len = key_len
        seq_starts = jnp.cumsum(jnp.concatenate([jnp.array([0]), seq_lengths[:-1]]))
        seq_ends = seq_starts + seq_lengths
        all_positions = jnp.arange(total_prefix_len)
        seq_mask = (all_positions[None, :] >= seq_starts[:, None]) & (
            all_positions[None, :] < seq_ends[:, None]
        )
        return seq_mask

    per_sequence_mask = create_decode_sequence_mask()
    final_mask = jnp.zeros((query_len, key_len), dtype=jnp.bool_)
    final_mask = final_mask.at[:num_seqs, :].set(per_sequence_mask)

    mask_value = jnp.finfo(attn_weights.dtype).min
    final_mask = final_mask[None, :, :]
    return jnp.where(final_mask, attn_weights, mask_value)
