from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np

import tensorflow as tf


class StateKey(object):
    CURRENT_IND = 'current_ind'
    ACTIVE_SEQ = 'active_seq'
    ACTIVE_LL = 'active_ll'
    ACTIVE_CACHE = 'active_cache'
    FINISHED_SEQ = 'finished_seq'
    FINISHED_LL = 'finished_ll'
    FINISHED_FLAG = 'finished_flag'


class BeamSearch(object):
    def __init__(self, f_callable, beam_size, batch_size, vocab_size, eos, normalization, max_length, padding=True):
        self.f_callable = f_callable
        self.beam_size = beam_size
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.eos = eos
        self.normalization = normalization
        self.max_length = max_length
        self.padding = padding
        self.dtype = tf.float32

    def __call__(self, initial_id, initial_cache):
        state, state_shapes = self._initial(initial_id, initial_cache)

        finished_state = tf.while_loop(
            self._cond, self._single_step, loop_vars=[state],
            shape_invariants=[state_shapes], parallel_iterations=1, back_prop=False)
        finished_state = finished_state[0]

        alive_seq = finished_state[StateKey.ACTIVE_SEQ]
        alive_log_probs = finished_state[StateKey.ACTIVE_LL]
        finished_seq = finished_state[StateKey.FINISHED_SEQ]
        finished_scores = finished_state[StateKey.FINISHED_LL]
        finished_flags = finished_state[StateKey.FINISHED_FLAG]

        finished_seq = tf.where(
            tf.reduce_any(finished_flags, 1), finished_seq, alive_seq)
        finished_scores = tf.where(
            tf.reduce_any(finished_flags, 1), finished_scores, alive_log_probs)
        return finished_seq, finished_scores

    def _initial(self, initial_ids, initial_cache):
        """
        Return initial state dictionary and its shape invariants.
        :param initial_ids:
        :param initial_cache:
        :return:
        """
        for key, value in initial_cache.items():
            for inner_value in tf.nest.flatten(value):
                if inner_value.dtype != self.dtype:
                    raise TypeError(
                        "initial_cache element for key '%s' has dtype %s that does not "
                        "match SequenceBeamSearch's dtype of %s. Value: %s" %
                        (key, value.dtype.name, self.dtype.name, inner_value))

        # Current loop index (starts at 0)
        cur_index = tf.constant(0)

        # Create alive sequence with shape [batch_size, beam_size, 1]
        alive_seq = _expand_to_beam_size(initial_ids, self.beam_size)
        alive_seq = tf.expand_dims(alive_seq, axis=2)
        if self.padding:
            alive_seq = tf.tile(alive_seq, [1, 1, self.max_length + 1])

        # Create tensor for storing initial log probabilities.
        # Assume initial_ids are prob 1.0
        initial_log_probs = tf.constant(
            [[0.] + [-float("inf")] * (self.beam_size - 1)], dtype=self.dtype)
        alive_log_probs = tf.tile(initial_log_probs, [self.batch_size, 1])

        # Expand all values stored in the dictionary to the beam size, so that each
        # beam has a separate cache.
        alive_cache = tf.nest.map_structure(
            lambda t: _expand_to_beam_size(t, self.beam_size), initial_cache)

        # Initialize tensor storing finished sequences with filler values.
        finished_seq = tf.zeros(tf.shape(alive_seq), tf.int32)

        # Set scores of the initial finished seqs to negative infinity.
        finished_scores = tf.ones([self.batch_size, self.beam_size],
                                  dtype=self.dtype) * -inf(self.dtype)

        # Initialize finished flags with all False values.
        finished_flags = tf.zeros([self.batch_size, self.beam_size], tf.bool)

        # Create state dictionary
        state = {
            StateKey.CURRENT_IND: cur_index,
            StateKey.ACTIVE_SEQ: alive_seq,
            StateKey.ACTIVE_LL: alive_log_probs,
            StateKey.ACTIVE_CACHE: alive_cache,
            StateKey.FINISHED_SEQ: finished_seq,
            StateKey.FINISHED_LL: finished_scores,
            StateKey.FINISHED_FLAG: finished_flags
        }
        return state

    def _cond(self, state):
        i = state[StateKey.CURRENT_IND]
        alive_log_probs = state[StateKey.ACTIVE_LL]
        finished_scores = state[StateKey.FINISHED_LL]
        finished_flags = state[StateKey.FINISHED_FLAG]

        not_at_max_decode_length = tf.less(i, self.max_length)

        # Calculate largest length penalty (the larger penalty, the better score).
        max_length_norm = _length_normalization(self.normalization, self.max_length,
                                                dtype=self.dtype)
        # Get the best possible scores from alive sequences.
        best_alive_scores = alive_log_probs[:, 0] / max_length_norm

        # Compute worst score in finished sequences for each batch element
        finished_scores *= tf.cast(finished_flags,
                                   self.dtype)  # set filler scores to zero
        lowest_finished_scores = tf.reduce_min(finished_scores, axis=1)

        # If there are no finished sequences in a batch element, then set the lowest
        # finished score to -INF for that element.
        finished_batches = tf.reduce_any(finished_flags, 1)
        lowest_finished_scores += ((1.0 -
                                    tf.cast(finished_batches, self.dtype)) *
                                   -inf(self.dtype))

        worst_finished_score_better_than_best_alive_score = tf.reduce_all(
            tf.greater(lowest_finished_scores, best_alive_scores)
        )

        return tf.logical_and(
            not_at_max_decode_length,
            tf.logical_not(worst_finished_score_better_than_best_alive_score)
        )

    def _single_step(self, state):
        topk_seq, topk_log_prob, topk_id, cache = self._next_prediction(state)

        finish_flag = tf.equal(topk_id, self.eos)

        active_state = self._get_new_alive_state(topk_seq, topk_log_prob,
                                                 finish_flag, cache)
        finished_state = self._get_new_finished_state(state, topk_seq, topk_log_prob,
                                                      finish_flag)

        new_state = {StateKey.CURRENT_IND: state[StateKey.CURRENT_IND] + 1}
        new_state.update(active_state)
        new_state.update(finished_state)
        return [new_state]

    def _next_prediction(self, state):
        ind = state.get(StateKey.CURRENT_IND)
        active_seq = state.get(StateKey.ACTIVE_SEQ)
        active_ll = state.get(StateKey.ACTIVE_LL)
        active_cache = state.get(StateKey.ACTIVE_CACHE)

        if self.padding:
            flat_id = tf.reshape(tf.slice(active_seq, [0, 0, ind], [self.batch_size, self.beam_size, 1]),
                                 [self.batch_size * self.beam_size, -1])
        else:
            flat_id = _flatten_beam_dim(active_seq)

        flat_cache = tf.nest.map_structure(_flatten_beam_dim, active_cache)
        flat_ll, flat_cache = self.f_callable(flat_id, ind, flat_cache)

        new_ll = _unflatten_beam_dim(flat_ll, self.batch_size, self.beam_size)
        new_cache = tf.nest.map_structure(
            lambda t: _unflatten_beam_dim(t, self.batch_size, self.beam_size),
            flat_cache)

        candidate_log_prob = _ll(new_ll)

        log_prob = candidate_log_prob + tf.expand_dims(active_ll, axis=2)

        flat_log_prob = tf.reshape(log_prob,
                                   [-1, self.beam_size * self.vocab_size])

        topk_log_prob, topk_ind = tf.nn.top_k(flat_log_prob, k=self.beam_size * 2)

        topk_beam_ind = topk_ind // self.vocab_size

        topk_seq, new_cache = _gather_beams(
            [active_seq, new_cache], topk_beam_ind, self.batch_size,
            self.beam_size * 2)

        topk_id = topk_ind % self.vocab_size

        if self.padding:
            topk_seq = tf.transpose(topk_seq, perm=[2, 0, 1])
            topk_seq = tf.tensor_scatter_nd_update(topk_seq, [[ind + 1]],
                                                   tf.expand_dims(topk_id, axis=0))
            topk_seq = tf.transpose(topk_seq, perm=[1, 2, 0])
        else:
            topk_seq = tf.concat([topk_seq, tf.expand_dims(topk_id, axis=2)], axis=2)
        return topk_seq, topk_log_prob, topk_id, new_cache

    def _get_new_alive_state(self, new_seq, new_log_probs, new_finished_flags,
                             new_cache):
        # To prevent finished sequences from being considered, set log probs to -inf
        new_log_probs += tf.cast(new_finished_flags, self.dtype) * -inf(self.dtype)

        top_alive_seq, top_alive_log_probs, top_alive_cache = _gather_topk_beams(
            [new_seq, new_log_probs, new_cache], new_log_probs, self.batch_size,
            self.beam_size)

        return {
            StateKey.ACTIVE_SEQ: top_alive_seq,
            StateKey.ACTIVE_LL: top_alive_log_probs,
            StateKey.ACTIVE_CACHE: top_alive_cache}

    def _get_new_finished_state(self, state, new_seq, new_log_probs,
                                new_finished_flags):
        """Combine new and old finished sequences, and gather the top k sequences.
        Args:
          state: A dictionary with the current loop state.
          new_seq: New sequences generated by growing the current alive sequences
            int32 tensor with shape [batch_size, beam_size, i + 1]
          new_log_probs: Log probabilities of new sequences float32 tensor with
            shape [batch_size, beam_size]
          new_finished_flags: A boolean Tensor indicates which sequences are live
            inside the beam.
        Returns:
          Dictionary with finished keys from _StateKeys:
            {Top beam_size finished sequences based on score,
             Scores of finished sequences,
             Finished flags of finished sequences}
        """
        i = state[StateKey.CURRENT_IND]
        finished_seq = state[StateKey.FINISHED_SEQ]
        finished_scores = state[StateKey.FINISHED_LL]
        finished_flags = state[StateKey.FINISHED_FLAG]

        # First append a column of 0-ids to finished_seq to increment the length.
        # New shape of finished_seq: [batch_size, beam_size, i + 1]
        if not self.padding:
            finished_seq = tf.concat([
                finished_seq,
                tf.zeros([self.batch_size, self.beam_size, 1], tf.int32)
            ],
                axis=2)

        # Calculate new seq scores from log probabilities.
        length_norm = _length_normalization(self.normalization, i + 1, dtype=self.dtype)
        new_scores = new_log_probs / length_norm

        # Set the scores of the still-alive seq in new_seq to large negative values.
        new_scores += ((1. - tf.cast(new_finished_flags, self.dtype)) *
                       -inf(self.dtype))

        # Combine sequences, scores, and flags.
        finished_seq = tf.concat([finished_seq, new_seq], axis=1)
        finished_scores = tf.concat([finished_scores, new_scores], axis=1)
        finished_flags = tf.concat([finished_flags, new_finished_flags], axis=1)

        # Return the finished sequences with the best scores.
        top_finished_seq, top_finished_scores, top_finished_flags = (
            _gather_topk_beams([finished_seq, finished_scores, finished_flags],
                               finished_scores, self.batch_size, self.beam_size))

        return {
            StateKey.FINISHED_SEQ: top_finished_seq,
            StateKey.FINISHED_LL: top_finished_scores,
            StateKey.FINISHED_FLAG: top_finished_flags
        }


def _shape_list(tensor):
    shape = tensor.get_shape().as_list()

    # Ensure that the shape values are not None
    dynamic_shape = tf.shape(tensor)
    for i in range(len(shape)):
        if shape[i] is None:
            shape[i] = dynamic_shape[i]
    return shape


def _flatten_beam_dim(tensor):
    """
    [N B *] -> [NB *]
    :param tensor:
    :return:
    """
    shape = _shape_list(tensor)
    shape[0] *= shape[1]
    shape.pop(1)  # Remove beam dim
    return tf.reshape(tensor, shape)


def _unflatten_beam_dim(tensor, batch_size, beam_size):
    shape = _shape_list(tensor)
    new_shape = [batch_size, beam_size] + shape[1:]
    return tf.reshape(tensor, new_shape)


def inf(dtype):
    if dtype == "float32" or dtype == "bfloat16":
        return 1e7
    elif dtype == "float16":

        return np.finfo(np.float16).max  # pylint: disable=no-member
    else:
        raise AssertionError('Invalid dtype: %s' % dtype)


def _expand_to_beam_size(tensor, beam_size):
    tensor = tf.expand_dims(tensor, axis=1)
    tile_dims = [1] * tensor.shape.ndims
    tile_dims[1] = beam_size

    return tf.tile(tensor, tile_dims)


def _gather_beams(nested, beam_indices, batch_size, new_beam_size):
    # Computes the i'th coodinate that contains the batch index for gather_nd.
    # Batch pos is a tensor like [[0,0,0,0,],[1,1,1,1],..].
    batch_pos = tf.range(batch_size * new_beam_size) // new_beam_size
    batch_pos = tf.reshape(batch_pos, [batch_size, new_beam_size])

    # Create coordinates to be passed to tf.gather_nd. Stacking creates a tensor
    # with shape [batch_size, beam_size, 2], where the last dimension contains
    # the (i, j) gathering coordinates.
    coordinates = tf.stack([batch_pos, beam_indices], axis=2)

    return tf.nest.map_structure(
        lambda state: tf.gather_nd(state, coordinates), nested)


def _gather_topk_beams(nested, score_or_log_prob, batch_size, beam_size):
    """Gather top beams from nested structure."""
    _, topk_indexes = tf.nn.top_k(score_or_log_prob, k=beam_size)
    return _gather_beams(nested, topk_indexes, batch_size, beam_size)


def _ll(logits):
    return logits - tf.reduce_logsumexp(logits, axis=2, keepdims=True)


def _length_normalization(alpha, length, dtype=tf.float32):
    return tf.pow(((5. + tf.cast(length, dtype)) / 6.), alpha)
