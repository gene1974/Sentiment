import torch
import torch.nn as nn

def logsumexp(tensor, dim=-1, keepdim=False):
    max_score, _ = tensor.max(dim, keepdim=keepdim)
    if keepdim:
        stable_vec = tensor - max_score
    else:
        stable_vec = tensor - max_score.unsqueeze(dim)
    return max_score + (stable_vec.exp().sum(dim, keepdim=keepdim)).log()

# come from allennlp
def viterbi_decode(tag_sequence, transition_matrix, tag_observations=None):
    sequence_length, num_tags = list(tag_sequence.size())
    if tag_observations:
        if len(tag_observations) != sequence_length:
            raise ValueError("Observations were provided, but they were not the same length "
                             "as the sequence. Found sequence of length: {} and evidence: {}"
                             .format(sequence_length, tag_observations))
    else:
        tag_observations = [-1 for _ in range(sequence_length)]
    path_scores = []
    path_indices = []
    if tag_observations[0] != -1:
        one_hot = torch.zeros(num_tags)
        one_hot[tag_observations[0]] = 100000.
        path_scores.append(one_hot)
    else:
        path_scores.append(tag_sequence[0, :])

    # Evaluate the scores for all possible paths.
    for timestep in range(1, sequence_length):
        # Add pairwise potentials to current scores.
        summed_potentials = path_scores[timestep - 1].unsqueeze(-1) + transition_matrix
        scores, paths = torch.max(summed_potentials, 0)
        # If we have an observation for this timestep, use it
        # instead of the distribution over tags.
        observation = tag_observations[timestep]
        if observation != -1:
            one_hot = torch.zeros(num_tags)
            one_hot[observation] = 100000.
            path_scores.append(one_hot)
        else:
            path_scores.append(tag_sequence[timestep, :] + scores.squeeze())
        path_indices.append(paths.squeeze())

    # Construct the most likely sequence backwards.
    viterbi_score, best_path = torch.max(path_scores[-1], 0)
    viterbi_path = [int(best_path.numpy())]
    for backward_timestep in reversed(path_indices):
        viterbi_path.append(int(backward_timestep[viterbi_path[-1]]))
    # Reverse the backward path.
    viterbi_path.reverse()
    return viterbi_path, viterbi_score

def is_transition_allowed(from_tag, from_entity, to_tag, to_entity):
    '''
    transition rules of BIOES tagging scheme
    from_tag & to_tag: string, ['B', 'I', 'O', 'E', 'S']
    from_entity & to_entity: string, ['PER', 'LOC', 'ORG', 'REL']s
    '''
    if from_tag == "START":
        return to_tag in ['O', 'B', 'S']
    if to_tag == "END":
        return from_tag in ['O', 'E', 'S']
    if from_tag in ['O', 'E', 'S'] and to_tag in ["O", "B", "S"]:
        return True
    elif from_tag in ['B', 'I'] and to_tag in ['I', 'E'] and from_entity == to_entity:
        return True
    else:
        return False

def allowed_transitions(tag_dict):
    '''
    allowed transition index pairs
    tag_dict: index to label dict
    '''
    allowed = []
    labels = list(tag_dict.items())
    for from_label_index, from_label in labels:
        if from_label in ["START", "END"]:
            from_tag = from_label
            from_entity = "" 
        else:
            from_tag = from_label[0]
            from_entity = from_label[2:]
        for to_label_index, to_label in labels:
            if to_label in ["START", "END"]:
                to_tag = to_label
                to_entity = ""
            else:
                to_tag = to_label[0]
                to_entity = to_label[2:]
            if is_transition_allowed(from_tag, from_entity, to_tag, to_entity):
                allowed.append((from_label_index, to_label_index))
    return allowed

# mostly come from allennlp ConditionalRandomField
class CRF(nn.Module):
    def __init__(self, tag_dict):
        super(CRF, self).__init__()
        self.tag_dict = tag_dict
        self.num_tags = len(tag_dict)
        if "START" not in self.tag_dict:
            self.tag_dict.update({self.num_tags: "START", self.num_tags + 1: "END"})
        self.transitions = nn.Parameter(torch.Tensor(self.num_tags, self.num_tags))
        constraint = allowed_transitions(self.tag_dict)
        constraint_mask = torch.Tensor(self.num_tags + 2, self.num_tags + 2).fill_(0.)
        for i, j in constraint:
            constraint_mask[i, j] = 1.
        self._constraint_mask = torch.nn.Parameter(constraint_mask, requires_grad=False)
        self.start_transitions = nn.Parameter(torch.Tensor(self.num_tags))
        self.end_transitions = nn.Parameter(torch.Tensor(self.num_tags))
        # init params
        nn.init.xavier_normal_(self.transitions)
        nn.init.normal_(self.start_transitions)
        nn.init.normal_(self.end_transitions)

    def _input_likelihood(self, logits, mask):
        '''
        logits: (batch_size, sen_len, num_tags)
        mask: (batch_size, sen_len), means whether the token is padded
        '''
        batch_size, sequence_length, num_tags = logits.size()
        # mask: (sen_len, batch_size)
        mask = mask.float().transpose(0, 1).contiguous()
        # logits: (sen_len, batch_size, num_tags)
        logits = logits.transpose(0, 1).contiguous()
        # alpha: (batch_size, num_tags)
        alpha = logits[0] + self.start_transitions.view(1, num_tags)
        for i in range(1, sequence_length):
            emit_scores = logits[i].view(batch_size, 1, num_tags)
            transition_scores = self.transitions.view(1, num_tags, num_tags)
            broadcast_alpha = alpha.view(batch_size, num_tags, 1)
            # inner: (batch_size, num_tags, num_tags)
            inner = broadcast_alpha + emit_scores + transition_scores
            # if mask == 0, then keep alpha unchanged
            alpha = (logsumexp(inner, 1) * mask[i].view(batch_size, 1) + \
                     alpha * (1 - mask[i]).view(batch_size, 1))
        return logsumexp(alpha + self.end_transitions.view(1, num_tags))

    def _joint_likelihood(self, logits, tags, mask):
        '''
        logits: (batch_size, sen_len, num_tags)
        tags: (batch_size, sen_len)
        mask: (batch_size, sen_len), means whether the token is padded
        '''
        batch_size, sequence_length, _ = logits.data.shape
        # logits: (sen_len, batch_size, num_tags)
        logits = logits.transpose(0, 1).contiguous()
        # mask: (sen_len, batch_size)
        mask = mask.float().transpose(0, 1).contiguous()
        # tags: (sen_len, batch_size)
        tags = tags.transpose(0, 1).contiguous()
        score = self.start_transitions.index_select(0, tags[0])
        for i in range(sequence_length-1):
            current_tag, next_tag = tags[i], tags[i + 1]
            transition_score = self.transitions[current_tag.view(-1), next_tag.view(-1)]
            emit_score = logits[i].gather(1, current_tag.view(batch_size, 1)).squeeze(1)
            score = score + transition_score * mask[i + 1] + emit_score * mask[i]
        last_tag_index = mask.sum(0).long() - 1
        last_tags = tags.gather(0, last_tag_index.view(1, batch_size)).squeeze(0)
        last_transition_score = self.end_transitions.index_select(0, last_tags)
        last_inputs = logits[-1]
        last_input_score = last_inputs.gather(1, last_tags.view(-1, 1))
        last_input_score = last_input_score.squeeze()
        score = score + last_transition_score + last_input_score * mask[-1]
        return score

    def forward(self, inputs, tags, mask):
        '''
        inputs: (batch_size, sen_len, num_tags)
        tags: (batch_size, sen_len)
        mask: (batch_size, sen_len)
        '''
        if mask is None:
            mask = torch.ones(*tags.size(), dtype=torch.long)
        log_denominator = self._input_likelihood(inputs, mask)
        log_numerator = self._joint_likelihood(inputs, tags, mask)
        return torch.sum(log_numerator - log_denominator)

    def viterbi_tags(self, logits, mask):
        _, max_seq_length, num_tags = logits.size()
        # Get the tensors out of the variables
        logits, mask = logits.data, mask.data
        start_tag = num_tags
        end_tag = num_tags + 1
        transitions = torch.Tensor(num_tags + 2, num_tags + 2).fill_(-10000.)
        # Apply transition constraints
        constrained_transitions = (
                self.transitions * self._constraint_mask[:num_tags, :num_tags] +
                -10000.0 * (1 - self._constraint_mask[:num_tags, :num_tags])
        )
        transitions[:num_tags, :num_tags] = constrained_transitions.data
        transitions[start_tag, :num_tags] = (self.start_transitions.detach() * self._constraint_mask[start_tag, :num_tags].data \
            + -10000. * (1 - self._constraint_mask[start_tag, :num_tags].detach()))
        transitions[:num_tags, end_tag] = (self.end_transitions.detach() * self._constraint_mask[:num_tags, end_tag].data \
            + -10000. * (1 - self._constraint_mask[:num_tags, end_tag].detach()))
        best_paths = []
        # Pad the max sequence length by 2 to account for start_tag + end_tag.
        tag_sequence = torch.Tensor(max_seq_length + 2, num_tags + 2)

        for prediction, prediction_mask in zip(logits, mask):
            sequence_length = torch.sum(prediction_mask)
            # Start with everything totally unlikely
            tag_sequence.fill_(-10000.)
            # At timestep 0 we must have the START_TAG
            tag_sequence[0, start_tag] = 0.
            # At steps 1, ..., sequence_length we just use the incoming prediction
            tag_sequence[1:(sequence_length + 1), :num_tags] = prediction[:sequence_length]
            # And at the last timestep we must have the END_TAG
            tag_sequence[sequence_length + 1, end_tag] = 0.
            # We pass the tags and the transitions to ``viterbi_decode``.
            viterbi_path, viterbi_score = viterbi_decode(tag_sequence[:(sequence_length + 2)], transitions)
            # Get rid of START and END sentinels and append.
            viterbi_path = viterbi_path[1:-1]
            best_paths.append((viterbi_path, viterbi_score.item()))
        
        predicted_tags = [x for x, y in best_paths]
        predicted_tensor = mask * 0.
        for i, instance_tags in enumerate(predicted_tags):
            for j, tag_id in enumerate(instance_tags):
                predicted_tensor[i, j] = tag_id
        return predicted_tensor.long()
