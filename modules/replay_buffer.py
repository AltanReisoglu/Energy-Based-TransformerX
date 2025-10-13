import random
import torch

class CausalReplayBuffer: # replay buffer meant to be used by causal models (see indexing)
    def __init__(self, max_size, sample_size):
        self.buffer = []
        self.max_size = max_size
        self.sample_size = int(sample_size)

    def add(self, record):
        self.buffer.append(record)
        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)

    def get_batch(self, all_data):
        fresh_inputs = all_data[:, :-1]
        fresh_gt_outputs = all_data[:, 1:]
        if not self.buffer: # just use original data pass back None
            return fresh_inputs, None, fresh_gt_outputs
        num_samples = min(self.sample_size, len(self.buffer))
        samples = random.sample(self.buffer, num_samples)
        replay_inputs = torch.cat([s["all_data"][:, :-1] for s in samples], dim=0)
        replay_gt_outputs = torch.cat([s["all_data"][:, 1:] for s in samples], dim=0)
        replay_predicted_outputs = torch.cat([s["predicted_outputs"] for s in samples], dim=0)
        combined_inputs = torch.cat([fresh_inputs, replay_inputs], dim=0) #NOTE need to cat fresh inputs first, forward funcs assume it
        combined_gt_outputs = torch.cat([fresh_gt_outputs, replay_gt_outputs], dim=0)
        return combined_inputs, replay_predicted_outputs, combined_gt_outputs

    def update(self, all_data, final_predicted_outputs):
        for i in range(all_data.size(0)):
            record = {
                "all_data": all_data[i].unsqueeze(0),
                "predicted_outputs": final_predicted_outputs[i].detach().unsqueeze(0)
            }
            self.add(record)