import random
class Sampler:
    def __init__(self, num_contexts):
        self.num_contexts = num_contexts

    def sample(self, contexts):
        return []
    
class RandomSampler(Sampler):
    def sample(self, contexts):
        return random.sample(contexts, self.num_contexts)

class SequentialSampler(Sampler):
    def sample(self, contexts):
        return contexts[:self.num_contexts]