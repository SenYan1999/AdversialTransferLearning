import torch
from model import DomainClassifier
from model import AdversialAdam
from torch.optim import Adam

model = DomainClassifier('bert-base-chinese', 768, 128, 1, 6, 0.1, 2)
encoder_params = filter(lambda x:id(x) not in list(map(id,model.classifier.parameters())), model.parameters())
adversial_optimizer = AdversialAdam(encoder_params, lr=1e-3)
classifier_optimizer = Adam(model.classifier.parameters(), lr=1e-3)

x = torch.LongTensor([[100, 67, 874, 0, 0, 0], [100, 3425, 8764, 854, 6542, 0]])
y = torch.LongTensor([1, 1])

out = model(x)[0]
loss = torch.nn.functional.nll_loss(out, y)

loss.backward()

adversial_optimizer.step()
classifier_optimizer.step()

print('done')