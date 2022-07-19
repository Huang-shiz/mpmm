import torch
from torch import nn
from util.mpmm import MPMM
from fastNLP import Tester, Vocabulary, EarlyStopCallback
from fastNLP import Trainer, CrossEntropyLoss, AccuracyMetric
from fastNLP.io import CSVLoader
from fastNLP.embeddings import StaticEmbedding
from fastNLP.core.callback import WarmupCallback, GradientClipCallback
from fastNLP.core.metrics import ClassifyFPreRecMetric, ConfusionMatrixMetric, AccuracyMetric


class my_model(nn.Module):
    def __init__(self, embed):
        super().__init__()
        self.embed = embed
        self.mpmm = MPMM(embed=embed, hidden_size=100, num_classes=4, dropout=0.2)

    def forward(self, words1, words2):
        outputs = self.mpmm(words1, words2)
        return {'pred':outputs} 


print('loading data...')
loader = CSVLoader(headers=('sent1','sent2','label'))
data_bundle = loader.load('./data')
device = '0'
bs = 64

print('preprocessing...')
MAP = {'0':'duplicate', '1':'direct', '2':'indirect', '3':'isolated'}
data_bundle.apply(lambda x: x['sent1'].split(), new_field_name='words1', is_input=True)
data_bundle.apply(lambda x: x['sent2'].split(), new_field_name='words2', is_input=True)
data_bundle.apply(lambda x: MAP[x['label']], new_field_name='target', is_target=True)

train_data = data_bundle.get_dataset('train')
dev_data = data_bundle.get_dataset('dev')
test_data = data_bundle.get_dataset('test')
print(train_data[:10])

print('vocabulary...')
vocab = Vocabulary()
vocab = vocab.load('/home/huang/data/w2v/vocab.txt')
vocab.index_dataset(train_data, dev_data, test_data, field_name=['words1', 'words2'], new_field_name=['words1', 'words2'])

target_vocab = Vocabulary(padding=None, unknown=None)
target_vocab.add_word_lst(['duplicate','direct','indirect','isolated'])
target_vocab.index_dataset(train_data, dev_data, test_data, field_name='target')

data_bundle.set_vocab(field_name='words', vocab=vocab)
data_bundle.set_vocab(field_name='target', vocab=target_vocab)
data_bundle.set_input('words1', 'words2')
data_bundle.set_target('target')
print(train_data[:10])

print('embedding...')
path = '/home/huang/data/w2v/emb300.txt'
embed = StaticEmbedding(data_bundle.get_vocab('words'), model_dir_or_name=path, requires_grad=False, dropout=0.1, word_dropout=0.1)


model = my_model(embed=embed)

optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(num_params)

callbacks = [WarmupCallback(warmup=0.2, schedule='linear'), EarlyStopCallback(5),
            GradientClipCallback(clip_value=5, clip_type='value')]
trainer = Trainer(train_data=train_data, model=model, optimizer=optimizer,
                  loss=CrossEntropyLoss(), device=device, batch_size=bs, dev_data=dev_data,
                  metrics=AccuracyMetric(), n_epochs=30, print_every=1, update_every=1, callbacks=callbacks)

print('training...')
trainer.train()


metric = ClassifyFPreRecMetric(f_type='macro', only_gross=False)
tester = Tester(data=test_data, model=model, device=device, batch_size=bs, metrics=metric)
tester.test()

metric = ConfusionMatrixMetric()
tester = Tester(data=test_data, model=model, device=device, batch_size=bs, metrics=metric)
tester.test()
