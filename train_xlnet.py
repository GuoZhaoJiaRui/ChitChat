train_samples = []
train_labels = []

with open('./train.txt', 'r') as f:
    while True:
        s1 = f.readline()
        if not s1:
            # 如果读取到内容为空，则读取结束
            break
        s2 = f.readline()
        _ = f.readline()
        train_samples.append(s1.replace('\n', ''))
        train_labels.append(s2.replace('\n', ''))

len(train_samples), len(train_labels)



from transformers import XLNetTokenizer

# 使用 XLNet 的分词器
tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')

input_ids = []
input_labels = []
for text, ori_labels in zip(train_samples, train_labels):
    l = text.split(' ')
    labels = []
    text_ids = []
    for word, label in zip(l, ori_labels):
        if word == '':
            continue
        tokens = tokenizer.tokenize(word)
        for i, j in enumerate(tokens):
            if i == 0:
                labels.append(int(label))
                text_ids.append(tokenizer.convert_tokens_to_ids(j))
            else:
                labels.append(9)
                text_ids.append(tokenizer.convert_tokens_to_ids(j))
    input_ids.append(text_ids)
    input_labels.append(labels)

len(input_ids), len(input_labels)












import torch
from torch.utils.data import DataLoader, TensorDataset

del train_samples
del train_labels

for j in range(len(input_ids)):
    # 将样本数据填充至长度为 128
    i = input_ids[j]
    if len(i) != 128:
        input_ids[j].extend([0]*(128 - len(i)))

for j in range(len(input_labels)):
    # 将样本数据填充至长度为 128
    i = input_labels[j]
    if len(i) != 128:
        input_labels[j].extend([10]*(128 - len(i)))

# 构建数据集和数据迭代器，设定 batch_size 大小为 8
train_set = TensorDataset(torch.LongTensor(input_ids),
                          torch.LongTensor(input_labels))
train_loader = DataLoader(dataset=train_set,
                          batch_size=8,
                          shuffle=True)
# train_loader



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device




import torch.nn as nn
from transformers import XLNetModel

class NERModel(nn.Module):
    def __init__(self, num_class=11):
        super(NERModel, self).__init__()
        # 读取 XLNet 预训练模型
        self.model = XLNetModel.from_pretrained("./")
        self.dropout = nn.Dropout(0.1)
        self.l1 = nn.Linear(768, num_class)

    def forward(self, x, attention_mask=None):
        outputs = self.model(x, attention_mask=attention_mask)
        x = outputs[0]  # 形状为 batch * seqlen * 768
        x = self.dropout(x)
        x = self.l1(x)
        return x
def loss_function(logits, target, masks, num_class=11):
    criterion = nn.CrossEntropyLoss(reduction='none')
    logits = logits.view(-1, num_class)
    target = target.view(-1)
    masks = masks.view(-1)
    cross_entropy = criterion(logits, target)
    loss = cross_entropy * masks
    loss = loss.sum() / (masks.sum() + 1e-12)  # 加上 1e-12 防止被除数为 0
    loss = loss.to(device)
    return loss


from torch.optim import Adam

model = NERModel()
model.to(device)
model.train()

optimizer = Adam(model.parameters(), lr=1e-5)






from torch.autograd import Variable
import time

pre = time.time()

epoch = 3

for i in range(epoch):
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data).to(device), Variable(target).to(device)

        optimizer.zero_grad()

        # 生成掩膜
        mask = []
        for sample in data:
            mask.append([1 if i != 0 else 0 for i in sample])
        mask = torch.FloatTensor(mask).to(device)

        output = model(data, attention_mask=mask)

        # 得到模型预测结果
        pred = torch.argmax(output, dim=2)

        loss = loss_function(output, target, mask)
        loss.backward()

        optimizer.step()

        if ((batch_idx+1) % 10) == 1:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss:{:.6f}'.format(
                i+1, batch_idx, len(train_loader), 100. *
                batch_idx/len(train_loader), loss.item()
            ))

        if batch_idx == len(train_loader)-1:
            # 在每个 Epoch 的最后输出一下结果
            print('labels:', target)
            print('pred:', pred)

print('训练时间：', time.time()-pre)

# 训练结束后，可以使用验证集观察模型的训练效果。

# 读取验证集数据和构建数据迭代器的方式与训练集相同。

eval_samples = []
eval_labels = []

with open('./dev.txt', 'r') as f:
    while True:
        s1 = f.readline()
        if not s1:
            break
        s2 = f.readline()
        _ = f.readline()
        eval_samples.append(s1.replace('\n', ''))
        eval_labels.append(s2.replace('\n', ''))

len(eval_samples)

# 这里使用和训练集同样的方式修改标签，不再赘述
input_ids = []
input_labels = []
for text, ori_labels in zip(eval_samples, eval_labels):
    l = text.split(' ')
    labels = []
    text_ids = []
    for word, label in zip(l, ori_labels):
        if word == '':
            continue
        tokens = tokenizer.tokenize(word)
        for i, j in enumerate(tokens):
            if i == 0:
                labels.append(int(label))
                text_ids.append(tokenizer.convert_tokens_to_ids(j))
            else:
                labels.append(9)
                text_ids.append(tokenizer.convert_tokens_to_ids(j))
    input_ids.append(text_ids)
    input_labels.append(labels)

del eval_samples
del eval_labels

for j in range(len(input_ids)):
    # 将样本数据填充至长度为 128
    i = input_ids[j]
    if len(i) != 128:
        input_ids[j].extend([0]*(128 - len(i)))

for j in range(len(input_labels)):
    # 将样本数据填充至长度为 128
    i = input_labels[j]
    if len(i) != 128:
        input_labels[j].extend([10]*(128 - len(i)))

# 构建数据集和数据迭代器，设定 batch_size 大小为 1
eval_set = TensorDataset(torch.LongTensor(input_ids),
                         torch.LongTensor(input_labels))
eval_loader = DataLoader(dataset=eval_set,
                         batch_size=1,
                         shuffle=False)
# eval_loader

# 将模型设置为验证模式，输入验证集数据。

from tqdm import tqdm_notebook as tqdm

model.eval()

correct = 0
total = 0

for batch_idx, (data, target) in enumerate(tqdm(eval_loader)):
    data = data.to(device)
    target = target.float().to(device)

    # 生成掩膜
    mask = []
    for sample in data:
        mask.append([1 if i != 0 else 0 for i in sample])
    mask = torch.Tensor(mask).to(device)

    output = model(data, attention_mask=mask)

    # 得到模型预测结果
    pred = torch.argmax(output, dim=2)

    # 将掩膜添加到预测结果上，便于计算准确率
    pred = pred.float()
    pred = pred * mask
    target = target * mask

    pred = pred[:, 0:mask.sum().int().item()]
    target = target[:, 0:mask.sum().int().item()]

    correct += (pred == target).sum().item()
    total += mask.sum().item()

print('正确分类的标签数：{}，标签总数：{}，准确率：{:.2f}%'.format(
    correct, total, 100.*correct/total))

if __name__ == '__main__':
    pass