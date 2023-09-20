
from mindspore import context
# 设置  mindspore下的 GPU环境
context.set_context(device_id=1, device_target="GPU")

import mindspore as ms
import numpy as np
import mindspore.dataset as ds
from mindspore import nn, ops
import os
import nibabel as nib



dataset_dir= "/media/sdd/gaoxingyu/Project2/Dataset" # 数据集根目录
train_batch_size = 4 # 批量大小
test_batch_size = 1
#image_size = [76, 94, 76] # 训练图像空间大小
workers = 4 # 并行线程个数
num_classes = 2 # 分类数量



# 自定义数据集 （训练集和测试集）
class create_dataset_train:
    def __init__(self, dataset_dir):  

        dataset_dir_to = os.path.join(dataset_dir, 'TRAIN')

        self.path = dataset_dir_to
        filename_list = os.listdir(self.path)
        train_data = []

        for image_label in filename_list:
            train_data.append(image_label)
        train_data = np.asarray(train_data)
        self.name = train_data

    def __len__(self):
        return len(self.name)
        

    def __getitem__(self, index):
        file_name = self.name[index]                    
        path = os.path.join(self.path, file_name)
        label = np.array(file_name[0]).astype(np.int32) 

        out = nib.load(path).get_fdata()
        data = np.array(out).astype(np.float32) 

        max_val_ = data.max()
        min_val_ = data.min()
        data = (data - min_val_) / (max_val_ - min_val_)

        data = np.expand_dims(data, axis=0)
        return (data, label)

class create_dataset_test:
    def __init__(self, dataset_dir):  

        dataset_dir_to = os.path.join(dataset_dir, 'ADNI2')

        self.path = dataset_dir_to
        filename_list = os.listdir(self.path)
        train_data = []

        for image_label in filename_list:
            train_data.append(image_label)
        train_data = np.asarray(train_data)
        self.name = train_data

    def __len__(self):
        return len(self.name)
        

    def __getitem__(self, index):
        file_name = self.name[index]                    
        path = os.path.join(self.path, file_name)
        label = np.array(file_name[0]).astype(np.int32) 

        out = nib.load(path).get_fdata()
        data = np.array(out).astype(np.float32) 


        max_val_ = data.max()
        min_val_ = data.min()
        data = (data - min_val_) / (max_val_ - min_val_)

        data = np.expand_dims(data, axis=0)
        return (data, label)
                

# 利用上面写好的那个函数，获取处理后的训练与测试数据集
dataset_generator_train = create_dataset_train(dataset_dir)
dataset_train = ds.GeneratorDataset(dataset_generator_train, num_parallel_workers=workers, column_names=["data", "label"], shuffle=True).batch(train_batch_size)



dataset_generator_test = create_dataset_test(dataset_dir)
dataset_test = ds.GeneratorDataset(dataset_generator_test, num_parallel_workers=workers, column_names=["data", "label"], shuffle=False).batch(test_batch_size)



step_size_train = dataset_train.get_dataset_size()
step_size_val = dataset_train.get_dataset_size()




"""
构建网络
"""
from CNNplusTransformer_model_reg import DenseNet21plusTransformer_reg
from CNNplusTransformer_model import DenseNet21plusTransformer

"""
这个函数主要是用来处理预训练模型的 就是如果有预训练模型参数需要在训练之前输入 就把pretrained设为True 此处由于没有预训练模型提供 因此后面在训练的时候设置的是False

"""
from mindspore import load_checkpoint, load_param_into_net
def _model(pretrained: bool = True):
    # num_classes = 2
    model2 = DenseNet21plusTransformer(2)
    #存储路径
    model_ckpt = "./BestCheckpoint/best.ckpt"

    if pretrained:
        param_dict = load_checkpoint(model_ckpt)
        load_param_into_net(model, param_dict)

    return model2


def _model_reg(pretrained: bool = False):
    # num_classes = 2
    model1 = DenseNet21plusTransformer_reg(2)
    #存储路径
    model_ckpt = "./BestCheckpoint/best.ckpt"

    if pretrained:
        param_dict = load_checkpoint(model_ckpt)
        load_param_into_net(model, param_dict)

    return model1


"""
训练
"""
###############################################################################################################
###############################################################################################################
###############################################################################################################
###############################################################################################################
###############################################################################################################
###############################################################################################################
###############################################################################################################
###############################################################################################################
###############################################################################################################
import mindspore as ms
# ms.set_context(device_target='GPU')
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 定义网络，此处不采用预训练，即将pretrained设置为False
MindSpore_Model1 = _model_reg(pretrained=True)
MindSpore_Model2 = _model(pretrained=True)

#param.requires_grad = True表示所有参数都需要求梯度进行更新。
for param in MindSpore_Model1.get_parameters():
    param.requires_grad = True

for param in MindSpore_Model2.get_parameters():
    param.requires_grad = False

# 设置训练的轮数和学习率 #*****************************************************************************************
num_epochs = 120    
#基于余弦衰减函数计算学习率。学习率最小值为0.0001，最大值为0.0005，具体API见文档https://www.mindspore.cn/docs/zh-CN/master/api_python/nn/mindspore.nn.cosine_decay_lr.html?highlight=cosine_decay_lr
lr = nn.cosine_decay_lr(min_lr=0.0001, max_lr=0.0005, total_step=step_size_train * num_epochs,
                        step_per_epoch=step_size_train, decay_epoch=num_epochs)
# 定义优化器和损失函数
#Adam优化器，具体可参考论文https://arxiv.org/abs/1412.6980
opt = nn.Adam(params=MindSpore_Model1.trainable_params(), learning_rate=lr)

# 交叉熵损失
loss_fn = nn.CrossEntropyLoss()
loss_mse = nn.MSELoss()

#前向传播，计算loss
def forward_fn(inputs, targets):
    logits = MindSpore_Model1(inputs)
    loss = loss_mse(logits, targets)
    return loss

#计算梯度和loss
grad_fn = ops.value_and_grad(forward_fn, None, opt.parameters)

def train_step(inputs, targets):
    targets = MindSpore_Model2(inputs)
    loss, grads = grad_fn(inputs, targets)
    opt(grads)
    return loss

# 实例化模型
model = ms.Model(MindSpore_Model1, loss_fn, opt, metrics={"Accuracy": nn.Accuracy()})


# 创建迭代器
# data_loader_train = dataset_train.create_tuple_iterator(num_epochs=num_epochs)
# data_loader_val = dataset_train.create_tuple_iterator(num_epochs=num_epochs)

# 最佳模型存储路径
best_acc = 0
best_ckpt_dir = "./BestCheckpoint"
best_ckpt_path = "./BestCheckpoint/DenseNet_best.ckpt"


import stat

# 开始循环训练
print("Start Training Loop ...")

for epoch in range(num_epochs):
    losses = []
    MindSpore_Model1.set_train()

    # 为每轮训练读入数据
    for i, data in enumerate(dataset_train):
        images = data[0]
        labels = data[1]


        output = MindSpore_Model1(images)
        targets = MindSpore_Model2(images)
        loss = train_step(images, targets)
        if i%30 == 0 or i == step_size_train -1:
            print('Epoch: [%3d/%3d], Steps: [%3d/%3d], Train Loss: [%5.3f]'%(epoch+1, num_epochs, i+1, step_size_train, loss))
        losses.append(loss)

    # 每个epoch结束后，验证准确率
    acc = model.eval(dataset_test,  dataset_sink_mode=False)['Accuracy']

    

    print("-" * 50)
    print("Epoch: [%3d/%3d], Average Train Loss: [%5.3f], Accuracy: [%5.3f]" % (
        epoch+1, num_epochs, sum(losses)/len(losses), acc
    ))
    print("-" * 50)

    if acc > best_acc:
        best_acc = acc
        if not os.path.exists(best_ckpt_dir):
            os.mkdir(best_ckpt_dir)
        if os.path.exists(best_ckpt_path):
            os.chmod(best_ckpt_path, stat.S_IWRITE)#取消文件的只读属性，不然删不了
            os.remove(best_ckpt_path)
        ms.save_checkpoint(MindSpore_Model1, best_ckpt_path)

print("=" * 80)
print(f"End of validation the best Accuracy is: {best_acc: 5.3f}, "
      f"save the best ckpt file in {best_ckpt_path}", flush=True)
print("=" * 80)

