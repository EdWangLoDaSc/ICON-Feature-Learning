import model_utils_pytorch as mu_pt
import torch
import torch.nn as nn
from transformers_utils_pytorch import *



bs = 10
model_dim = 256 # input/output dimension of the transformer
sequence_length = 32
n_layers = 2
n_heads = 8
head_dim = 64
widening_factor = 4


transformer_model = SelfAttnTransformer(n_layers = n_layers, 
                            n_heads = n_heads, 
                            head_dim = head_dim, 
                            model_dim = model_dim, 
                            widening_factor = widening_factor)


# 掩码处理
def build_matrices_from_data_shape(data_shape, mode, shot_num_min, return_shape_list = False):
  '''
  data_shape is the shape of data, usually obtained by tree.tree_map(lambda x: x.shape, data)
  '''
  demo_num = data_shape.demo_cond_k[0]
  demo_cond_len = data_shape.demo_cond_k[1]
  demo_qoi_len = data_shape.demo_qoi_k[1]
  quest_cond_len = data_shape.quest_cond_k[1]
  quest_qoi_len = data_shape.quest_qoi_k[1]

  cond_bool_list, qoi_kv_bool_list, qoi_k_bool_list = mu_pt.build_bool_sequence(demo_num, mode, shot_num_min)
  cond_len_list_raw = [demo_cond_len] * demo_num + [quest_cond_len]
  qoi_kv_len_list_raw = [demo_qoi_len] * demo_num + [quest_qoi_len]
  qoi_k_len_list_raw = [demo_qoi_len] * demo_num + [quest_qoi_len]
  cond_len_list = [i * j for i, j in zip(cond_bool_list, cond_len_list_raw)]
  qoi_kv_len_list = [i * j for i, j in zip(qoi_kv_bool_list, qoi_kv_len_list_raw)]
  qoi_k_len_list = [i * j for i, j in zip(qoi_k_bool_list, qoi_k_len_list_raw)]
  
  basic_mask = mu_pt.build_basic_mask(cond_len_list = cond_len_list, 
                                qoi_kv_len_list = qoi_kv_len_list, 
                                qoi_k_len_list = qoi_k_len_list)
  index_pos = mu_pt.build_index_integer(cond_len_list= cond_len_list,
                                      qoi_kv_len_list = qoi_kv_len_list,
                                      qoi_k_len_list = qoi_k_len_list)
  out_mask = mu_pt.build_out_mask(cond_len_list = cond_len_list,
                          qoi_kv_len_list= qoi_kv_len_list,
                          qoi_k_len_list = qoi_k_len_list,
                          num_range = (shot_num_min, demo_num + 1))

  
  if return_shape_list:
    return basic_mask, index_pos, out_mask, cond_len_list, qoi_kv_len_list, qoi_k_len_list
  else:
    return basic_mask, index_pos, out_mask


class ICON_LM(nn.Module):
    def __init__(self, config, basic_mask, index_pos, out_mask, input_dim):
        super(ICON_LM, self).__init__()
        self.config = config
        self.basic_mask = basic_mask  # torch.Tensor of shape [sequence_length, sequence_length]
        self.index_pos = index_pos  # torch.Tensor of indices
        self.out_mask = out_mask  # torch.Tensor of indices or boolean mask

        model_dim = config['transformer']['model_dim']
        out_dim = config['out_dim']
        num_embeddings = config['demo_max_num'] * 3

        self.pre_projection = nn.Linear(input_dim, model_dim)
        self.func_pos_embedding = nn.Embedding(num_embeddings, model_dim)
        self.transformer = SelfAttnTransformer(**config['transformer'])
        self.post_projection = nn.Linear(model_dim, out_dim)

    def basic_forward(self, data, mode, index_pos, basic_mask):
        demo_num = len(data.demo_cond_k)
        cond_bool_list, qoi_kv_bool_list, qoi_k_bool_list = mu_pt.build_bool_sequence(
            demo_num, mode=mode, shot_num_min=0
        )
        sequence = mu_pt.build_data_sequence(
            data, cond_bool_list, qoi_kv_bool_list, qoi_k_bool_list
        )
        mask = mu_pt.build_data_mask(data, cond_bool_list, qoi_kv_bool_list, qoi_k_bool_list)

        sequence = self.pre_projection(sequence)
        sequence = sequence + self.func_pos_embedding(index_pos)

        # Reshape mask from [i] to [sequence_length, i]
        mask = mask.unsqueeze(0).repeat(sequence.size(0), 1)
        mask = mask * basic_mask  # Assuming basic_mask is [sequence_length, mask_length]

        # Expand mask to include the number of heads
        mask = mask.unsqueeze(0).repeat(self.config['transformer']['n_heads'], 1, 1)

        sequence = self.transformer(sequence, mask=mask)
        sequence = self.post_projection(sequence)
        return sequence

    def forward(self, data):
        """
        Standard forward pass using pre-computed matrices.
        Used for training to predict all QoIs based on previous examples and the current condition.
        """
        sequence = self.basic_forward(data, 'train', self.index_pos, self.basic_mask)
        sequence = sequence[self.out_mask]
        return sequence

    def predict(self, data):
        """
        For flexible data shapes, builds basic mask, index, and out mask on the fly.
        Used for testing to predict the last QoI, i.e., the question QoI.
        """
        data_shape = {key: value.shape for key, value in data.items()}
        basic_mask, index_pos, out_mask = build_matrices_from_data_shape(
            data_shape, mode='test', shot_num_min=0
        )
        sequence = self.basic_forward(data, 'test', index_pos, basic_mask)
        sequence = sequence[-data.quest_qoi_mask.shape[-1]:, :]
        return sequence




# hyperpara,eg:
model_config = {"demo_max_num": 6,
                "index_mode": "learn",
                "transformer": {"n_layers":6,
                                "n_heads":8, 
                                "head_dim":256, 
                                "model_dim":256, 
                                "widening_factor": 4, 
                              },
                "out_dim": 1
              }


basic_mask, index_pos, out_mask = build_matrices_from_data_shape(data_shape,  mode = 'train', shot_num_min = 0)
icon_lm_model = ICON_LM(model_config, basic_mask = basic_mask, index_pos = index_pos, out_mask = out_mask)



data_shape = {key: value.shape for key, value in test_data.items()}
basic_mask, index_pos, out_mask = build_matrices_from_data_shape(
    data_shape, mode='train', shot_num_min=0
)

torch.manual_seed(0)

input_dim = test_data.demo_cond_k.shape[-1]

# 初始化模型
icon_lm_model = ICON_LM(
    config=model_config,
    basic_mask=basic_mask,
    index_pos=index_pos,
    out_mask=out_mask,
    input_dim=input_dim
)


def train(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for data in dataloader:
        for key in data:
            data[key] = data[key].to(device)

        # 前向传播
        optimizer.zero_grad()
        outputs = model(data)
        targets = data['targets'] 
        loss = criterion(outputs, targets)

        # 反向传播和优化
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(dataloader)
    return average_loss

def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for data in dataloader:
            for key in data:
                data[key] = data[key].to(device)

            outputs = model(data)
            targets = data['targets']  
            loss = criterion(outputs, targets)

            total_loss += loss.item()

    average_loss = total_loss / len(dataloader)
    return average_loss

def evaluate(model, dataloader, device):
    model.eval()
    predictions = []
    targets_list = []
    with torch.no_grad():
        for data in dataloader:
            for key in data:
                data[key] = data[key].to(device)

            # 前向传播
            outputs = model.predict(data)
            targets = data['targets']  # 假设您的数据中有 'targets' 字段

            predictions.append(outputs.cpu())
            targets_list.append(targets.cpu())


    predictions = torch.cat(predictions, dim=0)
    targets = torch.cat(targets_list, dim=0)


    mse_loss = nn.MSELoss()(predictions, targets)
    return mse_loss.item()

def main():
    # 配置参数
    num_epochs = 10
    batch_size = 32
    learning_rate = 1e-4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 准备数据（假设的哈）
    train_data_list, val_data_list, test_data_list = load_data()

    # 创建数据集和数据加载器
    train_dataset = ICONLMDataset(train_data_list)
    val_dataset = ICONLMDataset(val_data_list)
    test_dataset = ICONLMDataset(test_data_list)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)

    sample_data = train_data_list[0]
    data_shape = {key: value.shape for key, value in sample_data.items()}
    basic_mask, index_pos, out_mask = build_matrices_from_data_shape(
        data_shape, mode='train', shot_num_min=0
    )

    input_dim = sample_data['demo_cond_k'].shape[-1]

    model_config = {
        "demo_max_num": 6,
        "index_mode": "learn",
        "transformer": {
            "n_layers": 6,
            "n_heads": 8,
            "head_dim": 256,
            "model_dim": 256,
            "widening_factor": 4,
        },
        "out_dim": 1
    }

    model = ICON_LM(
        config=model_config,
        basic_mask=basic_mask.to(device),
        index_pos=index_pos.to(device),
        out_mask=out_mask,
        input_dim=input_dim
    ).to(device)

    criterion = nn.MSELoss()  # 或者根据您的任务选择合适的损失函数
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    test_mse = evaluate(model, test_loader, device)
    print(f"Test MSE Loss: {test_mse:.4f}")

    torch.save(model.state_dict(), 'icon_lm_model.pth')



