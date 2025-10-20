import torch
import torch.nn as nn

class UpgradedCVAE(nn.Module):
    def __init__(self, target_dim, cont_cond_dim, latent_dim,
                 cat_dims, embedding_dims):
        super(UpgradedCVAE, self).__init__()

        self.target_dim = target_dim
        self.latent_dim = latent_dim

        # 1. 条件特征嵌入层
        # cat_dims example: {'Hour': 24, 'DayOfWeek': 7, 'Month': 12}
        # embedding_dims example: {'Hour': 8, 'DayOfWeek': 4, 'Month': 4}
        self.hour_embed = nn.Embedding(cat_dims['Hour'], embedding_dims['Hour'])
        self.day_embed = nn.Embedding(cat_dims['DayOfWeek'], embedding_dims['DayOfWeek'])
        self.month_embed = nn.Embedding(cat_dims['Month'], embedding_dims['Month'])

        total_embedding_dim = embedding_dims['Hour'] + embedding_dims['DayOfWeek'] + embedding_dims['Month']
        conditional_dim = cont_cond_dim + total_embedding_dim

        # 2. 编码器 (Encoder)
        encoder_input_dim = target_dim + conditional_dim
        self.encoder_net = nn.Sequential(
            nn.Linear(encoder_input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_log_var = nn.Linear(128, latent_dim)

        # 3. 解码器 (Decoder) - 多任务结构
        decoder_input_dim = latent_dim + conditional_dim
        self.decoder_shared = nn.Sequential(
            nn.Linear(decoder_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU()
        )
        # 为每个Zone创建独立的输出头
        self.decoder_zone1 = nn.Linear(256, 1)
        self.decoder_zone2 = nn.Linear(256, 1)
        self.decoder_zone3 = nn.Linear(256, 1)

    def process_conditions(self, c_cont, c_cat):
        """
        处理连续和类别条件特征，将类别特征嵌入并与连续特征拼接。
        c_cat shape: (batch_size, num_cat_features)
        """
        hour_emb = self.hour_embed(c_cat[:, 0])
        day_emb = self.day_embed(c_cat[:, 1])
        month_emb = self.month_embed(c_cat[:, 2])

        c_emb = torch.cat([hour_emb, day_emb, month_emb], dim=1)
        c_combined = torch.cat([c_cont, c_emb], dim=1)
        return c_combined

    def encode(self, x, c_cont, c_cat):
        """编码器前向传播，计算均值和对数方差。"""
        c = self.process_conditions(c_cont, c_cat)
        inputs = torch.cat([x, c], dim=1)
        h = self.encoder_net(inputs)
        return self.fc_mu(h), self.fc_log_var(h)

    def reparameterize(self, mu, log_var):
        """重参数化技巧，从潜在分布中采样。"""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, c_cont, c_cat):
        """解码器前向传播，从潜在向量和条件信息重建数据。"""
        c = self.process_conditions(c_cont, c_cat)
        inputs = torch.cat([z, c], dim=1)

        h_shared = self.decoder_shared(inputs)

        out1 = self.decoder_zone1(h_shared)
        out2 = self.decoder_zone2(h_shared)
        out3 = self.decoder_zone3(h_shared)

        # 将三个Zone的输出拼接起来，并用Sigmoid确保范围在[0,1]
        return torch.sigmoid(torch.cat([out1, out2, out3], dim=1))

    def forward(self, x, c_cont, c_cat):
        """模型的完整前向传播。"""
        mu, log_var = self.encode(x, c_cont, c_cat)
        z = self.reparameterize(mu, log_var)
        reconstruction = self.decode(z, c_cont, c_cat)
        return reconstruction, mu, log_var

    @torch.no_grad()
    def generate(self, c_cont, c_cat):
        """
        根据给定的条件信息生成新的电力负荷数据。
        直接从标准正态分布中采样潜在向量 z。
        """
        z = torch.randn(c_cont.size(0), self.latent_dim, device=c_cont.device)
        return self.decode(z, c_cont, c_cat)
