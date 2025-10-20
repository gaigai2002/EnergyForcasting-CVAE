import torch

def mmd_loss(p, q, kernel='rbf', sigma=1.0):
    """
    计算p和q两个batch之间的MMD损失 (Maximum Mean Discrepancy)。
    p: 真实样本 (batch_size, dim)
    q: 生成样本 (batch_size, dim)
    """
    xx, yy, zz = torch.mm(p, p.t()), torch.mm(q, q.t()), torch.mm(p, q.t())

    # RBF 核 (高斯核)
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    Kxx = torch.exp(-0.5 * (rx.t() + rx - 2 * xx) / sigma ** 2)
    Kyy = torch.exp(-0.5 * (ry.t() + ry - 2 * yy) / sigma ** 2)
    Kxy = torch.exp(-0.5 * (rx.t() + ry - 2 * zz) / sigma ** 2)

    return Kxx.mean() + Kyy.mean() - 2 * Kxy.mean()


def combined_loss_function(recon_x, x, mu, log_var, generated_x,
                           zone_weights, beta=1.0, alpha=1.0):
    """
    计算总损失 = 加权重建损失 + beta * KLD + alpha * MMD。
    """
    # 1. 加权重建损失 (Weighted MSE)
    # recon_x, x 均为 (batch_size, target_dim)
    # zone_weights 为 (target_dim,)
    recon_loss_per_zone = torch.mean((recon_x - x) ** 2, dim=0)  # 按Zone计算MSE
    weighted_recon_loss = torch.sum(recon_loss_per_zone * zone_weights)

    # 2. KL 散度损失
    kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    kld_loss /= x.size(0)  # 平均到每个样本

    # 3. MMD 损失
    mmd = mmd_loss(x, generated_x)

    # 4. 组合损失
    total_loss = weighted_recon_loss + beta * kld_loss + alpha * mmd

    return total_loss, weighted_recon_loss, kld_loss, mmd
