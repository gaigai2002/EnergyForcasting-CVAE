import torch
import torch.optim as optim
from tqdm.notebook import tqdm

def train_model(model, train_loader, test_loader, optimizer,
                device, config, loss_function):
    """
    训练CVAE模型。
    """
    train_losses, test_mses = [], []

    for epoch in tqdm(range(config.EPOCHS), desc="Total Training Progress"):
        model.train()
        total_loss, total_recon, total_kld, total_mmd = 0, 0, 0, 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.EPOCHS} Train", unit="batch")
        for c_cont, c_cat, x in pbar:
            c_cont, c_cat, x = c_cont.to(device), c_cat.to(device), x.to(device)

            optimizer.zero_grad()

            # 前向传播
            recon_x, mu, log_var = model(x, c_cont, c_cat)

            # 为MMD损失生成样本
            generated_x_for_mmd = model.generate(c_cont, c_cat)

            # 计算损失
            loss, recon, kld, mmd = loss_function(
                recon_x, x, mu, log_var, generated_x_for_mmd, config.ZONE_WEIGHTS, config.BETA, config.ALPHA
            )

            # 反向传播和优化
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_recon += recon.item()
            total_kld += kld.item()
            total_mmd += mmd.item()

            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Recon': f'{recon.item():.4f}',
                'KLD': f'{kld.item():.4f}',
                'MMD': f'{mmd.item():.4f}'
            })

        avg_loss = total_loss / len(train_loader)
        avg_recon = total_recon / len(train_loader)
        avg_kld = total_kld / len(train_loader)
        avg_mmd = total_mmd / len(train_loader)
        train_losses.append(avg_loss)
        print(
            f"Epoch {epoch + 1} Avg Train Loss: {avg_loss:.4f} | Recon: {avg_recon:.4f} | KLD: {avg_kld:.4f} | MMD: {avg_mmd:.4f}")

        # --- 在测试集上评估 ---
        model.eval()
        all_recon_x = []
        all_x = []
        with torch.no_grad():
            for c_cont, c_cat, x in test_loader:
                c_cont, c_cat, x = c_cont.to(device), c_cat.to(device), x.to(device)
                recon_x, _, _ = model(x, c_cont, c_cat)
                all_recon_x.append(recon_x)
                all_x.append(x)

        all_recon_x = torch.cat(all_recon_x, dim=0)
        all_x = torch.cat(all_x, dim=0)

        # 计算每个Zone的MSE
        test_mse = torch.mean((all_recon_x - all_x) ** 2, dim=0)
        test_mses.append(test_mse.cpu().numpy())

        print(f"Epoch {epoch + 1} Test MSE -> "
              f"Zone1: {test_mse[0]:.6f}, "
              f"Zone2: {test_mse[1]:.6f}, "
              f"Zone3: {test_mse[2]:.6f} (Weighted)")
        print("-" * 80)

    print("训练完成!")
    return model, train_losses, test_mses
