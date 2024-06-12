import torch
torch.manual_seed(20230226)  # 固定随机种子
torch.manual_seed(20230226)  # 设置CPU的随机种子
torch.cuda.manual_seed(20230226)  # 设置GPU的随机种子
def train(model, optimizer, train_loader, valid_loader, prototype_dict, args):
    num_stop_dropping = 0
    best_valid_loss = float('inf')

    for epoch in range(1, args.vae_epochs + 1): #100
        train_loss = train_one_epoch(model, train_loader, optimizer, args)
        valid_loss = valid_one_epoch(model, valid_loader, args)

        if valid_loss < best_valid_loss:
            num_stop_dropping = 0
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), args.check_pt_vae_model_path)
        else:
            num_stop_dropping += 1
        if num_stop_dropping >= args.vae_early_stop_tolerance:
            print('Have not increased for %d check points, early stop training' % num_stop_dropping)
            break
        print(f'VAE | Epochs: {epoch} | Train Loss: {train_loss: .4f} | Valid Loss: {valid_loss: .4f} | Early Stop: {num_stop_dropping} ')


def train_one_epoch(model, train_loader, optimizer, args):
    model.train()
    train_loss = 0
    device='cuda'
    for batch_idx, batch in enumerate(train_loader):
        x, x_pro = batch #[256, 5120],[256, 5120]
        x = x.to(device)
        x_pro = x_pro.to(device)
        optimizer.zero_grad()
        mu, logvar, recon_feats = model(x, x_pro)
        loss = VAE_loss(recon_feats, x, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    return train_loss

def valid_one_epoch(model, valid_loader, args):
    model.eval()
    valid_loss = 0
    device = 'cuda'
    # print('111')
    with torch.no_grad():
        for batch_idx, batch in enumerate(valid_loader):
            x, x_pro = batch
            x = x.to(device)
            x_pro = x_pro.to(device)
            mu, logvar, recon_feats = model(x, x_pro)
            loss = VAE_loss(recon_feats, x, mu, logvar)
            valid_loss += loss.item()
    return valid_loss

def VAE_loss(recon_feats, x, mu, logvar):
    recon_loss = torch.abs(recon_feats - x).mean(1)  # 计算每个样本的L1损失
    recon_loss = torch.mean(recon_loss)
    kl_loss = (1 + logvar - logvar.exp() - mu.pow(2)).sum(1)
    kl_loss = -0.5 * torch.mean(kl_loss)
    L_vae =0.2* recon_loss +0.8* kl_loss #
    return L_vae


