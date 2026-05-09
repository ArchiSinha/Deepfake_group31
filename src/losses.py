import torch.nn as nn

bce_loss = nn.BCELoss()
l1_loss  = nn.L1Loss()


def discriminator_loss(real_preds, fake_preds):
    """D wants real→1, fake→0"""
    real_labels = real_preds.new_ones(real_preds.shape)
    fake_labels = fake_preds.new_zeros(fake_preds.shape)
    loss = (bce_loss(real_preds, real_labels) + bce_loss(fake_preds, fake_labels)) / 2
    return loss


def generator_loss(fake_preds, fake_imgs, real_imgs, lambda_l1=10.0):
    """G wants D to think fakes are real + stay close to input"""
    adv_loss = bce_loss(fake_preds, fake_preds.new_ones(fake_preds.shape))
    rec_loss = l1_loss(fake_imgs, real_imgs)
    return adv_loss + lambda_l1 * rec_loss 
