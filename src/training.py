import torch
from torch.nn import MSELoss, L1Loss


def train_on_batch_discriminator(discriminator, imgs, optimiser, device):

    images, labels, fake_labels = imgs

    batch_size = images.shape[0]

    h, w = images.shape[2:4]
    disc_patch = (1, h // 2 ** 2, w // 2 ** 2)

    input_images = torch.cat([images, images], axis=0)
    input_labels = torch.cat([labels, fake_labels], axis=0)

    # INFER BATCH_SIZE AND DISC_PATCH FROM INPUTS!

    valid = torch.ones((batch_size,) + disc_patch)
    fake = torch.zeros((batch_size,) + disc_patch)

    targets = torch.cat([valid, fake], axis=0).to(device)

    outputs = discriminator(input_images, input_labels)

    # clear previous gradients
    optimiser.zero_grad()

    # forward pass
    d_loss = MSELoss()(outputs, targets)

    # calculate gradients
    d_loss.backward()

    # descent step
    optimiser.step()

    return d_loss


def train_on_batch_roi_gan(roi_gan, data, optimiser, device):

    images, labels, fake_labels, bboxes = data

    batch_size = images.shape[0]

    num_roi = bboxes.shape[0]

    input_images = torch.cat([images, images], axis=0)
    input_labels = torch.cat([labels, fake_labels], axis=0)
    input_bboxes = torch.cat([bboxes, bboxes], axis=0)

    input_bboxes[num_roi:, 0] += batch_size  # increment image id
    input_bboxes[:, 1:] = input_bboxes[:, 1:] / 2  # correct for pooling

    valid = torch.ones((num_roi, 1))
    fake = torch.zeros((num_roi, 1))

    targets = torch.cat([valid, fake], axis=0).to(device)

    # clear previous gradients
    optimiser.zero_grad()

    # forward pass
    validity = roi_gan(input_images, input_labels, input_bboxes)

    # calculate loss
    d_loss = MSELoss()(validity, targets)

    # backpropagate
    d_loss.backward()

    # descent step
    optimiser.step()

    return d_loss


def train_on_batch_combined(models, data, optimiser, lamdbas, device):

    # clear previous gradients
    optimiser.zero_grad()

    generator, discriminator, roi_gan = models
    images, labels, bboxes = data

    batch_size = images.shape[0]

    h, w = images.shape[2:4]
    disc_patch = (1, h // 2 ** 2, w // 2 ** 2)

    bboxes[:, 1:] = bboxes[:, 1:] / 2  # correct for pooling

    fake_labels = generator(images)

    # Discriminators determines validity of translated images
    num_roi = bboxes.shape[0]
    valid_roi = torch.ones((num_roi, 1)).to(device)
    validity_roi = roi_gan(images, fake_labels, bboxes)

    # Discriminators determines validity of translated images
    valid_patch = torch.ones((batch_size,) + disc_patch).to(device)
    validity_patch = discriminator(images, fake_labels)

    # Best results with (0.2, 1, 5)

    lambda_dis, lambda_roi, lambda_gen = lamdbas

    g_loss = lambda_dis * MSELoss()(validity_patch, valid_patch) + \
             lambda_roi * MSELoss()(validity_roi, valid_roi) + \
             lambda_gen * L1Loss()(labels, fake_labels)

    # calculate gradients
    g_loss.backward()

    # descent step
    optimiser.step()

    return g_loss
