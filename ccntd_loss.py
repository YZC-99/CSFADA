import torch.nn as nn
import torch

kl_loss = nn.KLDivLoss(reduction='none')


def ccntd_loss(model,source_model,data,train_ulb_loader,iteritor_train_ulb,args):
    model.train()
    image, label = data['image'].cuda(), data['label'].cuda()
    output = model(image)

    with torch.no_grad():
        output_in_source_model = source_model(image)
        output_in_source_model_soft = torch.softmax(output_in_source_model / args.temp, dim=1)
    # NTD
    label_index = torch.cat((label.unsqueeze(1), label.unsqueeze(1)), dim=1)
    outputs_soft_log = torch.log_softmax(output, dim=1)
    outputs_soft_none_target_log = outputs_soft_log[label_index == 0]
    output_in_source_model_soft_none_target = output_in_source_model_soft[label_index == 0]
    labeled_loss_value = kl_loss(outputs_soft_none_target_log, output_in_source_model_soft_none_target)
    labeled_kl_loss_value = labeled_loss_value

    # CNTD
    try:
        data_ulb = next(iteritor_train_ulb)
    except StopIteration:
        iteritor_train_ulb = iter(train_ulb_loader)
        data_ulb = next(iteritor_train_ulb)
    image_ulb = data_ulb['image'].cuda()
    output_ulb = model(image_ulb)
    with torch.no_grad():
        source_output_ulb = source_model(image_ulb)
    source_output_ulb_soft = torch.softmax(source_output_ulb / args.temp, dim=1)
    outputs_soft_ulb_log = torch.log_softmax(output_ulb, dim=1)

    pred_ulb = torch.argmax(output_ulb, dim=1)
    pred_ulb_source = torch.argmax(source_output_ulb, dim=1)
    mask = ((pred_ulb == 0) & (pred_ulb_source == 0)).unsqueeze(1).repeat(1, 2, 1, 1)
    outputs_none_target_soft_ulb_log = outputs_soft_ulb_log[mask]
    output_in_source_model_soft_none_target_ulb = source_output_ulb_soft[mask]

    unlabeled_loss_value = kl_loss(outputs_none_target_soft_ulb_log, output_in_source_model_soft_none_target_ulb)

    kl_loss_value = labeled_kl_loss_value.mean() + unlabeled_loss_value.mean()

    return kl_loss_value