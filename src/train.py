import torch
import os
import numpy as np
import random
import yaml
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm
from rpn_layer import FasterRCNN
from load_data import load_data
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# TRAIN MODEL FUNCTION
def train(args):

    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    dataset_config = config['dataset_params']
    model_config = config['model_params']
    train_config = config['train_params']

    # Seed initialization
    seed = train_config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)

    list_folder = [os.path.join('../dataset', folder) for folder in os.listdir('../dataset')]
    dataloader_dict = load_data(list_folder)

    model = FasterRCNN(model_config, dataset_config['num_classes']).to(device)

    optimizer = torch.optim.SGD(lr=train_config['lr'],
                                params=filter(lambda p: p.requires_grad, model.parameters()),
                                weight_decay=5E-4,
                                momentum=0.9)
    scheduler = MultiStepLR(optimizer, milestones=train_config['lr_steps'], gamma=0.1)

    num_epochs = train_config['num_epochs']
    acc_steps = train_config['acc_steps']

    for epoch in range(num_epochs):
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
                step_count = 1
            else:
                model.eval()

            rpn_classification_losses = []
            rpn_localization_losses = []
            frcnn_classification_losses = []
            frcnn_localization_losses = []

            for batch in tqdm(dataloader_dict[phase]):
                im = batch['image'].float().to(device)
                target = batch['target']
                target['bboxes'] = target['bboxes'].float().to(device)
                target['labels'] = target['labels'].long().to(device)

                with torch.set_grad_enabled(phase == "train"):
                    rpn_output, frcnn_output = model(im, target)
                    rpn_loss = rpn_output['rpn_classification_loss'] + rpn_output['rpn_localization_loss']
                    frcnn_loss = frcnn_output['frcnn_classification_loss'] + frcnn_output['frcnn_localization_loss']
                    loss = (rpn_loss + frcnn_loss) / acc_steps

                    if phase == "train":
                        loss.backward()
                        if step_count % acc_steps == 0:
                            optimizer.step()
                            optimizer.zero_grad()
                        step_count += 1

                rpn_classification_losses.append(rpn_output['rpn_classification_loss'].item())
                rpn_localization_losses.append(rpn_output['rpn_localization_loss'].item())
                frcnn_classification_losses.append(frcnn_output['frcnn_classification_loss'].item())
                frcnn_localization_losses.append(frcnn_output['frcnn_localization_loss'].item())

            print(f'Finished epoch {epoch} phase {phase}')

            loss_output = f'{phase} Information: '
            loss_output += f'RPN Classification Loss: {np.mean(rpn_classification_losses):.4f} '
            loss_output += f'| RPN Localization Loss: {np.mean(rpn_localization_losses):.4f} '
            loss_output += f'| FRCNN Classification Loss: {np.mean(frcnn_classification_losses):.4f} '
            loss_output += f'| FRCNN Localization Loss: {np.mean(frcnn_localization_losses):.4f}'
            print(loss_output)

            if phase == "train":
                scheduler.step()
                torch.save(model.state_dict(), os.path.join(train_config['task_name'], train_config['ckpt_name']))

    print('Done Training...')


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description=' faster R CNN arguments')
    parser.add_argument('--config', dest='config_path', default='config.yaml', type=str)
    args = parser.parse_args()
    train(args)






