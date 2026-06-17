import random
import os

import argparse
from datetime import datetime
import torch
from tqdm import tqdm
import clip
from utils import *
from torch import nn
import logging
import bisect
from sortedcontainers import SortedList
import numpy as np
import torch.backends.cudnn as cudnn
import torch.nn.functional as F


def setup_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

class DOTA(nn.Module):
    def __init__(self, cfg, input_shape, num_classes, clip_weights, streaming_update_Sigma=True):
        super(DOTA, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.streaming_update_Sigma = streaming_update_Sigma
        self.epsilon = cfg['epsilon']
        self.mu = clip_weights.T.to(self.device)  # initialize mu with clip_weights
        self.c = torch.ones(num_classes, dtype=torch.float32).to(self.device)
        self.Sigma = cfg['sigma'] * torch.eye(input_shape, dtype=torch.float32).repeat(num_classes, 1, 1).to(self.device)
        self.overall_Sigma = torch.mean(self.Sigma, dim=0)
        self.Lambda = torch.pinverse(self.overall_Sigma.double()).to(self.device).half()

    # Update the covariance and the mean for the corresponding category
    def fit(self, x, y):
        x = x.to(self.device)
        y = y.to(self.device)  # y is now a probability distribution (soft labels)
        with torch.no_grad():
            sum_weights = torch.sum(y, dim=0)  
            weighted_x = torch.matmul(y.T, x)  
            new_mu = (weighted_x + self.c.unsqueeze(1) * self.mu) / (sum_weights.unsqueeze(1) + self.c.unsqueeze(1)) 
            new_c = self.c + sum_weights

            # Update the covariance matrix for each category
            if self.streaming_update_Sigma:
                x_minus_mu = x.unsqueeze(1) - self.mu.unsqueeze(0)  # Shape: (batch_size, num_classes, input_shape)
                weighted_x_minus_mu = y.unsqueeze(2) * x_minus_mu  # Shape: (batch_size, num_classes, input_shape)
                delta = torch.einsum('bji,bjk->jik', weighted_x_minus_mu, x_minus_mu)  # Shape: (num_classes, input_shape, input_shape)
                self.Sigma = (self.c[:, None, None] * self.Sigma + delta) / (self.c[:, None, None] + sum_weights[:, None, None])

            # Update the total covariance matrix, mean matrix, and count sections
            self.overall_Sigma = torch.mean(self.Sigma, dim=0)
            self.mu = new_mu
            self.c = new_c
            
    # Update the inverse matrix to include a small identity matrix when calculating the inverse matrix to ensure that it is full rank
    def update(self):
        self.Lambda = torch.inverse(
            (1 - self.epsilon) * self.overall_Sigma + self.epsilon * torch.eye(self.input_shape).to(
            self.device)).half()

    # Calculate the results of Dota predictions
    def predict(self, X):
        X = X.to(self.device)
        with torch.no_grad():
            Lambda = self.Lambda
            M = self.mu.transpose(1, 0).half()
            W = torch.matmul(Lambda, M)  
            c = 0.5 * torch.sum(M * W, dim=0)
            scores = torch.matmul(X, W) - c
            return scores

def get_arguments():
    """Get arguments of the test-time adaptation."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', default='configs', help='settings of TDA on specific dataset in yaml format.')
    parser.add_argument('--datasets', dest='datasets', default='I', type=str, help="Datasets to process, separated by a slash (/). Example: I/A/V/R/S")
    parser.add_argument('--data-root', dest='data_root', type=str, default='./dataset/', help='Path to the datasets directory. Default is ./dataset/')
    parser.add_argument('--backbone', dest='backbone', type=str, default='ViT-B/16', choices=['ViT-B/16'], help='CLIP model backbone to use: ViT-B/16.')
    parser.add_argument('--log-path', dest='log_path', type=str, default='./log', help='Path to the log file.')
    args = parser.parse_args()
    return args


def run_test_dota(params, loader, clip_model, clip_weights, dota_model, logger):
    recent_sample_count = 1000
    fusion_accuracies = []
    # It is used to store the maximum value of each sample feature and sort it to determine whether it is a sample with high uncertainty
    # Initialize the unconfident detector, gamma is the proportion of the true label obtained
    with torch.no_grad():
        for i, (images, target) in enumerate(tqdm(loader, desc='Processed test images: ')):
             # When data augmentation is used, the top 10% of enhanced images are selected to train the model
            image_features, clip_logits, loss, prob_map, pred = get_clip_logits_aug(images, clip_model, clip_weights)
            pred, target, prop_entropy = torch.tensor(pred).cuda(), target.cuda(), get_entropy(loss, clip_weights)
            dota_logits = dota_model.predict(image_features.mean(0).unsqueeze(0))

            # Choose a smaller weight, so that model relies more on the original clip initially
            dota_weights = torch.clamp(params['rho'] * dota_model.c.mean() / image_features.size(0), max=params['eta'])   
            # Clip and Dota prediction weights are added to form the final prediction
            final_logits = clip_logits + dota_weights*dota_logits

            # Calculate the prediction accuracy of mixed weights, dota weights, clip weights and add them to the list
            fusion_acc = cls_acc(final_logits, target)
            fusion_accuracies.append(fusion_acc)

            dota_model.fit(image_features, prob_map)

            # Update the inverse matrix
            dota_model.update()                
            # Print the information
            if (i + 1) % recent_sample_count == 0:
                recent_fusion_accuracy = sum(fusion_accuracies[-recent_sample_count:]) / recent_sample_count
                logger.info(
                    "Last {} samples' accuracies - Fusion: {:.2f}% | "
                    "Overall accuracies - Fusion: {:.2f}% ".format(
                        recent_sample_count, recent_fusion_accuracy,
                        sum(fusion_accuracies) / len(fusion_accuracies),
                    )
                )

        return {
            'overall_fusion_accuracy': sum(fusion_accuracies) / len(fusion_accuracies),
            'recent_fusion_accuracy': sum(fusion_accuracies[-recent_sample_count:]) / min(recent_sample_count, len(fusion_accuracies)),
        }


def main():
    args = get_arguments()
    config_path = args.config
    clip_model, preprocess = clip.load(args.backbone)
    clip_model.eval()
    datasets = args.datasets.split('/')
    for dataset_name in datasets:
        # Set random seed
        setup_seeds(1)
        # Prepare logs and other content to facilitate printout information
        date = datetime.now().strftime("%b%d_%H-%M-%S")
        backbone_safe = args.backbone.replace('/', '_') 
        group_name = f"{backbone_safe}_{dataset_name}_{date}"
        logging.basicConfig(filename=os.path.join(args.log_path, group_name), level=logging.INFO, format='%(asctime)s %(message)s')
        logger = logging.getLogger()        
        logger.info(f"Processing {dataset_name} dataset.")

        # Obtain the hyperparameter information of the dataset
        cfg = get_config_file(config_path, dataset_name)
        logger.info("\nRunning dataset configurations:")
        logger.info(cfg)

        test_loader, classnames, template = build_test_data_loader(dataset_name, args.data_root, preprocess)
        clip_weights = clip_classifier(classnames, template, clip_model)
        tensor_matrix = torch.full((clip_weights.shape[0], clip_weights.shape[1]), 0.001)
        dota_model = DOTA(cfg, input_shape=clip_weights.shape[0], num_classes=clip_weights.shape[1], clip_weights=tensor_matrix)
        dota_model.eval()

        acc = run_test_dota(cfg, test_loader, clip_model, clip_weights, dota_model, logger)
        logger.info(acc)


if __name__ == "__main__":
    main()

