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


class ConfidenceChecker:
    def __init__(self, gamma=0):
        """
        Initialize the unconfident detector
        :param gamma: Floating-point numbers, lower bound percentiles (e.g. 0.05 for 5%), the smallest 5% are considered unconfident samples
        """
        self.gamma = gamma
        self.sorted_values = SortedList()
    
    def add_value(self, value):
        """
        Add a new value to the sorted list
        :param value: floating-point number, a new value
        """
        self.sorted_values.add(value)
    
    def is_last_element_unconfident(self, last_value):
        """
        Detect if the last element is not confident
        :param last_value: floating-point number, the value of the last element
        :return: Boolean, whether the last element deviates significantly from the primary data, is an overly unconfident sample
        """
        if len(self.sorted_values) == 0 or self.gamma == 0:
            return False  # If there are no remaining elements or if the confidence level is 0, it cannot be judged
        
        # Calculate the lower bound percentile
        lower_bound = self.sorted_values[int(len(self.sorted_values) * self.gamma)]
        
        # Determines if the last element is smaller than the nether
        return last_value < lower_bound

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
    parser.add_argument('--data-root', dest='data_root', type=str, default='/data1/yangjialong/TDA/data/', help='Path to the datasets directory. Default is ./dataset/')
    parser.add_argument('--backbone', dest='backbone', type=str, default='ViT-B/16', choices=['RN50', 'ViT-B/16'], help='CLIP model backbone to use: RN50 or ViT-B/16.')
    parser.add_argument('--log-path', dest='log_path', type=str, default='./log', help='Path to the log file.')
    args = parser.parse_args()
    return args


def run_test_dota(params, loader, clip_model, clip_weights, dota_model, logger):
    recent_sample_count = 1000
    unconfident_num = 0
    fusion_accuracies, dota_accuracies, clip_accuracies = [], [], []
    # It is used to store the maximum value of each sample feature and sort it to determine whether it is a sample with high uncertainty
    # The higher the value, the lower the uncertainty we assume here
    entropy_list = []
    # Initialize the unconfident detector, gamma is the proportion of the true label obtained
    checker = ConfidenceChecker(params['gamma'])
    with torch.no_grad():
        for i, (images, target) in enumerate(tqdm(loader, desc='Processed test images: ')):
             # When data augmentation is used, the top 10% of enhanced images are selected to train the model
            image_features, clip_logits, loss, prob_map, pred = get_clip_logits_aug(images, clip_model, clip_weights)
            pred, target, prop_entropy = torch.tensor(pred).cuda(), target.cuda(), get_entropy(loss, clip_weights)
            dota_logits = dota_model.predict(image_features.mean(0).unsqueeze(0))

            # Put the maximum value in the prop_entropy into the checker list after performing the softmax operation
            entropy_list.append(prop_entropy)
            softmax_output = F.softmax(clip_logits[0], dim=-1)
            max_logit = torch.max(softmax_output)
            checker.add_value(max_logit)

            # Choose a smaller weight, so that model relies more on the original clip initially
            dota_weights = torch.clamp(params['rho'] * dota_model.c.mean() / image_features.size(0), max=params['eta'])   
            # Clip and Dota prediction weights are added to form the final prediction
            final_logits = clip_logits + dota_weights*dota_logits

            # Calculate the prediction accuracy of mixed weights, dota weights, clip weights and add them to the list
            fusion_acc, dota_acc, clip_acc = cls_acc(final_logits, target), cls_acc(dota_logits, target), cls_acc(clip_logits, target)
            fusion_accuracies.append(fusion_acc)
            dota_accuracies.append(dota_acc)
            clip_accuracies.append(clip_acc)

            # Determine whether the sample is unconfident
            unconfident = checker.is_last_element_unconfident(max_logit)
            if unconfident: # If it is unconfident, it is fitted with a real label
                unconfident_num = unconfident_num+1
                one_hot_target = torch.nn.functional.one_hot(target, num_classes=prob_map.shape[1]).repeat_interleave(prob_map.shape[0], dim=0).half()
                dota_model.fit(image_features, one_hot_target)
            # For samples that did not use real labels, we used the weights predicted by the clip as the updated weights for the corresponding categories
            else :
                dota_model.fit(image_features, prob_map)

            # Update the inverse matrix
            dota_model.update()                
            # Print the information
            if (i + 1) % recent_sample_count == 0:
                recent_fusion_accuracy = sum(fusion_accuracies[-recent_sample_count:]) / recent_sample_count
                recent_dota_accuracy = sum(dota_accuracies[-recent_sample_count:]) / recent_sample_count
                recent_clip_accuracy = sum(clip_accuracies[-recent_sample_count:]) / recent_sample_count
                logger.info(
                    "Last {} samples' accuracies - Fusion: {:.2f}%, DOTA: {:.2f}%, CLIP: {:.2f}% | "
                    "Overall accuracies - Fusion: {:.2f}%, DOTA: {:.2f}%, CLIP: {:.2f}%, unconfident sample number: {:.2f}".format(
                        recent_sample_count, recent_fusion_accuracy, recent_dota_accuracy, recent_clip_accuracy,
                        sum(fusion_accuracies) / len(fusion_accuracies),
                        sum(dota_accuracies) / len(dota_accuracies),
                        sum(clip_accuracies) / len(clip_accuracies),
                        unconfident_num
                    )
                )

        return {
            'overall_fusion_accuracy': sum(fusion_accuracies) / len(fusion_accuracies),
            'overall_dota_accuracy': sum(dota_accuracies) / len(dota_accuracies),
            'overall_clip_accuracy': sum(clip_accuracies) / len(clip_accuracies),
            'recent_fusion_accuracy': sum(fusion_accuracies[-recent_sample_count:]) / min(recent_sample_count, len(fusion_accuracies)),
            'recent_dota_accuracy': sum(dota_accuracies[-recent_sample_count:]) / min(recent_sample_count, len(dota_accuracies)),
            'recent_clip_accuracy': sum(clip_accuracies[-recent_sample_count:]) / min(recent_sample_count, len(clip_accuracies)),
            'unconfident sample number': unconfident_num,
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
        dota_model = DOTA(cfg, input_shape=clip_weights.shape[0], num_classes=clip_weights.shape[1], clip_weights=clip_weights.clone())
        dota_model.eval()

        acc = run_test_dota(cfg, test_loader, clip_model, clip_weights, dota_model, logger)
        logger.info(acc)


if __name__ == "__main__":
    main()

