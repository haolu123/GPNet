import torch
import torch.nn as nn
import torch.nn.functional as F

class SparseAutoencoder(nn.Module):
    def __init__(self, n_input, n_output):
        super(SparseAutoencoder, self).__init__()
        
        # Encoder layers
        self.encoder1 = nn.Linear(n_input, 500)
        self.encoder2 = nn.Linear(500, 300)
        
        # Decoder layers
        self.decoder1 = nn.Linear(300, 500)
        self.decoder2 = nn.Linear(500, n_input)

        # classifier layers
        self.fc1 = nn.Linear(300, n_output)
        self.dropout = nn.Dropout(0.5)


    def forward(self, x):
        # Encoding
        x = F.relu(self.encoder1(x))
        x = F.relu(self.encoder2(x))
        y = x  # Save the encoding result for later use
        # Decoding
        x = F.relu(self.decoder1(x))
        x = self.decoder2(x)  # Output of decoder without activation function
        # classifier
        z = self.dropout(y)
        z = self.fc1(z)

        return y,x,z  # Return the encoding, decoding and classification results

def kl_divergence(rho, rho_hat):
    rho = torch.tensor(rho)
    return rho * torch.log(rho / rho_hat) + (1 - rho) * torch.log((1 - rho) / (1 - rho_hat))

def sparse_autoencoder_loss(model, model_output, classifier_output, classifier_label, original_data, rho, alpha, beta,gama, normalized_weights):
    # Reconstruction loss (Mean Squared Error)
    mse_loss = F.mse_loss(model_output, original_data)
    criterion = nn.CrossEntropyLoss(weight=normalized_weights)
    ce_loss = criterion(classifier_output, classifier_label)
    # Sparsity loss
    sparsity_loss = 0
    # for layer in [model.encoder1, model.encoder2]:
    #     rho_hat = torch.mean(F.sigmoid(layer.weight), 1)  # Average activation of hidden units
    #     sparsity_loss += torch.sum(kl_divergence(rho, rho_hat))
    
    # Total loss
    total_loss = gama * mse_loss + beta * sparsity_loss + alpha * ce_loss
    return total_loss, mse_loss, sparsity_loss, ce_loss

